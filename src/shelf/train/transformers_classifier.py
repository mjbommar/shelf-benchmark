"""Fine-tune transformers sequence classifiers on SHELF tasks.

This module adds a simple Trainer-based workflow for full-model or
head-only fine-tuning on SHELF classification tasks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from shelf.evaluate.metrics.classification import compute_classification_metrics
from shelf.evaluate.registry import get_task
from shelf.evaluate.tasks import TaskType

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Container for fine-tuning results."""

    output_dir: Path
    best_checkpoint: Path
    metrics: Dict[str, float]


def _freeze_base_model(model: AutoModelForSequenceClassification) -> None:
    """Freeze all base layers and leave classification head trainable."""
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze common classifier heads
    for attr in ("classifier", "score"):
        head = getattr(model, attr, None)
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True


def _prepare_label_maps(
    dataset: DatasetDict, label_field: str, label_space: Iterable[str] | None
) -> tuple[Dict[str, int], Dict[int, str]]:
    """Build label mappings from task spec or dataset."""
    if label_space:
        labels = list(label_space)
    else:
        labels = sorted({str(label) for label in dataset["train"][label_field]})

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def _encode_dataset(
    raw_ds: DatasetDict,
    tokenizer,
    *,
    text_field: str,
    label_field: str,
    label2id: Dict[str, int],
    max_length: int,
) -> DatasetDict:
    """Tokenize and encode datasets."""

    def _process(batch: Dict[str, Any]) -> Dict[str, Any]:
        tokenized = tokenizer(
            batch[text_field],
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = [label2id[str(lbl)] for lbl in batch[label_field]]
        return tokenized

    encoded = raw_ds.map(
        _process,
        batched=True,
        remove_columns=[c for c in raw_ds["train"].column_names if c != text_field],
    )
    encoded = encoded.with_format("torch")
    return encoded


def train_transformer_classifier(
    task_name: str,
    model_name: str,
    *,
    output_dir: str | Path,
    train_split: str = "train",
    eval_split: str = "validation",
    max_length: int = 512,
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    logging_steps: int = 50,
    save_total_limit: int = 2,
    seed: int = 42,
    fp16: bool | None = None,
    freeze_base: bool = False,
    trust_remote_code: bool = False,
) -> TrainResult:
    """Fine-tune a transformers classifier on a SHELF classification task.

    Args:
        task_name: Registry task name (e.g., "lcc_classification")
        model_name: Base HF model to fine-tune (e.g., "roberta-base")
        output_dir: Where to save checkpoints and logs
        train_split: Split used for training
        eval_split: Split used for validation
        max_length: Max sequence length for tokenization
        num_train_epochs: Number of training epochs
        learning_rate: Optimizer learning rate
        weight_decay: Weight decay
        warmup_ratio: Warmup ratio for scheduler
        per_device_train_batch_size: Train batch size per device
        per_device_eval_batch_size: Eval batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        logging_steps: Logging frequency
        save_total_limit: Max checkpoints to keep
        seed: Random seed
        fp16: Enable fp16 if True (auto-detect GPU when None)
        freeze_base: If True, train only the classification head
        trust_remote_code: Allow remote code for custom architectures
    """
    task_spec = get_task(task_name)
    if task_spec.task_type != TaskType.CLASSIFICATION:
        raise ValueError(f"Task {task_name} is not a classification task")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading dataset %s (%s)", task_spec.dataset_name, task_spec.dataset_config
    )
    raw_all = load_dataset(task_spec.dataset_name, task_spec.dataset_config)

    if isinstance(raw_all, DatasetDict) and all(
        s in raw_all for s in (train_split, eval_split)
    ):
        raw_ds = DatasetDict(
            {
                "train": raw_all[train_split],
                "validation": raw_all[eval_split],
            }
        )
    else:
        # Allow HF slicing expressions like "train[:1024]"
        logger.info("Loading dataset with split expressions: %s / %s", train_split, eval_split)
        train_ds = load_dataset(
            task_spec.dataset_name, task_spec.dataset_config, split=train_split
        )
        eval_ds = load_dataset(
            task_spec.dataset_name, task_spec.dataset_config, split=eval_split
        )
        raw_ds = DatasetDict({"train": train_ds, "validation": eval_ds})

    label2id, id2label = _prepare_label_maps(raw_ds, task_spec.label_field, task_spec.label_space)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=trust_remote_code
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        trust_remote_code=trust_remote_code,
    )

    if freeze_base:
        _freeze_base_model(model)

    encoded = _encode_dataset(
        raw_ds,
        tokenizer,
        text_field=task_spec.text_field,
        label_field=task_spec.label_field,
        label2id=label2id,
        max_length=max_length,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics_fn(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        y_pred = [id2label[int(i)] for i in preds]
        y_true = [id2label[int(i)] for i in labels]
        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            labels=list(id2label.values()),
            compute_per_class=False,
            compute_confusion_matrix=False,
        )
        # Keep core scalars only
        return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=seed,
        fp16=fp16 if fp16 is not None else torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    logger.info("Starting training for task %s with model %s", task_name, model_name)
    trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model()  # Saves to output_dir
    tokenizer.save_pretrained(output_dir)

    best_ckpt = Path(trainer.state.best_model_checkpoint or output_dir)

    return TrainResult(
        output_dir=output_dir,
        best_checkpoint=best_ckpt,
        metrics={k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
    )
