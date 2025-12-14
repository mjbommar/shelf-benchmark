"""Training CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from shelf.train import train_transformer_classifier

app = typer.Typer(name="train", help="Train / fine-tune classifiers")
console = Console()


@app.command("classify")
def cmd_train_classify(
    task: Annotated[str, typer.Argument(help="Classification task name")],
    model: Annotated[str, typer.Argument(help="HF base model id (e.g., roberta-base)")],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to save checkpoints",
        ),
    ],
    train_split: Annotated[
        str, typer.Option("--train-split", help="Dataset split for training")
    ] = "train",
    eval_split: Annotated[
        str, typer.Option("--eval-split", help="Dataset split for validation")
    ] = "validation",
    max_length: Annotated[
        int, typer.Option("--max-length", help="Max sequence length")
    ] = 512,
    epochs: Annotated[
        float, typer.Option("--epochs", "-e", help="Number of training epochs")
    ] = 3.0,
    lr: Annotated[
        float, typer.Option("--lr", help="Learning rate")
    ] = 2e-5,
    weight_decay: Annotated[
        float, typer.Option("--weight-decay", help="Weight decay")
    ] = 0.01,
    warmup_ratio: Annotated[
        float, typer.Option("--warmup-ratio", help="LR warmup ratio")
    ] = 0.1,
    train_bs: Annotated[
        int,
        typer.Option(
            "--train-batch-size",
            "-b",
            help="Per-device train batch size",
        ),
    ] = 8,
    eval_bs: Annotated[
        int, typer.Option("--eval-batch-size", help="Per-device eval batch size")
    ] = 8,
    grad_accum: Annotated[
        int,
        typer.Option(
            "--grad-accum",
            help="Gradient accumulation steps",
        ),
    ] = 1,
    logging_steps: Annotated[
        int, typer.Option("--logging-steps", help="Logging frequency")
    ] = 50,
    save_total_limit: Annotated[
        int,
        typer.Option(
            "--save-total-limit",
            help="Maximum checkpoints to keep",
        ),
    ] = 2,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    fp16: Annotated[
        bool, typer.Option("--fp16/--no-fp16", help="Enable FP16 when available")
    ] = True,
    freeze_base: Annotated[
        bool,
        typer.Option(
            "--freeze-base",
            help="Train only the classification head (freeze encoder)",
        ),
    ] = False,
    trust_remote_code: Annotated[
        bool,
        typer.Option(
            "--trust-remote-code",
            help="Allow remote code when loading HF models",
        ),
    ] = False,
):
    """Fine-tune a transformers classifier on a SHELF task."""
    try:
        result = train_transformer_classifier(
            task_name=task,
            model_name=model,
            output_dir=output_dir,
            train_split=train_split,
            eval_split=eval_split,
            max_length=max_length,
            num_train_epochs=epochs,
            learning_rate=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=eval_bs,
            gradient_accumulation_steps=grad_accum,
            logging_steps=logging_steps,
            save_total_limit=save_total_limit,
            seed=seed,
            fp16=fp16,
            freeze_base=freeze_base,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        console.print(f"[red]Training failed:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]Training complete![/green] Saved to {result.output_dir}")
    console.print("Best checkpoint:", result.best_checkpoint)
    if result.metrics:
        console.print("Validation metrics:")
        for k, v in result.metrics.items():
            console.print(f"  {k}: {v:.4f}")

