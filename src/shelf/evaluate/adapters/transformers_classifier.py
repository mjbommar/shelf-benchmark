"""Transformers sequence classification adapter.

Wraps HuggingFace AutoModelForSequenceClassification models to implement the
TextClassifier protocol. This supports evaluating fully fine-tuned
classification checkpoints directly (no shallow head training).
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shelf.evaluate.adapters.protocols import TextClassifier


def _choose_device(preferred: str | None = None) -> str:
    """Select a device string."""
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TransformersSequenceClassifier(TextClassifier):
    """HuggingFace transformers classifier adapter."""

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer,
        *,
        model_name: str,
        device: str | None = None,
        max_length: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._model_name = model_name
        self.device = _choose_device(device)
        self.max_length = max_length or getattr(tokenizer, "model_max_length", 512)

        # Move model to device once
        self.model.to(self.device)
        self.model.eval()

        # Cache label mappings if present
        self.id2label: Dict[int, str] = getattr(self.model.config, "id2label", {})
        self.label2id: Dict[str, int] = getattr(self.model.config, "label2id", {})

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> "TransformersSequenceClassifier":
        """Load a fine-tuned checkpoint."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name_or_path,
            device=device,
            max_length=max_length,
        )

    def predict(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[str]:
        """Predict labels for a list of texts."""
        preds: List[str] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits
                label_ids = torch.argmax(logits, dim=-1).tolist()
                preds.extend(self._ids_to_labels(label_ids))
        return preds

    def _ids_to_labels(self, ids: List[int]) -> List[str]:
        """Map prediction ids to string labels."""
        if self.id2label:
            return [self.id2label.get(i, str(i)) for i in ids]
        return [str(i) for i in ids]

    @property
    def model_name(self) -> str:
        return self._model_name
