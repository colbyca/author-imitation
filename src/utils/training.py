"""Shared training helpers built on top of Hugging Face Trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase, Trainer, TrainingArguments


class BertClassifierTrainer:
    """Convenience wrapper around Hugging Face Trainer for BERT-style classifiers."""

    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizerBase,
        *,
        training_args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable] = None,
        data_collator: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.compute_metrics = compute_metrics or self._default_metrics
        self.data_collator = data_collator or DataCollatorWithPadding(tokenizer)

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

    @staticmethod
    def _default_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="weighted",
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def train(self, **kwargs):
        return self.trainer.train(**kwargs)

    def evaluate(self, **kwargs):
        return self.trainer.evaluate(**kwargs)

    def predict(self, *args, **kwargs):
        return self.trainer.predict(*args, **kwargs)

    def save_model(self, output_dir: Optional[Path] = None) -> None:
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(str(output_dir or self.training_args.output_dir))

    def save_state(self) -> None:
        self.trainer.save_state()

    @property
    def state(self):
        return self.trainer.state

    @property
    def model_dir(self) -> str:
        return self.training_args.output_dir

