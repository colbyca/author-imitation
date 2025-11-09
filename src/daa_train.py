"""Train the multi-class BERT authorship attribution model (D_aa).

This script fine-tunes a BERT sequence classification model on the processed
Reuters dataset produced by :mod:`src.preprocess`. It evaluates the model at
regular checkpoint intervals and saves the best-performing checkpoint together
with training metrics for later inspection.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    TrainingArguments,
    set_seed,
)

from config import CONFIG
from utils import (
    BertClassifierTrainer,
    ensure_split,
    load_dataset_dict,
    read_label_mapping,
    setup_logging,
)

LOGGER = logging.getLogger("daa_train")


def parse_args() -> argparse.Namespace:
    cfg = CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=cfg.paths.discriminator_dataset_dir,
        help="Directory containing the processed discriminator dataset.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=cfg.paths.metadata_dir,
        help="Directory that stores preprocessing metadata (label mappings).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.paths.daa_model_dir,
        help="Directory to save the fine-tuned model and checkpoints.",
    )
    parser.add_argument(
        "--model-name",
        default=cfg.daa_training.model_name,
        help="Pre-trained BERT model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=cfg.daa_training.num_train_epochs,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=cfg.daa_training.learning_rate,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=cfg.daa_training.per_device_train_batch_size,
        help="Per-device batch size for training.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=cfg.daa_training.per_device_eval_batch_size,
        help="Per-device batch size for evaluation.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=cfg.daa_training.gradient_accumulation_steps,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=cfg.daa_training.warmup_steps,
        help="Number of warmup steps for the LR scheduler.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=cfg.daa_training.weight_decay,
        help="Weight decay coefficient.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=cfg.daa_training.logging_steps,
        help="Logging interval in training steps.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=cfg.daa_training.eval_steps,
        help="Evaluation interval in training steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=cfg.daa_training.save_steps,
        help="Checkpoint save interval in training steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=cfg.daa_training.save_total_limit,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=cfg.daa_training.max_steps,
        help="Optional limit on the number of training steps.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=cfg.daa_training.fp16,
        help="Enable FP16 mixed precision training (if supported).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=cfg.daa_training.bf16,
        help="Enable BF16 mixed precision training (if supported).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=cfg.daa_training.gradient_checkpointing,
        help="Enable gradient checkpointing to reduce memory usage.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=cfg.daa_training.dataloader_num_workers,
        help="Number of data loading workers.",
    )
    parser.add_argument(
        "--report-to",
        nargs="*",
        default=cfg.daa_training.report_to,
        help="Integrations to report metrics to (e.g. wandb).",
    )
    parser.add_argument(
        "--metric-for-best-model",
        default=cfg.daa_training.metric_for_best_model,
        help="Metric used to select the best checkpoint.",
    )
    parser.add_argument(
        "--greater-is-better",
        action="store_true",
        default=True,
        help="Indicate if a larger metric value is better.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Overwrite the model output directory if it already exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.random_seed,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--logging-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--validation-split",
        default="validation",
        help="Dataset split to use for evaluation (default: validation).",
    )
    parser.add_argument(
        "--classification-report",
        action="store_true",
        help="Generate and save a detailed classification report after training.",
    )
    return parser.parse_args()


def build_model_and_tokenizer(model_name: str, num_labels: int, label2id: Dict[str, int], id2label: Dict[int, str]) -> Tuple[BertForSequenceClassification, BertTokenizerFast]:
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label={idx: label for idx, label in id2label.items()},
    )
    return model, tokenizer


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
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


def main() -> None:
    args = parse_args()
    setup_logging(args.logging_level)
    LOGGER.info("Arguments: %s", args)

    set_seed(args.seed)

    dataset = load_dataset_dict(args.dataset_dir)
    train_dataset = ensure_split(dataset, "train")
    eval_dataset = ensure_split(dataset, args.validation_split)

    label2id, id2label = read_label_mapping(args.metadata_dir)
    num_labels = len(label2id)
    model, tokenizer = build_model_and_tokenizer(args.model_name, num_labels, label2id, id2label)

    data_collator = DataCollatorWithPadding(tokenizer)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to or [],
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        use_mps_device=torch.backends.mps.is_available(),
    )

    classifier = BertClassifierTrainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    LOGGER.info("Starting training on %d samples with %d evaluation samples.", len(train_dataset), len(eval_dataset))
    train_result = classifier.train(resume_from_checkpoint=args.resume_from_checkpoint)
    classifier.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    LOGGER.info("Evaluating the best model on the %s split...", args.validation_split)
    eval_metrics = classifier.evaluate()
    metrics.update({f"final_{key}": value for key, value in eval_metrics.items()})
    metrics["eval_samples"] = len(eval_dataset)

    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved training metrics to %s", metrics_path)

    if args.classification_report:
        LOGGER.info("Generating classification report for the %s split...", args.validation_split)
        predictions = classifier.predict(eval_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=-1)
        report = classification_report(
            predictions.label_ids,
            predicted_labels,
            target_names=[id2label[idx] for idx in sorted(id2label)],
            zero_division=0,
        )
        report_path = output_dir / "classification_report.txt"
        report_path.write_text(report, encoding="utf-8")
        LOGGER.info("Saved classification report to %s", report_path)

    classifier.save_state()
    LOGGER.info("Training complete. Model saved to %s", output_dir)


if __name__ == "__main__":
    main()

