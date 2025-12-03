"""Train the augmented discriminator model (D_ag).

This script fine-tunes a BERT sequence classification model on the D_ag dataset
produced by "src.dag_preprocess". The model classifies text as:
- Target author
- Distractor authors
- Generated text

It evaluates the model at regular checkpoint intervals and saves the best-performing
checkpoint together with training metrics for later inspection.
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
    setup_logging,
)

LOGGER = logging.getLogger("dag_train")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    cfg = CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=cfg.paths.dag_dataset_dir,
        help="Directory containing the processed D_ag dataset.",
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
        default=cfg.paths.dag_model_dir,
        help="Directory to save the fine-tuned model and checkpoints.",
    )
    parser.add_argument(
        "--model-name",
        default=cfg.dag_training.model_name,
        help="Pre-trained BERT model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=cfg.dag_training.num_train_epochs,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=cfg.dag_training.learning_rate,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=cfg.dag_training.per_device_train_batch_size,
        help="Per-device batch size for training.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=cfg.dag_training.per_device_eval_batch_size,
        help="Per-device batch size for evaluation.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=cfg.dag_training.gradient_accumulation_steps,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=cfg.dag_training.warmup_steps,
        help="Number of warmup steps for the LR scheduler.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=cfg.dag_training.weight_decay,
        help="Weight decay coefficient.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=cfg.dag_training.logging_steps,
        help="Logging interval in training steps.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=cfg.dag_training.eval_steps,
        help="Evaluation interval in training steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=cfg.dag_training.save_steps,
        help="Checkpoint save interval in training steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=cfg.dag_training.save_total_limit,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=cfg.dag_training.max_steps,
        help="Optional limit on the number of training steps.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=cfg.dag_training.fp16,
        help="Enable FP16 mixed precision training (if supported).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=cfg.dag_training.bf16,
        help="Enable BF16 mixed precision training (if supported).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=cfg.dag_training.gradient_checkpointing,
        help="Enable gradient checkpointing to reduce memory usage.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=cfg.dag_training.dataloader_num_workers,
        help="Number of data loading workers.",
    )
    parser.add_argument(
        "--report-to",
        nargs="*",
        default=cfg.dag_training.report_to,
        help="Integrations to report metrics to (e.g. wandb).",
    )
    parser.add_argument(
        "--metric-for-best-model",
        default=cfg.dag_training.metric_for_best_model,
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


def read_dag_label_mapping(metadata_dir: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Read label mapping metadata for D_ag dataset."""
    mapping_path = metadata_dir / "dag_label_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"D_ag label mapping file not found at {mapping_path}.")
    
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    raw_label2id = payload.get("label2id", {})
    raw_id2label = payload.get("id2label", {})
    if not raw_label2id or not raw_id2label:
        raise ValueError(f"Invalid label mapping contents in {mapping_path}.")
    
    label2id = {str(name): int(idx) for name, idx in raw_label2id.items()}
    id2label = {int(idx): str(name) for idx, name in raw_id2label.items()}
    return label2id, id2label


def build_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> Tuple[BertForSequenceClassification, BertTokenizerFast]:
    """Build BERT model and tokenizer for sequence classification."""
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label={idx: label for idx, label in id2label.items()},
    )
    return model, tokenizer


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute classification metrics."""
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
    """Main training function."""
    args = parse_args()
    setup_logging(args.logging_level)
    LOGGER.info("Arguments: %s", args)
    
    set_seed(args.seed)
    
    # Load dataset
    dataset = load_dataset_dict(args.dataset_dir)
    train_dataset = ensure_split(dataset, "train")
    eval_dataset = ensure_split(dataset, args.validation_split)
    
    # Load label mapping
    label2id, id2label = read_dag_label_mapping(args.metadata_dir)
    num_labels = len(label2id)
    
    LOGGER.info(
        "Loaded dataset with %d labels: %s",
        num_labels,
        ", ".join(f"{label} ({idx})" for label, idx in sorted(label2id.items(), key=lambda x: x[1])),
    )
    
    # Build model and tokenizer
    model, tokenizer = build_model_and_tokenizer(
        args.model_name,
        num_labels,
        label2id,
        id2label,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # Prepare output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
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
        eval_strategy="no",
        eval_steps=args.eval_steps,
        save_strategy="no",
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
    
    # Create trainer
    classifier = BertClassifierTrainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    LOGGER.info(
        "Starting training on %d samples with %d evaluation samples.",
        len(train_dataset),
        len(eval_dataset),
    )
    
    # Train
    train_result = classifier.train(resume_from_checkpoint=args.resume_from_checkpoint)
    classifier.save_model()
    
    # Collect metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    
    # Evaluate best model
    LOGGER.info("Evaluating the best model on the %s split...", args.validation_split)
    eval_metrics = classifier.evaluate()
    metrics.update({f"final_{key}": value for key, value in eval_metrics.items()})
    metrics["eval_samples"] = len(eval_dataset)
    
    # Save metrics
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved training metrics to %s", metrics_path)
    
    # Generate classification report if requested
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

