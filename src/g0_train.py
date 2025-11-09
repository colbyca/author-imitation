"""Train the baseline GPT-2 generator (G0) on the target author's corpus.

This script loads the tokenised dataset produced by :mod:`src.preprocess`
and fine-tunes a GPT-2 model using Hugging Face's ``Trainer`` API. The
resulting model is saved to the location defined in :mod:`src.config`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_from_disk
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import CONFIG
from utils import setup_logging

LOGGER = logging.getLogger("g0_train")


def parse_args() -> argparse.Namespace:
    cfg = CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=cfg.paths.generator_dataset_dir,
        help="Path to the tokenised generator dataset (default: config value).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cfg.paths.g0_model_dir,
        help="Directory to store the fine-tuned model (default: config value).",
    )
    parser.add_argument(
        "--model-name",
        default=cfg.tokenizers.gpt2_model_name,
        help="Base GPT-2 checkpoint to fine-tune (default: config value).",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=cfg.g0_training.num_train_epochs,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=cfg.g0_training.per_device_train_batch_size,
        help="Batch size per device for training.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=cfg.g0_training.per_device_eval_batch_size,
        help="Batch size per device for evaluation.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=cfg.g0_training.gradient_accumulation_steps,
        help="Number of steps to accumulate gradients before an optimizer step.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=cfg.g0_training.learning_rate,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=cfg.g0_training.warmup_steps,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=cfg.g0_training.weight_decay,
        help="Weight decay coefficient.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=cfg.g0_training.logging_steps,
        help="Logging interval in steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=cfg.g0_training.save_steps,
        help="Checkpoint saving interval in steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=cfg.g0_training.save_total_limit,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=cfg.g0_training.eval_ratio,
        help="Fraction of the dataset reserved for evaluation (0 to disable).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=cfg.g0_training.max_steps,
        help="Limit on total training steps (overrides epochs when set).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=cfg.g0_training.fp16,
        help="Enable mixed precision training with float16 (if supported).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=cfg.g0_training.bf16,
        help="Enable mixed precision training with bfloat16 (if supported).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=cfg.g0_training.gradient_checkpointing,
        help="Enable gradient checkpointing to reduce memory usage.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=cfg.g0_training.dataloader_num_workers,
        help="Number of subprocesses for data loading.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.random_seed,
        help="Random seed for reproducibility.",
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
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--report-to",
        nargs="*",
        default=cfg.g0_training.report_to,
        help="Tracking integrations to report metrics to (e.g. wandb).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation during training regardless of eval ratio.",
    )
    parser.add_argument(
        "--logging-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def load_generator_dataset(path: Path) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(
            f"Generator dataset not found at {path}. Run src/preprocess.py first."
        )

    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            LOGGER.warning(
                "Dataset at %s is a DatasetDict. Using the 'train' split for training.",
                path,
            )
            dataset = dataset["train"]
        else:
            raise ValueError(
                "DatasetDict loaded from disk does not contain a 'train' split."
            )

    if not isinstance(dataset, Dataset):
        raise TypeError("Expected a Dataset object after loading from disk.")

    if len(dataset) == 0:
        raise ValueError("Loaded generator dataset is empty.")

    required_columns = {"input_ids", "labels"}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(
            "Generator dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    return dataset


def split_dataset(dataset: Dataset, eval_ratio: float, seed: int) -> tuple[Dataset, Optional[Dataset]]:
    if eval_ratio is None or eval_ratio <= 0.0:
        return dataset, None

    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1 (exclusive) if provided.")

    dataset = dataset.shuffle(seed=seed)
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    return split["train"], split["test"]


def prepare_tokenizer(model_name: str) -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    args = parse_args()
    setup_logging(args.logging_level)
    LOGGER.info("Arguments: %s", args)

    set_seed(args.seed)

    dataset = load_generator_dataset(args.dataset_dir)
    train_dataset, eval_dataset = split_dataset(dataset, 0.0 if args.no_eval else args.eval_ratio, args.seed)

    LOGGER.info(
        "Dataset ready: train=%d samples%s",
        len(train_dataset),
        "" if eval_dataset is None else f", validation={len(eval_dataset)} samples",
    )

    tokenizer = prepare_tokenizer(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        use_mps_device=True if torch.backends.mps.is_available() else False,
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
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no" if eval_dataset is None else "epoch",
        save_strategy="epoch",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="loss",
        max_steps=args.max_steps if args.max_steps is not None else -1,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to or [],
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    LOGGER.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    if eval_dataset is not None:
        LOGGER.info("Evaluating on validation set...")
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
        metrics["eval_samples"] = len(eval_dataset)

    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved metrics to %s", metrics_path)

    trainer.save_state()

    LOGGER.info("Training complete. Model saved to %s", output_dir)


if __name__ == "__main__":
    main()
