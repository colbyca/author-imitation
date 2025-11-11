"""Main iterative training loop for generator-discriminator co-evolution.

This script implements hard negative mining to iteratively improve both
the generator and discriminator models through multiple training iterations.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import CONFIG
from utils import (
    BertClassifierTrainer,
    build_word_prompts,
    collect_texts,
    ensure_split,
    generate_sequences,
    load_dataset_dict,
    load_generator,
    read_label_mapping,
    select_device,
    setup_logging,
)
from bert_evaluate import (
    classify_texts,
    compute_confidence_scores,
    format_distribution,
    save_confidence_scores,
    select_hard_negatives,
)

import sys
import subprocess

LOGGER = logging.getLogger("main_loop")


def load_generator_texts_metadata(metadata_dir: Path) -> List[str]:
    """Load the original texts used to create the generator dataset."""
    target_corpus_path = metadata_dir / "target_corpus_train.txt"
    if not target_corpus_path.exists():
        raise FileNotFoundError(
            f"Target corpus file not found at {target_corpus_path}. "
            "This is needed to track original texts for augmentation."
        )
    texts = target_corpus_path.read_text(encoding="utf-8").split("\n\n")
    return [text.strip() for text in texts if text.strip()]


def find_latest_generator_model(
    output_root: Path,
    iteration: int,
    g0_model_dir: Path,
) -> Tuple[Path, Optional[int]]:
    """Return the most recent generator checkpoint path and its source iteration."""
    if iteration == 0:
        return g0_model_dir, None

    for idx in range(iteration - 1, -1, -1):
        candidate = output_root / f"iteration_{idx}" / "generators" / f"G_{idx + 1}"
        if candidate.exists():
            return candidate, idx

    return g0_model_dir, None


def find_latest_discriminator_model(
    output_root: Path,
    iteration: int,
    d0_model_dir: Path,
) -> Tuple[Path, Optional[int]]:
    """Return the most recent discriminator checkpoint path and its source iteration."""
    if iteration == 0:
        return d0_model_dir, None

    for idx in range(iteration - 1, -1, -1):
        candidate = output_root / f"iteration_{idx}" / "discriminators" / f"D_{idx + 1}"
        if candidate.exists():
            return candidate, idx

    return d0_model_dir, None


def save_generator_texts_metadata(output_dir: Path, texts: List[str]) -> None:
    """Save the list of texts used to create the generator dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "generator_texts.json"
    metadata = {
        "texts": texts,
        "num_texts": len(texts),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_generator_texts_metadata_from_iteration(iteration_dir: Path) -> Optional[List[str]]:
    """Load generator texts metadata from a previous iteration."""
    metadata_path = iteration_dir / "generator_texts.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("texts", [])


def create_generator_dataset_from_texts(
    texts: List[str],
    tokenizer: GPT2TokenizerFast,
    block_size: int,
    stride: int,
) -> Dataset:
    """Create a generator dataset from a list of texts."""
    if not texts:
        raise ValueError("No texts provided for generator dataset.")
    
    concatenated = "\n\n".join(texts)
    encoding = tokenizer(
        concatenated,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    input_ids = encoding["input_ids"]
    if not input_ids:
        raise ValueError("Tokenisation produced an empty sequence for generator.")
    
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if stride >= block_size:
        raise ValueError("stride must be smaller than block_size.")
    
    total_length = len(input_ids)
    if total_length < block_size:
        raise ValueError(
            "Generator dataset too small for the requested block size. "
            f"Have {total_length} tokens but require at least {block_size}."
        )
    
    step = block_size - stride if stride > 0 else block_size
    max_start = total_length - block_size + 1
    examples = [input_ids[i : i + block_size] for i in range(0, max_start, step)]
    
    dataset = Dataset.from_dict(
        {
            "input_ids": examples,
            "labels": [example.copy() for example in examples],
        }
    )
    return dataset


def augment_generator_dataset(
    previous_texts: List[str],
    previous_hard_neg_indices: Optional[List[int]],
    hard_negatives: List[Tuple[str, float]],
    ratio: Tuple[int, int],
    tokenizer: GPT2TokenizerFast,
    block_size: int,
    stride: int,
) -> Tuple[Dataset, List[str], List[int]]:
    """Augment generator dataset by replacing samples with hard negatives.
    
    Args:
        previous_texts: List of texts from previous iteration (or original)
        previous_hard_neg_indices: Indices in previous_texts that are hard negatives (None for iteration 0)
        hard_negatives: List of (text, confidence) tuples for new hard negatives (already sorted by confidence)
        ratio: (real, generated) ratio tuple
        tokenizer: GPT-2 tokenizer
        block_size: Block size for tokenization
        stride: Stride for tokenization
    
    Returns:
        Tuple of (augmented_dataset, updated_texts_list, new_hard_neg_indices)
    """
    # Extract texts from hard negatives (already sorted by confidence descending)
    new_hard_neg_texts = [text for text, _ in hard_negatives]
    
    # Calculate target counts
    total_texts = len(previous_texts)
    ratio_sum = ratio[0] + ratio[1]
    target_generated = int(total_texts * ratio[1] / ratio_sum)
    target_real = total_texts - target_generated
    
    # Separate real texts from hard negatives
    if previous_hard_neg_indices is None:
        # First iteration: all texts are real
        real_texts = previous_texts.copy()
        existing_hard_neg_texts: List[str] = []
        existing_hard_neg_indices: List[int] = []
    else:
        # Later iterations: separate real from hard negatives
        existing_hard_neg_indices = sorted(set(previous_hard_neg_indices))
        real_texts = [text for i, text in enumerate(previous_texts) if i not in existing_hard_neg_indices]
        existing_hard_neg_texts = [previous_texts[i] for i in existing_hard_neg_indices]
    
    # Merge existing and new hard negatives, keeping best by confidence
    # New hard negatives are already sorted by confidence (descending)
    # We'll prioritize new ones since we have their confidences
    all_hard_negatives = new_hard_neg_texts + existing_hard_neg_texts
    # Take top target_generated (prioritizing new ones)
    selected_hard_negs = all_hard_negatives[:target_generated]
    
    # Combine real texts with selected hard negatives
    texts_to_use = real_texts[:target_real] + selected_hard_negs
    
    # Ensure we have exactly total_texts
    if len(texts_to_use) < total_texts:
        # Pad with remaining real texts
        remaining_real = real_texts[target_real:]
        texts_to_use.extend(remaining_real[:total_texts - len(texts_to_use)])
    elif len(texts_to_use) > total_texts:
        # Truncate (shouldn't happen, but be safe)
        texts_to_use = texts_to_use[:total_texts]
    
    # Calculate new hard negative indices (they're at the end after real texts)
    new_hard_neg_indices = list(range(target_real, len(texts_to_use)))
    
    # Create dataset from updated texts
    dataset = create_generator_dataset_from_texts(texts_to_use, tokenizer, block_size, stride)
    
    return dataset, texts_to_use, new_hard_neg_indices


def augment_discriminator_dataset(
    previous_dataset: DatasetDict,
    hard_negatives: List[Tuple[str, float]],
    generated_label_id: int,
    tokenizer: BertTokenizerFast,
    max_length: int,
    pad_to_max: bool,
) -> DatasetDict:
    """Augment discriminator dataset by replacing generated class samples.
    
    Args:
        previous_dataset: Previous iteration's discriminator dataset
        hard_negatives: List of (text, confidence) tuples for new hard negatives
        generated_label_id: Label ID for the "generated" class
        tokenizer: BERT tokenizer
        max_length: Max sequence length
        pad_to_max: Whether to pad to max length
    
    Returns:
        Augmented DatasetDict
    """
    # Extract texts from hard negatives (sorted by confidence, descending)
    new_hard_neg_texts = [text for text, _ in sorted(hard_negatives, key=lambda x: x[1], reverse=True)]
    
    # Split 50/50 between train and validation
    num_train = len(new_hard_neg_texts) // 2
    train_hard_negs = new_hard_neg_texts[:num_train]
    val_hard_negs = new_hard_neg_texts[num_train:]
    
    # Get existing records from previous dataset
    train_records = []
    val_records = []
    
    for record in previous_dataset["train"]:
        train_records.append({
            "text": str(record["text"]),
            "author": str(record.get("author", "")),
            "label": int(record["label"]),
        })
    
    for record in previous_dataset["validation"]:
        val_records.append({
            "text": str(record["text"]),
            "author": str(record.get("author", "")),
            "label": int(record["label"]),
        })
    
    # Count existing generated samples
    train_generated_indices = [i for i, r in enumerate(train_records) if r["label"] == generated_label_id]
    val_generated_indices = [i for i, r in enumerate(val_records) if r["label"] == generated_label_id]
    
    # Replace oldest generated samples with new hard negatives
    # Sort indices to replace from the beginning (oldest)
    train_generated_indices.sort()
    val_generated_indices.sort()
    
    # Replace train generated samples
    num_train_replace = min(len(train_hard_negs), len(train_generated_indices))
    for i, idx in enumerate(train_generated_indices[:num_train_replace]):
        train_records[idx]["text"] = train_hard_negs[i]
        train_records[idx]["author"] = "generated"
    
    # Replace validation generated samples
    num_val_replace = min(len(val_hard_negs), len(val_generated_indices))
    for i, idx in enumerate(val_generated_indices[:num_val_replace]):
        val_records[idx]["text"] = val_hard_negs[i]
        val_records[idx]["author"] = "generated"
    
    # Create new dataset
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(val_records),
    })
    
    # Re-tokenize
    padding = "max_length" if pad_to_max else "longest"
    
    def tokenize_batch(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            examples["text"],
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
    
    tokenised = dataset_dict.map(
        tokenize_batch,
        batched=True,
        desc="Tokenising augmented discriminator dataset",
    )
    
    return tokenised


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    cfg = CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--g0-model-dir",
        type=Path,
        help="Path to the original generator model (G0).",
    )
    parser.add_argument(
        "--g0-dataset-dir",
        type=Path,
        help="Path to the dataset used to train G0.",
    )
    parser.add_argument(
        "--d0-model-dir",
        type=Path,
        help="Path to the original multiclass BERT discriminator (D0).",
    )
    parser.add_argument(
        "--d0-dataset-dir",
        type=Path,
        help="Path to the dataset used to train D0.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        help="Path to the metadata directory containing label mappings.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=cfg.loop.num_iterations,
        help="Number of loop iterations to run.",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=cfg.loop.num_sequences,
        help="Number of sequences to generate per iteration.",
    )
    parser.add_argument(
        "--prompt-num-words",
        type=int,
        default=cfg.loop.prompt_num_words,
        help="Number of words to use for prompts.",
    )
    parser.add_argument(
        "--hard-negative-threshold",
        type=float,
        default=cfg.loop.hard_negative_threshold,
        help="Confidence threshold for hard negatives.",
    )
    parser.add_argument(
        "--num-hard-negs-per-iteration",
        type=int,
        default=cfg.loop.num_hard_negs_per_iteration,
        help="Maximum number of hard negatives per iteration.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=cfg.paths.loop_root,
        help="Root directory for saving iterations.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for classification.",
    )
    parser.add_argument(
        "--logging-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--_run-step",
        type=str,
        default=None,
        help="Internal flag to run a specific training step."
    )
    parser.add_argument(
        "--_dataset-dir",
        type=Path,
        help="Internal flag for dataset path."
    )
    parser.add_argument(
        "--_output-dir",
        type=Path,
        help="Internal flag for output path."
    )
    parser.add_argument(
        "--_seed",
        type=int,
        help="Internal flag for seed."
    )
    return parser.parse_args()


def train_generator(
    dataset: Dataset,
    output_dir: Path,
    seed: int,
) -> None:
    """Train a generator model using the same approach as g0_train.py.
    
    Always starts from a fresh, untrained GPT-2 model.
    """
    cfg = CONFIG.g0_training
    
    # Prepare tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(CONFIG.tokenizers.gpt2_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Always load a fresh, untrained base model
    LOGGER.info("Loading fresh base model: %s", CONFIG.tokenizers.gpt2_model_name)
    model = GPT2LMHeadModel.from_pretrained(CONFIG.tokenizers.gpt2_model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        use_mps_device=True if torch.backends.mps.is_available() else False,
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        eval_strategy="no",
        save_strategy="no",  # Disable checkpointing
        save_total_limit=0,  # No checkpoints to keep
        load_best_model_at_end=False,
        max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to=cfg.report_to or [],
        dataloader_pin_memory=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    set_seed(seed)
    LOGGER.info("Starting generator training...")
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    trainer.save_state()
    del trainer, model, tokenizer


def train_discriminator(
    dataset: DatasetDict,
    metadata_dir: Path,
    output_dir: Path,
    seed: int,
) -> None:
    """Train a discriminator model using the same approach as dag_train.py.
    
    Always starts from a fresh, untrained BERT model.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    
    cfg = CONFIG.dag_training
    
    # Load label mapping
    dag_label_path = metadata_dir / "dag_label_mapping.json"
    if not dag_label_path.exists():
        raise FileNotFoundError(f"DAG label mapping not found at {dag_label_path}")
    
    label_data = json.loads(dag_label_path.read_text(encoding="utf-8"))
    label2id = {str(k): int(v) for k, v in label_data["label2id"].items()}
    id2label = {int(k): str(v) for k, v in label_data["id2label"].items()}
    num_labels = len(label2id)
    
    # Always load a fresh, untrained base model
    LOGGER.info("Loading fresh base model: %s", cfg.model_name)
    tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
    model = BertForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label={idx: label for idx, label in id2label.items()},
    )
    
    data_collator = DataCollatorWithPadding(tokenizer)
    
    train_dataset = ensure_split(dataset, "train")
    eval_dataset = ensure_split(dataset, "validation")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="no",  # Disable checkpointing
        save_total_limit=0,  # No checkpoints to keep
        max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to=cfg.report_to or [],
        load_best_model_at_end=False,  # No checkpoints to load from
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
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
    
    set_seed(seed)
    LOGGER.info("Starting discriminator training...")
    train_result = classifier.train()
    classifier.save_model()
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            raise RuntimeError(f"Model parameter '{name}' contains NaN values")
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    eval_metrics = classifier.evaluate()
    metrics.update({f"final_{key}": value for key, value in eval_metrics.items()})
    metrics["eval_samples"] = len(eval_dataset)
    
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    # Generate classification report on validation dataset
    LOGGER.info("Generating classification report on validation dataset...")
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
    del classifier, model, tokenizer


def main() -> None:
    """Main training loop."""
    args = parse_args()
    setup_logging(args.logging_level)
    
    

    if args._run_step == "generator":
        setup_logging(args.logging_level)
        LOGGER.info("--- Running Generator Subprocess ---")
        dataset = load_from_disk(str(args._dataset_dir))
        train_generator(dataset, args._output_dir, args._seed)
        LOGGER.info("--- Generator Subprocess Complete ---")
        sys.exit(0)
        
    if args._run_step == "discriminator":
        setup_logging(args.logging_level)
        LOGGER.info("--- Running Discriminator Subprocess ---")
        dataset = load_from_disk(str(args._dataset_dir))
        train_discriminator(
            dataset,
            args.metadata_dir,
            args._output_dir,
            args._seed
        )
        LOGGER.info("--- Discriminator Subprocess Complete ---")
        sys.exit(0)

    # Validate inputs
    if not args.g0_model_dir.exists():
        raise FileNotFoundError(f"G0 model directory not found: {args.g0_model_dir}")
    if not args.g0_dataset_dir.exists():
        raise FileNotFoundError(f"G0 dataset directory not found: {args.g0_dataset_dir}")
    if not args.d0_model_dir.exists():
        raise FileNotFoundError(f"D0 model directory not found: {args.d0_model_dir}")
    if not args.d0_dataset_dir.exists():
        raise FileNotFoundError(f"D0 dataset directory not found: {args.d0_dataset_dir}")
    if not args.metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {args.metadata_dir}")
    
    # Setup device
    device = select_device(args.device)
    LOGGER.info("Using device: %s", device)
    
    # Load initial texts for generator
    generator_texts = load_generator_texts_metadata(args.metadata_dir)
    LOGGER.info("Loaded %d original generator texts", len(generator_texts))
    
    # Load label mappings
    label2id, id2label = read_label_mapping(args.metadata_dir)
    target_author = CONFIG.preprocess.target_author
    
    # Load DAG label mapping
    dag_label_path = args.metadata_dir / "dag_label_mapping.json"
    if not dag_label_path.exists():
        raise FileNotFoundError(f"DAG label mapping not found at {dag_label_path}")
    dag_label_data = json.loads(dag_label_path.read_text(encoding="utf-8"))
    dag_label2id = {str(k): int(v) for k, v in dag_label_data["label2id"].items()}
    dag_id2label = {int(k): str(v) for k, v in dag_label_data["id2label"].items()}
    generated_label_id = dag_label2id.get("generated")
    if generated_label_id is None:
        raise ValueError("Generated label not found in DAG label mapping")
    
    # Initialize tokenizers
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(CONFIG.tokenizers.gpt2_model_name)
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    bert_tokenizer = BertTokenizerFast.from_pretrained(CONFIG.tokenizers.bert_model_name)
    
    # Main loop
    LOGGER.info("=" * 80)
    LOGGER.info("Starting Main Training Loop")
    LOGGER.info("=" * 80)
    LOGGER.info("Arguments: %s", args)
    for iteration in range(args.num_iterations):
        print("\n" * 2)
        LOGGER.info("=" * 80)
        LOGGER.info("=== Iteration %d ===", iteration)
        LOGGER.info("=" * 80)
        
        iteration_dir = args.output_root / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        
        attempt = 0
        while attempt < 3:
            attempt_seed = CONFIG.random_seed + iteration * 100 + attempt
            LOGGER.info("Attempt %d for iteration %d (seed=%d)", attempt + 1, iteration, attempt_seed)
            rng = np.random.default_rng(attempt_seed)
            set_seed(attempt_seed)

            # Step 1: Generate text
            LOGGER.info("Step 1/6: Generating text...")
            generator_model_path, generator_source_iter = find_latest_generator_model(
                args.output_root,
                iteration,
                args.g0_model_dir,
            )
            if generator_source_iter is None:
                LOGGER.info("Using base generator (G0) at %s", generator_model_path)
            else:
                LOGGER.info(
                    "Using generator from iteration %d located at %s",
                    generator_source_iter,
                    generator_model_path,
                )

            model, tokenizer = load_generator(generator_model_path)
            model = model.to(device)
            model.eval()

            # Load texts from d0_dataset for prompts
            d0_dataset = load_dataset_dict(args.d0_dataset_dir)
            d0_train = ensure_split(d0_dataset, "train")
            all_texts = collect_texts(d0_train)

            # Build prompts
            prompts = build_word_prompts(
                all_texts,
                args.num_sequences,
                args.prompt_num_words,
                rng,
            )

            # Generate
            generation_kwargs = {
                "max_new_tokens": CONFIG.generation.max_new_tokens,
                "temperature": CONFIG.generation.temperature,
                "top_k": CONFIG.generation.top_k,
                "top_p": CONFIG.generation.top_p,
                "repetition_penalty": CONFIG.generation.repetition_penalty,
                "num_return_sequences": CONFIG.generation.num_return_sequences,
                "do_sample": CONFIG.generation.do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            samples = generate_sequences(
                model,
                tokenizer,
                prompts,
                device=device,
                generation_kwargs=generation_kwargs,
                logger=LOGGER,
            )

            # Save generations
            generations_path = iteration_dir / "generations.jsonl"
            with generations_path.open("w", encoding="utf-8") as f:
                for sample in samples:
                    record = {
                        "prompt": sample.prompt,
                        "completion": sample.completion,
                        "full_text": sample.full_text,
                    }
                    f.write(json.dumps(record) + "\n")
            LOGGER.info("Saved %d generations to %s", len(samples), generations_path)

            # Step 2: Identify hard negatives
            LOGGER.info("Step 2/6: Identifying hard negatives...")
            discriminator_model_path, discriminator_source_iter = find_latest_discriminator_model(
                args.output_root,
                iteration,
                args.d0_model_dir,
            )
            if discriminator_source_iter is None:
                LOGGER.info("Using base discriminator (D0) at %s", discriminator_model_path)
            else:
                LOGGER.info(
                    "Using discriminator from iteration %d located at %s",
                    discriminator_source_iter,
                    discriminator_model_path,
                )

            # Extract completion texts
            generated_texts = [sample.completion for sample in samples]

            # Classify
            predictions, logits, dag_id2label = classify_texts(
                generated_texts,
                discriminator_model_path,
                args.batch_size,
                device,
            )

            # Compute confidences
            confidences = compute_confidence_scores(logits)

            # Select hard negatives
            hard_negatives = select_hard_negatives(
                generated_texts,
                predictions,
                confidences,
                dag_id2label,
                target_author,
                args.hard_negative_threshold,
                args.num_hard_negs_per_iteration,
            )

            LOGGER.info(
                "Selected %d hard negatives (threshold=%.3f)",
                len(hard_negatives),
                args.hard_negative_threshold,
            )

            hard_negs_path = iteration_dir / "hard_negatives.jsonl"
            with hard_negs_path.open("w", encoding="utf-8") as f:
                for text, confidence in hard_negatives:
                    record = {"text": text, "confidence": confidence}
                    f.write(json.dumps(record) + "\n")

            dist_text = format_distribution(predictions, dag_id2label)
            dist_path = iteration_dir / "classification_distribution.txt"
            dist_path.write_text(dist_text, encoding="utf-8")

            conf_path = iteration_dir / "classification_confidences.json"
            save_confidence_scores(conf_path, generated_texts, confidences, dag_id2label)

            # Check if no hard negatives were found
            if len(hard_negatives) == 0:
                LOGGER.warning(
                    "No hard negatives found (threshold=%.3f) for iteration %d attempt %d. Retrying with a new seed.",
                    args.hard_negative_threshold,
                    iteration,
                    attempt + 1,
                )
                attempt += 1
                continue
            
            # Step 3: Augment generator dataset
            LOGGER.info("Step 3/6: Augmenting generator dataset...")
            if iteration == 0:
                previous_texts = generator_texts
                previous_hard_neg_indices = None
            else:
                prev_iter_dir = args.output_root / f"iteration_{iteration - 1}"
                prev_metadata = load_generator_texts_metadata_from_iteration(prev_iter_dir)
                if prev_metadata:
                    previous_texts = prev_metadata
                    # Load hard neg indices from metadata
                    metadata_file = prev_iter_dir / "generator_texts.json"
                    if metadata_file.exists():
                        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                        previous_hard_neg_indices = metadata.get("hard_neg_indices")
                    else:
                        previous_hard_neg_indices = None
                else:
                    previous_texts = generator_texts
                    previous_hard_neg_indices = None

            augmented_gen_dataset, updated_texts, new_hard_neg_indices = augment_generator_dataset(
                previous_texts,
                previous_hard_neg_indices,
                hard_negatives,
                CONFIG.loop.real_to_generated_ratio,
                gpt2_tokenizer,
                CONFIG.tokenizers.gpt2_block_size,
                CONFIG.tokenizers.gpt2_stride,
            )

            # Save augmented dataset
            gen_dataset_dir = iteration_dir / "generator_dataset"
            augmented_gen_dataset.save_to_disk(str(gen_dataset_dir))

            # Save metadata
            metadata = {
                "texts": updated_texts,
                "num_texts": len(updated_texts),
                "hard_neg_indices": new_hard_neg_indices,
            }
            metadata_path = iteration_dir / "generator_texts.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            LOGGER.info("Augmented generator dataset: %d samples", len(augmented_gen_dataset))

            # Step 4: Augment discriminator dataset
            LOGGER.info("Step 4/6: Augmenting discriminator dataset...")
            if iteration == 0:
                previous_disc_dataset = load_dataset_dict(args.d0_dataset_dir)
            else:
                prev_iter_dir = args.output_root / f"iteration_{iteration - 1}"
                previous_disc_dataset = load_dataset_dict(prev_iter_dir / "discriminator_dataset")

            augmented_disc_dataset = augment_discriminator_dataset(
                previous_disc_dataset,
                hard_negatives,
                generated_label_id,
                bert_tokenizer,
                CONFIG.tokenizers.bert_max_length,
                CONFIG.tokenizers.pad_to_max_length,
            )

            # Save augmented dataset
            disc_dataset_dir = iteration_dir / "discriminator_dataset"
            augmented_disc_dataset.save_to_disk(str(disc_dataset_dir))

            LOGGER.info(
                "Augmented discriminator dataset: train=%d, validation=%d",
                len(augmented_disc_dataset["train"]),
                len(augmented_disc_dataset["validation"]),
            )

            # Step 5: Train G_{i+1}
            LOGGER.info("Step 5/6: Training G_{%d}...", iteration + 1)
            gen_output_dir = iteration_dir / "generators" / f"G_{iteration + 1}"
            LOGGER.info("Spawning subprocess for G_{%d} training...", iteration + 1)
            gen_cmd = [
                "python", sys.argv[0],
                "--_run-step", "generator",
                "--_dataset-dir", str(gen_dataset_dir),
                "--_output-dir", str(gen_output_dir),
                "--_seed", str(CONFIG.random_seed + iteration),
                "--logging-level", args.logging_level,
            ]
            subprocess.run(gen_cmd, check=True)
            LOGGER.info("Generator G_{%d} saved to %s", iteration + 1, gen_output_dir)

            import gc
            gc.collect()

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Step 6: Train D_{i+1}
            LOGGER.info("Step 6/6: Training D_{%d}...", iteration + 1)
            disc_output_dir = iteration_dir / "discriminators" / f"D_{iteration + 1}"
            LOGGER.info("Spawning subprocess for D_{%d} training...", iteration + 1)
            disc_cmd = [
                "python", sys.argv[0],
                "--_run-step", "discriminator",
                "--_dataset-dir", str(disc_dataset_dir),
                "--_output-dir", str(disc_output_dir),
                "--_seed", str(CONFIG.random_seed + iteration),
                "--metadata-dir", str(args.metadata_dir),
                "--logging-level", args.logging_level,
            ]
            subprocess.run(disc_cmd, check=True)
            LOGGER.info("Discriminator D_{%d} saved to %s", iteration + 1, disc_output_dir)

            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            LOGGER.info("Iteration %d complete!", iteration)
            break

        if attempt == 3:
            LOGGER.error("Failed to find hard negatives after 3 attempts. Exiting.")
            sys.exit(1)
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("Main Training Loop Complete!")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()

