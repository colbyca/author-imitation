"""Main training loop for iterative generator/discriminator co-evolution.

This script implements the iterative training loop where generators G_i and
discriminators D_i co-evolve through hard negative mining. Each iteration:
1. Generates samples from G_i
2. Evaluates them with D_i to find hard negatives (samples classified as target)
3. Augments the generator dataset with hard negatives (9:1 ratio)
4. Retrains G_{i+1} from scratch on the augmented dataset
5. Augments the discriminator dataset by replacing generator class samples
6. Retrains D_{i+1} from scratch on the augmented dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Disable tokenizers parallelism to avoid fork warnings
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
    TrainingArguments,
    set_seed,
)

from config import CONFIG
from utils import (
    BertClassifierTrainer,
    build_word_prompts,
    generate_sequences,
    load_dataset_dict,
    load_generator,
    load_target_author_texts,
    read_label_mapping as read_regular_label_mapping,
    select_device,
    setup_logging,
)

LOGGER = logging.getLogger("main_loop")


@dataclass
class HardNegative:
    """A generated sample that fooled the discriminator."""

    text: str
    confidence: float
    predicted_label: str
    prompt: str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    cfg = CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Required paths
    parser.add_argument(
        "--g0-model-dir",
        type=Path,
        required=True,
        help="Path to trained G_0 model directory.",
    )
    parser.add_argument(
        "--d0-model-dir",
        type=Path,
        required=True,
        help="Path to trained D_0 model directory.",
    )
    parser.add_argument(
        "--g0-dataset-dir",
        type=Path,
        required=True,
        help="Path to original G_0 training dataset directory.",
    )
    parser.add_argument(
        "--d0-dataset-dir",
        type=Path,
        required=True,
        help="Path to original D_0 training dataset directory (D_aa format).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        required=True,
        help="Path to metadata directory with label mappings.",
    )
    
    # Loop configuration
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=cfg.main_loop.num_iterations,
        help="Number of loop iterations.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=cfg.paths.loop_root,
        help="Root directory for saving iterations.",
    )
    
    # Generation parameters
    parser.add_argument(
        "--samples-per-iteration",
        type=int,
        default=cfg.main_loop.samples_per_iteration,
        help="Number of samples to generate per iteration.",
    )
    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=cfg.main_loop.generation_max_new_tokens,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--generation-min-new-tokens",
        type=int,
        default=cfg.main_loop.generation_min_new_tokens,
        help="Minimum new tokens to generate.",
    )
    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=cfg.main_loop.generation_temperature,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--generation-top-k",
        type=int,
        default=cfg.main_loop.generation_top_k,
        help="Generation top-k.",
    )
    parser.add_argument(
        "--generation-top-p",
        type=float,
        default=cfg.main_loop.generation_top_p,
        help="Generation top-p (nucleus).",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=cfg.main_loop.generation_batch_size,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--prompt-char-limit",
        type=int,
        default=cfg.main_loop.prompt_char_limit,
        help="Character limit for prompts.",
    )
    
    # Hard negative mining
    parser.add_argument(
        "--hard-negative-threshold",
        type=float,
        default=cfg.main_loop.hard_negative_threshold,
        help="Confidence threshold for hard negatives.",
    )
    parser.add_argument(
        "--max-hard-negatives",
        type=int,
        default=cfg.main_loop.max_hard_negatives,
        help="Maximum number of hard negatives to use.",
    )
    
    # Training parameters
    parser.add_argument(
        "--generator-epochs",
        type=float,
        default=cfg.main_loop.generator_epochs,
        help="Number of epochs for generator training.",
    )
    parser.add_argument(
        "--discriminator-epochs",
        type=float,
        default=cfg.main_loop.discriminator_epochs,
        help="Number of epochs for discriminator training.",
    )
    parser.add_argument(
        "--generator-learning-rate",
        type=float,
        default=cfg.g0_training.learning_rate,
        help="Learning rate for generator training.",
    )
    parser.add_argument(
        "--discriminator-learning-rate",
        type=float,
        default=cfg.binary_discriminator.learning_rate,
        help="Learning rate for discriminator training.",
    )
    
    # Model configuration
    parser.add_argument(
        "--gpt2-model-name",
        type=str,
        default=cfg.tokenizers.gpt2_model_name,
        help="Base GPT-2 model name.",
    )
    parser.add_argument(
        "--bert-model-name",
        type=str,
        default=cfg.binary_discriminator.model_name,
        help="Base BERT model name.",
    )
    
    # System configuration
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.random_seed,
        help="Random seed.",
    )
    parser.add_argument(
        "--logging-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directories.",
    )
    
    return parser.parse_args()


def read_dag_label_mapping(metadata_dir: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Read D_ag label mapping from metadata directory."""
    mapping_path = metadata_dir / "dag_label_mapping.json"
    if mapping_path.exists():
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        raw_label2id = payload.get("label2id", {})
        raw_id2label = payload.get("id2label", {})
        label2id = {str(name): int(idx) for name, idx in raw_label2id.items()}
        id2label = {int(idx): str(name) for idx, name in raw_id2label.items()}
        return label2id, id2label
    
    # Fallback to regular label mapping
    return read_regular_label_mapping(metadata_dir)


def classify_texts_with_discriminator(
    texts: Sequence[str],
    model_dir: Path,
    batch_size: int,
    device: torch.device,
    metadata_dir: Optional[Path] = None,
    max_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Classify texts using a discriminator model.
    
    Returns:
        Tuple of (predicted_labels, prediction_logits, id2label)
    """
    # Try to load from model config first
    config_path = model_dir / "config.json"
    label2id = None
    id2label = None
    
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if "label2id" in config and "id2label" in config:
            raw_label2id = config.get("label2id", {})
            raw_id2label = config.get("id2label", {})
            label2id = {str(name): int(idx) for name, idx in raw_label2id.items()}
            id2label = {int(idx): str(name) for idx, name in raw_id2label.items()}
    
    # Fallback to metadata directory if config doesn't have labels
    if label2id is None or id2label is None:
        if metadata_dir is not None:
            label2id, id2label = read_dag_label_mapping(metadata_dir)
        else:
            # Try to find metadata directory
            possible_metadata = model_dir.parent.parent / "metadata"
            if possible_metadata.exists():
                label2id, id2label = read_dag_label_mapping(possible_metadata)
            else:
                raise ValueError("Could not find label mapping in model config or metadata directory")
    
    num_labels = len(label2id)
    LOGGER.info("Classifying %d texts with %d labels", len(texts), num_labels)
    
    # Load model and tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(str(model_dir))
    model = BertForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=num_labels,
        label2id=label2id,
        id2label={idx: label for idx, label in id2label.items()},
    )
    model.to(device)
    model.eval()
    
    # Determine max_length
    if max_length is None:
        max_length = CONFIG.tokenizers.bert_max_length
    
    # Create dataset
    dataset = Dataset.from_dict({"text": list(texts)})
    
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
    
    tokenized = dataset.map(tokenize, batched=True)
    
    # Setup trainer for prediction
    data_collator = DataCollatorWithPadding(tokenizer)
    training_args = TrainingArguments(
        output_dir=str(model_dir / "eval_outputs"),
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=4,
        do_eval=True,
        do_train=False,
        report_to=[],
        logging_dir=str(model_dir / "eval_outputs" / "logs"),
        no_cuda=device.type != "cuda",
    )
    
    classifier = BertClassifierTrainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        eval_dataset=tokenized,
        data_collator=data_collator,
    )
    
    # Predict
    predictions = classifier.predict(tokenized)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    pred_logits = predictions.predictions
    
    return pred_labels, pred_logits, id2label


def compute_confidence_scores(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities from logits."""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probabilities


def format_classification_distribution(
    predictions: np.ndarray,
    id2label: Dict[int, str],
) -> str:
    """Format the prediction distribution as a readable string."""
    counter = Counter(int(pred) for pred in predictions)
    total = len(predictions)
    
    lines = ["Classification Distribution:", "=" * 50]
    lines.append(f"Total samples: {total}")
    lines.append("")
    lines.append(f"{'Label':<20} {'Count':<10} {'Percentage':<10}")
    lines.append("-" * 50)
    
    # Sort by label index
    for label_id in sorted(id2label.keys()):
        label_name = id2label[label_id]
        count = counter.get(label_id, 0)
        percentage = (count / total * 100) if total > 0 else 0.0
        lines.append(f"{label_name:<20} {count:<10} {percentage:>6.2f}%")
    
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)


def generate_and_evaluate(
    generator_model_dir: Path,
    discriminator_model_dir: Path,
    target_author: str,
    num_samples: int,
    generation_kwargs: dict,
    device: torch.device,
    hard_negative_threshold: float,
    max_hard_negatives: Optional[int],
    metadata_dir: Path,
    d0_dataset_dir: Path,
) -> Tuple[List[HardNegative], np.ndarray, Dict[int, str], List[str]]:
    """Generate samples from G_i and evaluate with D_i to find hard negatives.
    
    Returns:
        Tuple of:
        - List of hard negatives sorted by confidence (descending)
        - Prediction labels array
        - Label ID to name mapping
        - List of generated texts
    """
    LOGGER.info("Loading generator model from %s", generator_model_dir)
    model, tokenizer = load_generator(generator_model_dir)
    model = model.to(device)
    
    # Load target author texts for prompts
    LOGGER.info("Loading target author texts for prompts")
    target_texts = load_target_author_texts(
        d0_dataset_dir,
        metadata_dir,
        target_author,
        split="train",
    )
    
    # Generate prompts
    rng = np.random.default_rng(CONFIG.random_seed)
    num_words = max(5, min(10, CONFIG.main_loop.prompt_char_limit // 10))
    prompts = build_word_prompts(target_texts, num_samples, num_words, rng)
    
    LOGGER.info("Generating %d samples from generator", num_samples)
    samples = generate_sequences(
        model,
        tokenizer,
        prompts,
        device=device,
        generation_kwargs=generation_kwargs,
        batch_size=CONFIG.main_loop.generation_batch_size,
        logger=LOGGER,
    )
    
    # Extract completion texts
    generated_texts = [sample.completion for sample in samples]
    prompt_texts = [sample.prompt for sample in samples]
    
    LOGGER.info("Evaluating %d generated samples with discriminator", len(generated_texts))
    pred_labels, pred_logits, id2label = classify_texts_with_discriminator(
        generated_texts,
        discriminator_model_dir,
        batch_size=32,
        device=device,
        metadata_dir=metadata_dir,
    )
    
    # Compute confidence scores
    confidences = compute_confidence_scores(pred_logits)
    
    # Find hard negatives (classified as target author)
    label2id, _ = read_regular_label_mapping(metadata_dir)
    target_label_id = label2id.get(target_author)
    if target_label_id is None:
        # Try D_ag format
        dag_label2id, _ = read_dag_label_mapping(metadata_dir)
        target_label_id = dag_label2id.get(target_author)
    
    if target_label_id is None:
        raise ValueError(f"Target author '{target_author}' not found in label mapping")
    
    hard_negatives: List[HardNegative] = []
    for i, (text, prompt, pred_label, conf_row) in enumerate(
        zip(generated_texts, prompt_texts, pred_labels, confidences)
    ):
        pred_label_id = int(pred_label)
        target_confidence = float(conf_row[target_label_id])
        predicted_label_name = id2label.get(pred_label_id, f"label_{pred_label_id}")
        
        # Check if classified as target author
        if pred_label_id == target_label_id:
            if target_confidence >= hard_negative_threshold:
                hard_negatives.append(
                    HardNegative(
                        text=text,
                        confidence=target_confidence,
                        predicted_label=predicted_label_name,
                        prompt=prompt,
                    )
                )
    
    # Sort by confidence (descending)
    hard_negatives.sort(key=lambda x: x.confidence, reverse=True)
    
    # Apply max limit if specified
    if max_hard_negatives is not None and len(hard_negatives) > max_hard_negatives:
        hard_negatives = hard_negatives[:max_hard_negatives]
    
    LOGGER.info(
        "Found %d hard negatives (threshold=%.2f, max=%s)",
        len(hard_negatives),
        hard_negative_threshold,
        max_hard_negatives or "unlimited",
    )
    
    return hard_negatives, pred_labels, id2label, generated_texts


def augment_generator_dataset(
    original_dataset: Dataset,
    hard_negatives: List[HardNegative],
    tokenizer: GPT2TokenizerFast,
    block_size: int,
    stride: int,
    previous_generated_samples: Optional[List[Tuple[str, float]]] = None,
    real_to_generated_ratio: float = 9.0,
) -> Tuple[Dataset, List[Tuple[str, float]]]:
    """Augment generator dataset with hard negatives maintaining 9:1 ratio.
    
    Strategy:
    1. Replace real data with hard negatives until 9:1 ratio is reached
    2. Once ratio is met, replace lower-confidence generated samples with
       higher-confidence ones from current hard negatives
    
    Args:
        original_dataset: Original G_0 dataset (all real samples)
        hard_negatives: Current iteration's hard negatives (sorted by confidence desc)
        previous_generated_samples: List of (text, confidence) from previous iteration
        real_to_generated_ratio: Target ratio of real:generated (default 9.0)
    
    Returns:
        Tuple of (augmented_dataset, new_generated_samples_list)
    """
    # Calculate target numbers
    num_real_total = len(original_dataset)
    target_num_generated = int(num_real_total / real_to_generated_ratio)
    target_num_real = num_real_total - target_num_generated
    
    # Track generated samples: list of (text, confidence)
    current_generated: List[Tuple[str, float]] = []
    
    if previous_generated_samples is not None and len(previous_generated_samples) > 0:
        # We have existing generated samples
        current_generated = previous_generated_samples.copy()
        num_current_generated = len(current_generated)
        
        if num_current_generated < target_num_generated:
            # Still need to replace more real samples to reach ratio
            num_to_replace = target_num_generated - num_current_generated
            # Replace real samples with hard negatives
            for i in range(min(num_to_replace, len(hard_negatives))):
                current_generated.append((hard_negatives[i].text, hard_negatives[i].confidence))
        else:
            # Ratio already met, replace lower-confidence with higher-confidence
            # Sort current generated by confidence (ascending - lowest first)
            current_generated.sort(key=lambda x: x[1])
            
            # Replace lower-confidence ones with higher-confidence hard negatives
            replacement_idx = 0
            for i, (text, conf) in enumerate(current_generated):
                if replacement_idx >= len(hard_negatives):
                    break
                if hard_negatives[replacement_idx].confidence > conf:
                    current_generated[i] = (
                        hard_negatives[replacement_idx].text,
                        hard_negatives[replacement_idx].confidence,
                    )
                    replacement_idx += 1
    else:
        # First iteration: replace real samples with hard negatives until ratio is met
        num_to_replace = min(target_num_generated, len(hard_negatives))
        for i in range(num_to_replace):
            current_generated.append((hard_negatives[i].text, hard_negatives[i].confidence))
    
    # Ensure we don't exceed target
    if len(current_generated) > target_num_generated:
        # Keep only highest confidence ones
        current_generated.sort(key=lambda x: x[1], reverse=True)
        current_generated = current_generated[:target_num_generated]
    
    LOGGER.info(
        "Augmenting generator dataset: %d real samples, %d generated samples (target ratio %.1f:1)",
        target_num_real,
        len(current_generated),
        real_to_generated_ratio,
    )
    
    # Tokenize generated texts following the same pattern as preprocess.py
    # Concatenate all generated texts and create sliding windows
    generated_texts = [text for text, _ in current_generated]
    
    if not generated_texts:
        # No generated texts, just return real samples
        num_real_to_keep = min(target_num_real, len(original_dataset))
        real_input_ids = list(original_dataset["input_ids"][:num_real_to_keep])
        real_labels = list(original_dataset["labels"][:num_real_to_keep])
        augmented_dataset = Dataset.from_dict({
            "input_ids": real_input_ids,
            "labels": real_labels,
        })
        return augmented_dataset, current_generated
    
    # Concatenate generated texts (same as original preprocessing)
    concatenated = "\n\n".join(generated_texts)
    encoding = tokenizer(
        concatenated,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    generated_input_ids = encoding["input_ids"]
    
    if not generated_input_ids:
        LOGGER.warning("Tokenization produced empty sequence for generated texts")
        generated_examples = []
    else:
        # Create sliding windows (same as original preprocessing)
        if stride >= block_size:
            stride = max(0, block_size - 1)  # Ensure stride < block_size
        
        step = block_size - stride if stride > 0 else block_size
        max_start = len(generated_input_ids) - block_size + 1
        
        if max_start <= 0:
            # Text is shorter than block_size, pad or truncate
            if len(generated_input_ids) < block_size:
                # Pad with EOS token or truncate
                generated_input_ids = generated_input_ids[:block_size]
                if len(generated_input_ids) < block_size:
                    pad_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
                    generated_input_ids = generated_input_ids + [pad_token_id] * (block_size - len(generated_input_ids))
            generated_examples = [generated_input_ids]
        else:
            generated_examples = [
                generated_input_ids[i : i + block_size]
                for i in range(0, max_start, step)
            ]
    
    # Take target_num_real samples from original dataset
    num_real_to_keep = min(target_num_real, len(original_dataset))
    real_input_ids = list(original_dataset["input_ids"][:num_real_to_keep])
    real_labels = list(original_dataset["labels"][:num_real_to_keep])
    
    # Combine real and generated
    # Each example is a list of token IDs (integers), labels are copies
    all_input_ids = real_input_ids + generated_examples
    all_labels = real_labels + [example.copy() for example in generated_examples]
    
    augmented_dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
    })
    
    LOGGER.info(
        "Augmented dataset: %d total samples (%d real + %d generated)",
        len(augmented_dataset),
        len(real_input_ids),
        len(generated_examples),
    )
    
    return augmented_dataset, current_generated


def augment_discriminator_dataset(
    original_dataset: DatasetDict,
    hard_negatives: List[HardNegative],
    label2id: Dict[str, int],
    tokenizer: BertTokenizerFast,
    max_length: int,
    pad_to_max: bool,
    previous_generated_samples: Optional[Dict[str, List[Tuple[str, float]]]] = None,
    generated_train_ratio: float = 0.5,
) -> Tuple[DatasetDict, Dict[str, List[Tuple[str, float]]]]:
    """Augment discriminator dataset by replacing generator class samples.
    
    Strategy:
    1. Replace ALL generator class samples with hard negatives (one-to-one, no adding)
    2. Once all original generator samples are replaced, replace lower-confidence
       generated samples with higher-confidence ones from current hard negatives
    
    Args:
        original_dataset: Original D_0 dataset (D_aa format)
        hard_negatives: Current iteration's hard negatives (sorted by confidence desc)
        previous_generated_samples: Dict with "train" and "validation" keys, each containing
                                   list of (text, confidence) tuples from previous iteration
        generated_train_ratio: Ratio of hard negatives for training vs validation
    
    Returns:
        Tuple of (augmented_dataset, new_generated_samples_dict)
    """
    generated_label = label2id.get("generated")
    if generated_label is None:
        raise ValueError("'generated' label not found in label2id mapping")
    
    # Extract records from original dataset
    train_records = []
    eval_records = []
    
    # Track generator sample indices
    train_generated_indices = []
    eval_generated_indices = []
    
    for i, record in enumerate(original_dataset["train"]):
        if int(record["label"]) == generated_label:
            train_generated_indices.append(i)
        train_records.append({
            "text": str(record["text"]),
            "author": str(record.get("author", "unknown")),
            "label": int(record["label"]),
            "split": "train",
            "source_file": str(record.get("source_file", "")),
        })
    
    for i, record in enumerate(original_dataset["validation"]):
        if int(record["label"]) == generated_label:
            eval_generated_indices.append(i)
        eval_records.append({
            "text": str(record["text"]),
            "author": str(record.get("author", "unknown")),
            "label": int(record["label"]),
            "split": "validation",
            "source_file": str(record.get("source_file", "")),
        })
    
    LOGGER.info(
        "Original dataset: %d train generator samples, %d eval generator samples",
        len(train_generated_indices),
        len(eval_generated_indices),
    )
    
    # Split hard negatives for train/eval
    num_train_generated = int(len(hard_negatives) * generated_train_ratio)
    train_hard_negatives = hard_negatives[:num_train_generated]
    eval_hard_negatives = hard_negatives[num_train_generated:]
    
    # Track current generated samples with confidences
    current_train_generated: List[Tuple[str, float]] = []
    current_eval_generated: List[Tuple[str, float]] = []
    
    if previous_generated_samples is not None:
        # We have previous generated samples
        current_train_generated = previous_generated_samples.get("train", []).copy()
        current_eval_generated = previous_generated_samples.get("validation", []).copy()
        
        # Check if all original samples have been replaced
        all_train_replaced = len(current_train_generated) >= len(train_generated_indices)
        all_eval_replaced = len(current_eval_generated) >= len(eval_generated_indices)
        
        if all_train_replaced and all_eval_replaced:
            # All replaced, now replace lower-confidence with higher-confidence
            # Sort by confidence (ascending - lowest first)
            current_train_generated.sort(key=lambda x: x[1])
            current_eval_generated.sort(key=lambda x: x[1])
            
            # Replace lower-confidence train samples
            replacement_idx = 0
            for i, (text, conf) in enumerate(current_train_generated):
                if replacement_idx >= len(train_hard_negatives):
                    break
                if train_hard_negatives[replacement_idx].confidence > conf:
                    current_train_generated[i] = (
                        train_hard_negatives[replacement_idx].text,
                        train_hard_negatives[replacement_idx].confidence,
                    )
                    replacement_idx += 1
            
            # Replace lower-confidence eval samples
            replacement_idx = 0
            for i, (text, conf) in enumerate(current_eval_generated):
                if replacement_idx >= len(eval_hard_negatives):
                    break
                if eval_hard_negatives[replacement_idx].confidence > conf:
                    current_eval_generated[i] = (
                        eval_hard_negatives[replacement_idx].text,
                        eval_hard_negatives[replacement_idx].confidence,
                    )
                    replacement_idx += 1
        else:
            # Still replacing original samples
            # Replace train generator samples
            replacement_idx = 0
            for gen_idx in train_generated_indices:
                if replacement_idx < len(train_hard_negatives):
                    current_train_generated.append((
                        train_hard_negatives[replacement_idx].text,
                        train_hard_negatives[replacement_idx].confidence,
                    ))
                    replacement_idx += 1
                else:
                    break
            
            # Replace eval generator samples
            replacement_idx = 0
            for gen_idx in eval_generated_indices:
                if replacement_idx < len(eval_hard_negatives):
                    current_eval_generated.append((
                        eval_hard_negatives[replacement_idx].text,
                        eval_hard_negatives[replacement_idx].confidence,
                    ))
                    replacement_idx += 1
                else:
                    break
    else:
        # First iteration: replace all original generator samples
        replacement_idx = 0
        for gen_idx in train_generated_indices:
            if replacement_idx < len(train_hard_negatives):
                current_train_generated.append((
                    train_hard_negatives[replacement_idx].text,
                    train_hard_negatives[replacement_idx].confidence,
                ))
                replacement_idx += 1
        
        replacement_idx = 0
        for gen_idx in eval_generated_indices:
            if replacement_idx < len(eval_hard_negatives):
                current_eval_generated.append((
                    eval_hard_negatives[replacement_idx].text,
                    eval_hard_negatives[replacement_idx].confidence,
                ))
                replacement_idx += 1
    
    # Now update the records with generated samples
    # Replace train generator samples
    replacement_idx = 0
    for gen_idx in train_generated_indices:
        if replacement_idx < len(current_train_generated):
            train_records[gen_idx] = {
                "text": current_train_generated[replacement_idx][0],
                "author": "generated",
                "label": generated_label,
                "split": "train",
                "source_file": f"hard_negative_{replacement_idx}",
            }
            replacement_idx += 1
    
    # Replace eval generator samples
    replacement_idx = 0
    for gen_idx in eval_generated_indices:
        if replacement_idx < len(current_eval_generated):
            eval_records[gen_idx] = {
                "text": current_eval_generated[replacement_idx][0],
                "author": "generated",
                "label": generated_label,
                "split": "validation",
                "source_file": f"hard_negative_eval_{replacement_idx}",
            }
            replacement_idx += 1
    
    LOGGER.info(
        "Augmented discriminator dataset: %d train samples (%d generated), %d eval samples (%d generated)",
        len(train_records),
        sum(1 for r in train_records if r["label"] == generated_label),
        len(eval_records),
        sum(1 for r in eval_records if r["label"] == generated_label),
    )
    
    # Create dataset dict
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(eval_records),
    })
    
    # Tokenize
    padding = "max_length" if pad_to_max else "longest"
    
    def tokenize_batch(examples: Dict[str, List[str]]) -> Dict[str, List]:
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
    
    new_generated_samples = {
        "train": current_train_generated,
        "validation": current_eval_generated,
    }
    
    return tokenised, new_generated_samples


def train_generator(
    dataset: Dataset,
    output_dir: Path,
    model_name: str,
    num_epochs: float,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> None:
    """Train a generator from scratch on the augmented dataset."""
    LOGGER.info("Training generator with %d samples for %.1f epochs", len(dataset), num_epochs)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    
    # Prepare tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=CONFIG.g0_training.per_device_train_batch_size,
        gradient_accumulation_steps=CONFIG.g0_training.gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=CONFIG.g0_training.warmup_steps,
        weight_decay=CONFIG.g0_training.weight_decay,
        logging_steps=CONFIG.g0_training.logging_steps,
        save_strategy="epoch",
        save_total_limit=CONFIG.g0_training.save_total_limit,
        fp16=CONFIG.g0_training.fp16,
        bf16=CONFIG.g0_training.bf16,
        gradient_checkpointing=CONFIG.g0_training.gradient_checkpointing,
        dataloader_num_workers=CONFIG.g0_training.dataloader_num_workers,
        report_to=CONFIG.g0_training.report_to or [],
        use_mps_device=torch.backends.mps.is_available(),
    )
    
    from transformers import Trainer
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    LOGGER.info("Starting generator training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    # Save metrics
    metrics = trainer.state.log_history
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    LOGGER.info("Generator training complete. Model saved to %s", output_dir)


def train_discriminator(
    dataset: DatasetDict,
    output_dir: Path,
    model_name: str,
    num_epochs: float,
    learning_rate: float,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    device: torch.device,
    seed: int,
) -> None:
    """Train a discriminator from scratch on the augmented dataset."""
    LOGGER.info(
        "Training discriminator with %d train samples, %d eval samples for %.1f epochs",
        len(dataset["train"]),
        len(dataset["validation"]),
        num_epochs,
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    
    # Build model and tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    num_labels = len(label2id)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label={idx: label for idx, label in id2label.items()},
    )
    
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=CONFIG.binary_discriminator.per_device_train_batch_size,
        per_device_eval_batch_size=CONFIG.binary_discriminator.per_device_eval_batch_size,
        gradient_accumulation_steps=CONFIG.binary_discriminator.gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=CONFIG.binary_discriminator.warmup_steps,
        weight_decay=CONFIG.binary_discriminator.weight_decay,
        logging_steps=CONFIG.binary_discriminator.logging_steps,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        fp16=CONFIG.binary_discriminator.fp16,
        bf16=CONFIG.binary_discriminator.bf16,
        gradient_checkpointing=CONFIG.binary_discriminator.gradient_checkpointing,
        dataloader_num_workers=CONFIG.binary_discriminator.dataloader_num_workers,
        report_to=CONFIG.binary_discriminator.report_to or [],
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        use_mps_device=torch.backends.mps.is_available(),
    )
    
    # Compute metrics function
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    def compute_metrics(eval_pred):
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
    
    classifier = BertClassifierTrainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    LOGGER.info("Starting discriminator training...")
    train_result = classifier.train()
    classifier.save_model()
    
    # Evaluate and save metrics
    eval_metrics = classifier.evaluate()
    metrics = train_result.metrics
    metrics.update({f"final_{key}": value for key, value in eval_metrics.items()})
    
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    LOGGER.info("Discriminator training complete. Model saved to %s", output_dir)


def main() -> None:
    """Main training loop."""
    args = parse_args()
    setup_logging(args.logging_level)
    LOGGER.info("Starting main training loop")
    LOGGER.info("Arguments: %s", args)
    
    set_seed(args.seed)
    device = select_device(args.device)
    LOGGER.info("Using device: %s", device)
    
    # Validate inputs
    if not args.g0_model_dir.exists():
        raise FileNotFoundError(f"G_0 model directory not found: {args.g0_model_dir}")
    if not args.d0_model_dir.exists():
        raise FileNotFoundError(f"D_0 model directory not found: {args.d0_model_dir}")
    if not args.g0_dataset_dir.exists():
        raise FileNotFoundError(f"G_0 dataset directory not found: {args.g0_dataset_dir}")
    if not args.d0_dataset_dir.exists():
        raise FileNotFoundError(f"D_0 dataset directory not found: {args.d0_dataset_dir}")
    if not args.metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {args.metadata_dir}")
    
    # Load target author from config
    target_author = CONFIG.preprocess.target_author
    
    # Load original datasets
    LOGGER.info("Loading original datasets")
    original_g_dataset = load_from_disk(str(args.g0_dataset_dir))
    if isinstance(original_g_dataset, DatasetDict):
        original_g_dataset = original_g_dataset["train"]
    
    original_d_dataset = load_dataset_dict(args.d0_dataset_dir)
    
    # Load label mappings
    label2id, id2label = read_regular_label_mapping(args.metadata_dir)
    
    # Try to get D_ag label mapping if available
    dag_label2id, dag_id2label = read_dag_label_mapping(args.metadata_dir)
    
    # Prepare output directory
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Prepare generation kwargs
    generation_kwargs = {
        "max_new_tokens": args.generation_max_new_tokens,
        "min_new_tokens": args.generation_min_new_tokens,
        "temperature": args.generation_temperature,
        "top_k": args.generation_top_k,
        "top_p": args.generation_top_p,
        "num_return_sequences": 1,
        "do_sample": True,
        "pad_token_id": None,  # Will be set by generate_sequences
        "eos_token_id": None,   # Will be set by generate_sequences
    }
    
    # Initialize models for iteration 0
    current_g_model_dir = args.g0_model_dir
    current_d_model_dir = args.d0_model_dir
    
    # Track generated samples across iterations
    previous_g_generated_samples: Optional[List[Tuple[str, float]]] = None
    previous_d_generated_samples: Optional[Dict[str, List[Tuple[str, float]]]] = None
    
    # Main loop
    for iteration in range(args.num_iterations):
        LOGGER.info("=" * 80)
        LOGGER.info("ITERATION %d / %d", iteration + 1, args.num_iterations)
        LOGGER.info("=" * 80)
        
        iteration_dir = output_root / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1-2: Generate and evaluate to find hard negatives
        LOGGER.info("Step 1-2: Generating samples and finding hard negatives")
        hard_negatives, pred_labels, id2label, generated_texts = generate_and_evaluate(
            generator_model_dir=current_g_model_dir,
            discriminator_model_dir=current_d_model_dir,
            target_author=target_author,
            num_samples=args.samples_per_iteration,
            generation_kwargs=generation_kwargs,
            device=device,
            hard_negative_threshold=args.hard_negative_threshold,
            max_hard_negatives=args.max_hard_negatives,
            metadata_dir=args.metadata_dir,
            d0_dataset_dir=args.d0_dataset_dir,
        )
        
        # Create and save classification distribution report
        distribution_text = format_classification_distribution(pred_labels, id2label)
        distribution_path = iteration_dir / "classification_distribution.txt"
        distribution_path.write_text(distribution_text, encoding="utf-8")
        LOGGER.info("Saved classification distribution to %s", distribution_path)
        # LOGGER.info("\n%s", distribution_text)
        
        # Save hard negatives
        hard_negatives_path = iteration_dir / "hard_negatives.jsonl"
        with hard_negatives_path.open("w", encoding="utf-8") as f:
            for hn in hard_negatives:
                record = {
                    "text": hn.text,
                    "confidence": hn.confidence,
                    "predicted_label": hn.predicted_label,
                    "prompt": hn.prompt,
                }
                f.write(json.dumps(record) + "\n")
        LOGGER.info("Saved %d hard negatives to %s", len(hard_negatives), hard_negatives_path)
        
        if not hard_negatives:
            LOGGER.warning("No hard negatives found in iteration %d. Skipping augmentation.", iteration)
            continue
        
        # Step 3: Augment generator dataset
        LOGGER.info("Step 3: Augmenting generator dataset")
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt2_model_name)
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        
        augmented_g_dataset, new_g_generated_samples = augment_generator_dataset(
            original_dataset=original_g_dataset,
            hard_negatives=hard_negatives,
            tokenizer=gpt2_tokenizer,
            block_size=CONFIG.tokenizers.gpt2_block_size,
            stride=CONFIG.tokenizers.gpt2_stride,
            previous_generated_samples=previous_g_generated_samples,
            real_to_generated_ratio=9.0,
        )
        previous_g_generated_samples = new_g_generated_samples
        
        # Save augmented generator dataset
        g_dataset_dir = iteration_dir / "generator_dataset"
        augmented_g_dataset.save_to_disk(str(g_dataset_dir))
        LOGGER.info("Saved augmented generator dataset to %s", g_dataset_dir)
        
        # Step 4: Train G_{i+1}
        LOGGER.info("Step 4: Training G_%d", iteration + 1)
        g_output_dir = iteration_dir / "generators" / f"G_{iteration + 1}"
        train_generator(
            dataset=augmented_g_dataset,
            output_dir=g_output_dir,
            model_name=args.gpt2_model_name,
            num_epochs=args.generator_epochs,
            learning_rate=args.generator_learning_rate,
            device=device,
            seed=args.seed,
        )
        current_g_model_dir = g_output_dir
        
        # Step 5: Augment discriminator dataset
        LOGGER.info("Step 5: Augmenting discriminator dataset")
        # Use D_ag label mapping if available, otherwise create it
        if "generated" not in dag_label2id:
            # Create D_ag label mapping
            distractors = [author for author in label2id.keys() if author != target_author]
            dag_label2id = {author: idx for idx, author in enumerate([target_author, *distractors])}
            dag_label2id["generated"] = len(dag_label2id)
            dag_id2label = {idx: label for label, idx in dag_label2id.items()}
        
        bert_tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name)
        augmented_d_dataset, new_d_generated_samples = augment_discriminator_dataset(
            original_dataset=original_d_dataset,
            hard_negatives=hard_negatives,
            label2id=dag_label2id,
            tokenizer=bert_tokenizer,
            max_length=CONFIG.tokenizers.bert_max_length,
            pad_to_max=CONFIG.tokenizers.pad_to_max_length,
            previous_generated_samples=previous_d_generated_samples,
            generated_train_ratio=0.5,
        )
        previous_d_generated_samples = new_d_generated_samples
        
        # Save augmented discriminator dataset
        d_dataset_dir = iteration_dir / "discriminator_dataset"
        augmented_d_dataset.save_to_disk(str(d_dataset_dir))
        LOGGER.info("Saved augmented discriminator dataset to %s", d_dataset_dir)
        
        # Step 6: Train D_{i+1}
        LOGGER.info("Step 6: Training D_%d", iteration + 1)
        d_output_dir = iteration_dir / "discriminators" / f"D_{iteration + 1}"
        train_discriminator(
            dataset=augmented_d_dataset,
            output_dir=d_output_dir,
            model_name=args.bert_model_name,
            num_epochs=args.discriminator_epochs,
            learning_rate=args.discriminator_learning_rate,
            label2id=dag_label2id,
            id2label=dag_id2label,
            device=device,
            seed=args.seed,
        )
        current_d_model_dir = d_output_dir
        
        LOGGER.info("Iteration %d complete. G_%d saved to %s, D_%d saved to %s",
                   iteration + 1, iteration + 1, g_output_dir, iteration + 1, d_output_dir)
    
    LOGGER.info("=" * 80)
    LOGGER.info("Main training loop complete!")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()

