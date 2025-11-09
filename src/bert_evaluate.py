"""Evaluate generated texts using any BERT classifier.

This script loads a JSONL file containing generated texts, runs them through
a BERT classifier model, and outputs a distribution of predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    TrainingArguments,
)

from config import CONFIG
from utils import (
    BertClassifierTrainer,
    select_device,
    setup_logging,
)

LOGGER = logging.getLogger("bert_evaluate")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to JSONL file containing generated texts.",
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing the trained BERT classifier model.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Path to save the prediction distribution results. If not provided, prints to stdout.",
    )
    parser.add_argument(
        "--confidences-file",
        type=Path,
        help="Optional path to save per-text prediction confidences for each class.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for classification.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run classification on.",
    )
    parser.add_argument(
        "--text-field",
        choices=["completion", "full_text", "text"],
        default="completion",
        help="Which field from the JSONL to use as text.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum sequence length for tokenization. Defaults to model config or CONFIG.tokenizers.bert_max_length.",
    )
    parser.add_argument(
        "--logging-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def load_generated_texts(path: Path, text_field: str) -> List[str]:
    """Load generated text samples from a JSONL file."""
    texts: List[str] = []
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    LOGGER.info("Loading generated texts from %s", path)
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                LOGGER.warning("Skipping invalid JSON on line %d: %s", line_num, e)
                continue
            
            # Handle the format from g0_generate.py
            if "samples" in record:
                # Multiple samples per record
                for sample in record["samples"]:
                    text = sample.get(text_field, "")
                    if text:
                        texts.append(text)
            elif text_field in record:
                # Direct field in record
                text = record.get(text_field, "")
                if text:
                    texts.append(text)
            elif "text" in record:
                # Fallback to "text" field
                text = record.get("text", "")
                if text:
                    texts.append(text)
    
    LOGGER.info("Loaded %d text samples from file.", len(texts))
    return texts


def read_label_mapping_from_model(model_dir: Path) -> tuple[Dict[str, int], Dict[int, str]]:
    """Read label mapping from the model's config.json file."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found at {config_path}.")
    
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_label2id = config.get("label2id", {})
    raw_id2label = config.get("id2label", {})
    
    if not raw_label2id or not raw_id2label:
        raise ValueError(
            f"Model config at {config_path} does not contain label2id/id2label mappings. "
            "This model may not be a classification model or may be missing required metadata."
        )
    
    # Convert to proper types (config may have string keys/values)
    label2id = {str(name): int(idx) for name, idx in raw_label2id.items()}
    id2label = {int(idx): str(name) for idx, name in raw_id2label.items()}
    
    LOGGER.info("Loaded label mapping: %d classes", len(label2id))
    return label2id, id2label


def classify_texts(
    texts: Sequence[str],
    model_dir: Path,
    batch_size: int,
    device: torch.device,
    max_length: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Classify texts using the BERT model.
    
    Returns:
        Tuple of (predicted_labels, prediction_logits, id2label)
    """
    # Load label mapping from model config
    label2id, id2label = read_label_mapping_from_model(model_dir)
    
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
        # Try to get from model config, fallback to CONFIG
        model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        max_length = model_config.get("max_position_embeddings", CONFIG.tokenizers.bert_max_length)
    
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
    pred_logits = predictions.predictions  # Raw logits for confidence calculation
    
    return pred_labels, pred_logits, id2label


def compute_confidence_scores(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities (confidence scores) from logits."""
    # Apply softmax to convert logits to probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probabilities


def format_distribution(
    predictions: np.ndarray,
    id2label: Dict[int, str],
) -> str:
    """Format the prediction distribution as a readable string."""
    counter = Counter(int(pred) for pred in predictions)
    total = len(predictions)
    
    lines = ["Prediction Distribution:", "=" * 50]
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


def save_results(
    output_path: Path,
    predictions: np.ndarray,
    id2label: Dict[int, str],
    texts: List[str],
) -> None:
    """Save detailed results to a JSON file."""
    results = {
        "total_samples": len(predictions),
        "distribution": {},
        "predictions": [],
    }
    
    counter = Counter(int(pred) for pred in predictions)
    total = len(predictions)
    
    for label_id in sorted(id2label.keys()):
        label_name = id2label[label_id]
        count = counter.get(label_id, 0)
        percentage = (count / total * 100) if total > 0 else 0.0
        results["distribution"][label_name] = {
            "count": int(count),
            "percentage": float(percentage),
        }
    
    # Include individual predictions (truncated text for brevity)
    for i, (text, pred_id) in enumerate(zip(texts, predictions)):
        results["predictions"].append({
            "index": i,
            "predicted_label": id2label[int(pred_id)],
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved detailed results to %s", output_path)


def save_confidence_scores(
    output_path: Path,
    texts: List[str],
    confidences: np.ndarray,
    id2label: Dict[int, str],
) -> None:
    """Save per-text confidence scores for each class."""
    results = {
        "total_samples": len(texts),
        "label_names": {str(idx): label for idx, label in id2label.items()},
        "predictions": [],
    }
    
    for i, (text, conf_row) in enumerate(zip(texts, confidences)):
        # Build confidence dict for this text
        conf_dict = {}
        for label_id in sorted(id2label.keys()):
            conf_dict[id2label[label_id]] = float(conf_row[label_id])
        
        # Find predicted label (highest confidence)
        pred_label_id = int(np.argmax(conf_row))
        
        results["predictions"].append({
            "index": i,
            "predicted_label": id2label[pred_label_id],
            "predicted_confidence": float(conf_row[pred_label_id]),
            "confidences": conf_dict,
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved confidence scores to %s", output_path)


def main() -> None:
    """Main function."""
    args = parse_args()
    setup_logging(args.logging_level)
    LOGGER.info("Arguments: %s", args)
    
    # Validate inputs
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    # Load texts
    texts = load_generated_texts(args.input_file, args.text_field)
    if not texts:
        raise ValueError("No texts found in input file.")
    
    # Setup device
    device = select_device(args.device)
    LOGGER.info("Using device: %s", device)
    
    # Classify
    predictions, logits, id2label = classify_texts(
        texts,
        args.model_dir,
        args.batch_size,
        device,
        args.max_length,
    )
    
    # Format distribution
    distribution_text = format_distribution(predictions, id2label)
    
    # Output results
    if args.output_file:
        # Save detailed JSON results
        save_results(args.output_file, predictions, id2label, texts)
        
        # Also save human-readable distribution
        dist_path = args.output_file.with_suffix(".txt")
        dist_path.write_text(distribution_text, encoding="utf-8")
        LOGGER.info("Saved distribution to %s", dist_path)
    else:
        # Print to stdout
        print(distribution_text)
    
    # Save confidence scores if requested
    if args.confidences_file:
        confidences = compute_confidence_scores(logits)
        save_confidence_scores(args.confidences_file, texts, confidences, id2label)


if __name__ == "__main__":
    main()

