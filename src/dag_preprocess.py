"""Preprocess dataset for the augmented discriminator D_ag.

This script creates a dataset for training D_ag, which classifies text as:
- Target author
- Distractor authors  
- Generated text (from a specified generator)

It loads the existing D_aa dataset and adds generated text samples to create
a multiclass classification dataset with an additional "generated" class.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast

from config import CONFIG
from utils import (
    load_dataset_dict,
    read_label_mapping,
    setup_logging,
)

LOGGER = logging.getLogger("dag_preprocess")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daa-dataset-dir",
        type=Path,
        default=CONFIG.paths.discriminator_dataset_dir,
        help="Path to the existing D_aa discriminator dataset.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=CONFIG.paths.metadata_dir,
        help="Path to metadata directory with label mappings.",
    )
    parser.add_argument(
        "--generated-file",
        type=Path,
        required=True,
        help="Path to JSONL file with generated texts (from gpt_generate.py or raw GPT-2).",
    )
    parser.add_argument(
        "--num-generated-samples",
        type=int,
        required=True,
        help="Number of generated text samples to use from the file.",
    )
    parser.add_argument(
        "--generated-train-ratio",
        type=float,
        default=0.5,
        help="Ratio of generated samples to use for training (rest for validation).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONFIG.paths.dag_dataset_dir,
        help="Directory to save the D_ag dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing dataset before writing new one.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG.random_seed,
        help="Random seed for reproducibility (used for shuffling samples).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def load_generated_texts_from_jsonl(path: Path, max_samples: int, seed: int) -> List[str]:
    """Load generated text completions from a JSONL file.
    
    Args:
        path: Path to JSONL file
        max_samples: Maximum number of samples to load
        seed: Random seed for shuffling (to ensure consistent selection)
    
    Returns:
        List of generated text completions
    """
    texts: List[str] = []
    if not path.exists():
        raise FileNotFoundError(f"Generated text file not found: {path}")
    
    LOGGER.info("Loading generated texts from %s (max %d samples)", path, max_samples)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                LOGGER.warning("Skipping invalid JSON line: %s", e)
                continue
            
            # Handle both formats: direct samples or nested structure
            if "samples" in record:
                for sample in record["samples"]:
                    completion = sample.get("completion", "")
                    if completion:
                        texts.append(completion)
            elif "completion" in record:
                texts.append(record["completion"])
            elif "full_text" in record:
                texts.append(record["full_text"])
    
    LOGGER.info("Loaded %d generated text samples from file.", len(texts))
    
    if len(texts) < max_samples:
        raise ValueError(
            f"Requested {max_samples} samples but only {len(texts)} found in file."
        )
    
    # Shuffle with seed for consistent selection, then take first max_samples
    random.seed(seed)
    random.shuffle(texts)
    texts = texts[:max_samples]
    
    LOGGER.info("Selected %d samples for dataset.", len(texts))
    return texts


def build_label_mapping_with_generated(
    target_author: str,
    distractors: Sequence[str],
) -> Dict[str, int]:
    """Create label mapping including the generated class."""
    authors = [target_author, *distractors]
    label2id = {author: idx for idx, author in enumerate(authors)}
    # Add generated class as the last label
    label2id["generated"] = len(authors)
    return label2id


def create_dag_dataset(
    daa_dataset: DatasetDict,
    generated_texts: List[str],
    generated_train_ratio: float,
    label2id: Dict[str, int],
    tokenizer: BertTokenizerFast,
    max_length: int,
    pad_to_max: bool,
) -> DatasetDict:
    """Create the D_ag dataset by combining D_aa data with generated texts."""
    
    # Get existing records from D_aa dataset
    # Map author names to new labels (labels should match since we preserve author order)
    train_records = [
        {
            "text": str(record["text"]),
            "author": str(record["author"]),
            "label": label2id.get(str(record["author"]), int(record["label"])),
            "split": "train",
            "source_file": str(record.get("source_file", "")),
        }
        for record in daa_dataset["train"]
    ]
    
    eval_records = [
        {
            "text": str(record["text"]),
            "author": str(record["author"]),
            "label": label2id.get(str(record["author"]), int(record["label"])),
            "split": "validation",
            "source_file": str(record.get("source_file", "")),
        }
        for record in daa_dataset["validation"]
    ]
    
    # Add generated texts
    generated_label = label2id["generated"]
    num_train_generated = int(len(generated_texts) * generated_train_ratio)
    
    for i, text in enumerate(generated_texts[:num_train_generated]):
        train_records.append({
            "text": text,
            "author": "generated",
            "label": generated_label,
            "split": "train",
            "source_file": f"generated_{i}",
        })
    
    for i, text in enumerate(generated_texts[num_train_generated:]):
        eval_records.append({
            "text": text,
            "author": "generated",
            "label": generated_label,
            "split": "validation",
            "source_file": f"generated_{num_train_generated + i}",
        })
    
    LOGGER.info(
        "Created dataset with %d train samples (%d generated) and %d eval samples (%d generated).",
        len(train_records),
        num_train_generated,
        len(eval_records),
        len(generated_texts) - num_train_generated,
    )
    
    # Create dataset dict
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(eval_records),
    })
    
    # Tokenize
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
        desc="Tokenising D_ag dataset",
    )
    
    return tokenised


def save_metadata(
    metadata_dir: Path,
    overwrite: bool,
    label2id: Dict[str, int],
    train_records: Sequence[Dict],
    eval_records: Sequence[Dict],
) -> None:
    """Save metadata for the D_ag dataset."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    label_path = metadata_dir / "dag_label_mapping.json"
    if label_path.exists() and not overwrite:
        raise FileExistsError(
            f"Label mapping already exists at {label_path}. Use --overwrite."
        )
    
    id2label = {str(idx): author for author, idx in label2id.items()}
    label_payload = {
        "target_author": CONFIG.preprocess.target_author,
        "label2id": label2id,
        "id2label": id2label,
    }
    label_path.write_text(json.dumps(label_payload, indent=2), encoding="utf-8")
    LOGGER.info("Saved label mapping to %s", label_path)
    
    # Summary
    summary_path = metadata_dir / "dag_preprocess_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(
            f"Summary already exists at {summary_path}. Use --overwrite."
        )
    
    train_counts = Counter(record["author"] for record in train_records)
    eval_counts = Counter(record["author"] for record in eval_records)
    
    summary_payload = {
        "config": {
            "preprocess": asdict(CONFIG.preprocess),
            "tokenizers": asdict(CONFIG.tokenizers),
        },
        "train_counts": dict(train_counts),
        "validation_counts": dict(eval_counts),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    LOGGER.info("Saved preprocessing summary to %s", summary_path)


def main() -> None:
    args = parse_args()
    setup_logging("DEBUG" if args.verbose else "INFO")
    
    LOGGER.info("Starting D_ag dataset preprocessing.")
    random.seed(args.seed)
    
    # Validate inputs
    if not args.daa_dataset_dir.exists():
        raise FileNotFoundError(f"D_aa dataset not found: {args.daa_dataset_dir}")
    if not args.metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {args.metadata_dir}")
    
    # Load existing D_aa dataset and metadata
    daa_dataset = load_dataset_dict(args.daa_dataset_dir)
    label2id_old, id2label_old = read_label_mapping(args.metadata_dir)
    
    # Determine authors
    target_author = CONFIG.preprocess.target_author
    distractors = [author for author in label2id_old.keys() if author != target_author]
    
    # Build new label mapping with generated class
    label2id = build_label_mapping_with_generated(target_author, distractors)
    
    LOGGER.info(
        "Target author: %s, Distractors: %d, Generated class label: %d",
        target_author,
        len(distractors),
        label2id["generated"],
    )
    
    # Load generated text samples from JSONL file
    generated_texts = load_generated_texts_from_jsonl(
        args.generated_file,
        args.num_generated_samples,
        args.seed,
    )
    
    # Initialize tokenizer
    LOGGER.info("Initializing BERT tokenizer.")
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG.tokenizers.bert_model_name)
    
    # Create dataset
    dag_dataset = create_dag_dataset(
        daa_dataset,
        generated_texts,
        args.generated_train_ratio,
        label2id,
        tokenizer,
        CONFIG.tokenizers.bert_max_length,
        CONFIG.tokenizers.pad_to_max_length,
    )
    
    # Prepare output directory
    if args.output_dir.exists():
        if args.overwrite:
            LOGGER.info("Removing existing dataset at %s", args.output_dir)
            shutil.rmtree(args.output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists at {args.output_dir}. Use --overwrite."
            )
    
    # Save dataset
    LOGGER.info("Saving D_ag dataset to %s", args.output_dir)
    dag_dataset.save_to_disk(str(args.output_dir))
    
    # Save metadata
    train_records = [
        {
            "author": record["author"],
        }
        for record in dag_dataset["train"]
    ]
    eval_records = [
        {
            "author": record["author"],
        }
        for record in dag_dataset["validation"]
    ]
    
    save_metadata(
        args.metadata_dir,
        args.overwrite,
        label2id,
        train_records,
        eval_records,
    )
    
    LOGGER.info("D_ag preprocessing complete.")


if __name__ == "__main__":
    arguments = parse_args()
    main()

