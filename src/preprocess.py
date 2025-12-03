"""Preprocess the Reuters C50 corpus for generator and discriminator training.

The script prepares two datasets:

1. GPT-2 causal language modelling dataset for the baseline generator (G_0).
2. BERT sequence classification dataset for the holdout authorship attribution
   discriminator (D_aa).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast, GPT2TokenizerFast

from config import CONFIG

LOGGER = logging.getLogger("preprocess")
WORD_PATTERN = re.compile(r"\w+")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing processed datasets before writing new ones.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """Configure log formatting."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def select_authors(train_dir: Path) -> Tuple[str, List[str]]:
    """Select the target author and distractors.

    Returns a tuple of the target author name and a list of distractor author
    names.
    """

    cfg = CONFIG.preprocess
    available_authors = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if cfg.target_author not in available_authors:
        raise ValueError(
            f"Target author '{cfg.target_author}' not found in {train_dir}."
        )

    if cfg.distractor_authors is not None:
        configured = [a for a in cfg.distractor_authors if a != cfg.target_author]
        missing = sorted(set(configured) - set(available_authors))
        if missing:
            raise ValueError(
                "Configured distractor authors missing from dataset: "
                + ", ".join(missing)
            )
        distractors = configured
    else:
        distractors = [a for a in available_authors if a != cfg.target_author]
        if cfg.num_distractor_authors is not None:
            if cfg.num_distractor_authors > len(distractors):
                raise ValueError(
                    "Requested more distractor authors than available "
                    f"({cfg.num_distractor_authors} > {len(distractors)})."
                )
            distractors = distractors[: cfg.num_distractor_authors]

    LOGGER.info(
        "Selected %s as target with %d distractor authors.",
        cfg.target_author,
        len(distractors),
    )
    return cfg.target_author, distractors


def build_label_mapping(target_author: str, distractors: Sequence[str]) -> Dict[str, int]:
    """Create a deterministic label mapping for the discriminator."""

    authors = [target_author, *distractors]
    return {author: idx for idx, author in enumerate(authors)}


def read_text_file(path: Path) -> str:
    """Read a text file as UTF-8, falling back to replacement characters."""

    return path.read_text(encoding="utf-8", errors="replace")


def normalise_text(text: str) -> str:
    """Apply basic normalisation driven by configuration."""

    cfg = CONFIG.preprocess
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if cfg.lowercase:
        text = text.lower()

    if cfg.compress_whitespace:
        if cfg.keep_newlines:
            text = re.sub(r"[ \t\f\v]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" ?\n ?", "\n", text)
        else:
            text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_valid_sample(text: str) -> bool:
    """Check whether a sample meets minimum length requirements."""

    cfg = CONFIG.preprocess
    if not text:
        return False
    if cfg.min_chars_per_sample and len(text) < cfg.min_chars_per_sample:
        return False
    if cfg.min_words_per_sample:
        word_count = len(WORD_PATTERN.findall(text))
        if word_count < cfg.min_words_per_sample:
            return False
    return True


def collect_records(
    split_dir: Path,
    authors: Sequence[str],
    split_name: str,
    label2id: Dict[str, int],
    max_docs: Optional[int],
) -> List[Dict[str, str]]:
    """Collect and clean raw documents for a given split."""

    records: List[Dict[str, str]] = []
    for author in authors:
        author_dir = split_dir / author
        if not author_dir.exists():
            LOGGER.warning("Skipping missing author directory: %s", author_dir)
            continue

        file_paths = sorted(author_dir.glob("*.txt"))
        if max_docs is not None:
            file_paths = file_paths[:max_docs]

        for path in file_paths:
            raw_text = read_text_file(path)
            cleaned_text = normalise_text(raw_text)
            if not is_valid_sample(cleaned_text):
                continue

            records.append(
                {
                    "text": cleaned_text,
                    "author": author,
                    "label": label2id[author],
                    "split": split_name,
                    "source_file": str(path.relative_to(CONFIG.paths.project_root)),
                }
            )

    LOGGER.info(
        "Collected %d documents for split '%s'.",
        len(records),
        split_name,
    )
    return records


def ensure_examples_per_author(records: Sequence[Dict[str, str]], authors: Sequence[str]) -> None:
    """Ensure each author has at least one retained sample."""

    by_author = Counter(record["author"] for record in records)
    missing = [author for author in authors if by_author.get(author, 0) == 0]
    if missing:
        raise ValueError(
            "No valid samples retained for author(s): " + ", ".join(missing)
        )


def create_generator_dataset(
    texts: Sequence[str],
    tokenizer: GPT2TokenizerFast,
    block_size: int,
    stride: int,
) -> Tuple[Dataset, Dict[str, int]]:
    """Tokenise target texts for GPT-2 causal language modelling."""

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

    stats = {
        "num_examples": len(examples),
        "total_tokens": total_length,
        "block_size": block_size,
        "stride": stride,
    }
    return dataset, stats


def create_discriminator_dataset(
    train_records: Sequence[Dict[str, str]],
    eval_records: Sequence[Dict[str, str]],
    tokenizer: BertTokenizerFast,
    max_length: int,
    pad_to_max: bool,
) -> DatasetDict:
    """Tokenise documents for the BERT-based discriminator."""

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(list(train_records)),
            "validation": Dataset.from_list(list(eval_records)),
        }
    )

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
        desc="Tokenising discriminator dataset",
    )
    return tokenised


def prepare_output_path(path: Path, overwrite: bool, description: str) -> None:
    """Ensure the output path is ready for writing."""

    if path.exists():
        if overwrite:
            LOGGER.info("Removing existing %s at %s", description, path)
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"{description.capitalize()} already exists at {path}. "
                "Use --overwrite to replace it."
            )


def save_metadata(
    metadata_dir: Path,
    overwrite: bool,
    label2id: Dict[str, int],
    generator_stats: Dict[str, int],
    train_records: Sequence[Dict[str, str]],
    eval_records: Sequence[Dict[str, str]],
) -> None:
    """Persist human-readable metadata about preprocessing."""

    metadata_dir.mkdir(parents=True, exist_ok=True)

    label_path = metadata_dir / "label_mapping.json"
    if label_path.exists() and overwrite:
        label_path.unlink()
    elif label_path.exists() and not overwrite:
        raise FileExistsError(
            f"Metadata file already exists at {label_path}. Use --overwrite."
        )

    id2label = {str(idx): author for author, idx in label2id.items()}
    label_payload = {
        "target_author": CONFIG.preprocess.target_author,
        "label2id": label2id,
        "id2label": id2label,
    }
    label_path.write_text(json.dumps(label_payload, indent=2), encoding="utf-8")

    summary_path = metadata_dir / "preprocess_summary.json"
    if summary_path.exists() and overwrite:
        summary_path.unlink()
    elif summary_path.exists() and not overwrite:
        raise FileExistsError(
            f"Metadata file already exists at {summary_path}. Use --overwrite."
        )

    train_counts = Counter(record["author"] for record in train_records)
    eval_counts = Counter(record["author"] for record in eval_records)

    summary_payload = {
        "config": {
            "preprocess": asdict(CONFIG.preprocess),
            "tokenizers": asdict(CONFIG.tokenizers),
        },
        "generator_stats": generator_stats,
        "train_counts": dict(train_counts),
        "validation_counts": dict(eval_counts),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    target_corpus_path = metadata_dir / "target_corpus_train.txt"
    if target_corpus_path.exists() and overwrite:
        target_corpus_path.unlink()
    elif target_corpus_path.exists() and not overwrite:
        raise FileExistsError(
            f"Metadata file already exists at {target_corpus_path}. Use --overwrite."
        )

    target_train_texts = [
        record["text"]
        for record in train_records
        if record["author"] == CONFIG.preprocess.target_author
    ]
    target_corpus_path.write_text("\n\n".join(target_train_texts), encoding="utf-8")


def main(overwrite: bool, verbose: bool) -> None:
    setup_logging(verbose)
    cfg = CONFIG

    LOGGER.info("Using configuration: %s", cfg)
    random.seed(cfg.random_seed)

    raw_root = cfg.paths.raw_reuters_dir
    train_dir = raw_root / "C50train"
    eval_dir = raw_root / "C50test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    target_author, distractor_authors = select_authors(train_dir)
    authors = [target_author, *distractor_authors]
    label2id = build_label_mapping(target_author, distractor_authors)

    train_records = collect_records(
        train_dir,
        authors,
        "train",
        label2id,
        cfg.preprocess.max_train_docs_per_author,
    )
    eval_records = collect_records(
        eval_dir,
        authors,
        "validation",
        label2id,
        cfg.preprocess.max_eval_docs_per_author,
    )

    ensure_examples_per_author(train_records, authors)
    ensure_examples_per_author(eval_records, authors)

    LOGGER.info("Initialising tokenizers.")
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(cfg.tokenizers.gpt2_model_name)
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    bert_tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizers.bert_model_name)

    generator_texts = [
        record["text"]
        for record in train_records
        if record["author"] == target_author
    ]
    if cfg.preprocess.use_test_split_for_generator:
        generator_texts.extend(
            record["text"]
            for record in eval_records
            if record["author"] == target_author
        )

    LOGGER.info(
        "Preparing generator dataset with %d documents.", len(generator_texts)
    )
    generator_dataset, generator_stats = create_generator_dataset(
        generator_texts,
        gpt2_tokenizer,
        cfg.tokenizers.gpt2_block_size,
        cfg.tokenizers.gpt2_stride,
    )

    LOGGER.info("Preparing discriminator dataset.")
    discriminator_dataset = create_discriminator_dataset(
        train_records,
        eval_records,
        bert_tokenizer,
        cfg.tokenizers.bert_max_length,
        cfg.tokenizers.pad_to_max_length,
    )

    generator_dir = cfg.paths.generator_dataset_dir
    discriminator_dir = cfg.paths.discriminator_dataset_dir

    prepare_output_path(generator_dir, overwrite, "generator dataset")
    prepare_output_path(discriminator_dir, overwrite, "discriminator dataset")

    LOGGER.info("Saving generator dataset to %s", generator_dir)
    generator_dataset.save_to_disk(str(generator_dir))

    LOGGER.info("Saving discriminator dataset to %s", discriminator_dir)
    discriminator_dataset.save_to_disk(str(discriminator_dir))

    LOGGER.info("Writing metadata to %s", cfg.paths.metadata_dir)
    save_metadata(
        cfg.paths.metadata_dir,
        overwrite,
        label2id,
        generator_stats,
        train_records,
        eval_records,
    )

    LOGGER.info("Preprocessing complete.")


if __name__ == "__main__":
    arguments = parse_args()
    main(overwrite=arguments.overwrite, verbose=arguments.verbose)