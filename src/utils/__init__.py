"""Shared utilities for the author imitation project."""

from .generation import (
    GeneratedSample,
    build_char_prompts,
    build_word_prompts,
    generate_sequences,
    load_generator,
)
from .datasets import (
    collect_texts,
    ensure_split,
    load_dataset_dict,
    load_target_author_texts,
    read_label_mapping,
    texts_for_label,
)
from .system import setup_logging, select_device
from .training import BertClassifierTrainer

__all__ = [
    "collect_texts",
    "ensure_split",
    "GeneratedSample",
    "build_char_prompts",
    "build_word_prompts",
    "generate_sequences",
    "load_dataset_dict",
    "load_generator",
    "load_target_author_texts",
    "BertClassifierTrainer",
    "read_label_mapping",
    "setup_logging",
    "select_device",
    "texts_for_label",
]

