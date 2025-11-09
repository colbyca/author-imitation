"""Global configuration for the author imitation project.

This module centralises project-wide settings so that data processing and
training scripts share a consistent source of truth. It currently focuses on
preparing the Reuters C50 dataset but can be extended with additional
configuration blocks as the project grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used across the project."""

    project_root: Path = Path(__file__).resolve().parent.parent
    data_root: Path = project_root / "data"
    raw_reuters_dir: Path = data_root / "reuter"
    processed_root: Path = data_root / "processed"
    reuters_processed_dir: Path = processed_root / "reuter"
    generator_dataset_dir: Path = reuters_processed_dir / "generator_g0"
    discriminator_dataset_dir: Path = reuters_processed_dir / "discriminator_daa"
    dag_dataset_dir: Path = reuters_processed_dir / "discriminator_dag"
    metadata_dir: Path = reuters_processed_dir / "metadata"
    models_root: Path = project_root / "models"
    g0_model_dir: Path = models_root / "gpt2_G0"
    daa_model_dir: Path = models_root / "bert_Daa"
    dag_model_dir: Path = models_root / "bert_Dag"
    loop_root: Path = models_root / "loop"
    loop_generators_dir: Path = loop_root / "generators"
    loop_discriminators_dir: Path = loop_root / "discriminators"
    loop_hard_negatives_dir: Path = loop_root / "hard_negatives"


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings that control how raw text is filtered and selected."""

    target_author: str = "AaronPressman"
    distractor_authors: Optional[List[str]] = None
    num_distractor_authors: int = 9
    max_train_docs_per_author: Optional[int] = None
    max_eval_docs_per_author: Optional[int] = None
    min_chars_per_sample: int = 200
    min_words_per_sample: int = 40
    lowercase: bool = False
    compress_whitespace: bool = True
    keep_newlines: bool = True
    use_test_split_for_generator: bool = False


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer and sequence-length settings for downstream models."""

    gpt2_model_name: str = "gpt2-large"
    gpt2_block_size: int = 512
    gpt2_stride: int = 0
    bert_model_name: str = "bert-base-uncased"
    bert_max_length: int = 512
    pad_to_max_length: bool = True


@dataclass(frozen=True)
class G0TrainingConfig:
    """Default hyperparameters for training the baseline generator."""

    learning_rate: float = 5e-5
    num_train_epochs: float = 10.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 50
    save_steps: int = 5
    save_total_limit: int = 2
    eval_ratio: float = 0.0
    max_steps: Optional[int] = None
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 4
    gradient_checkpointing: bool = False
    report_to: Optional[List[str]] = None


@dataclass(frozen=True)
class GenerationConfig:
    """Default generation controls for G0 text sampling."""

    max_new_tokens: int = 200
    min_new_tokens: int = 0
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    do_sample: bool = True
    num_beams: int = 1
    no_repeat_ngram_size: int = 0
    length_penalty: float = 1.0
    early_stopping: bool = False


@dataclass(frozen=True)
class DaaTrainingConfig:
    """Default hyperparameters for training the authorship attribution model."""

    model_name: str = "bert-base-uncased"
    learning_rate: float = 2e-5
    num_train_epochs: float = 4.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 0
    weight_decay: float = 0.01
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    max_steps: Optional[int] = None
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    report_to: Optional[List[str]] = None
    metric_for_best_model: str = "eval_accuracy"


@dataclass(frozen=True)
class DagTrainingConfig:
    """Default hyperparameters for training the augmented discriminator."""

    model_name: str = "bert-base-uncased"
    learning_rate: float = 2e-5
    num_train_epochs: float = 4.0
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    weight_decay: float = 0.01
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    max_steps: Optional[int] = None
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    report_to: Optional[List[str]] = None
    metric_for_best_model: str = "eval_accuracy"
    generated_eval_ratio: float = 0.1


@dataclass(frozen=True)
class BinaryDiscriminatorConfig:
    """Hyperparameters for binary discriminators used in the main loop."""

    model_name: str = "bert-base-uncased"
    learning_rate: float = 2e-5
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    weight_decay: float = 0.01
    logging_steps: int = 50
    eval_ratio: float = 0.2
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    report_to: Optional[List[str]] = None


@dataclass(frozen=True)
class MainLoopConfig:
    """Controls for the iterative generator/discriminator training loop."""

    num_iterations: int = 10
    samples_per_iteration: int = 200
    generation_max_new_tokens: int = 200
    generation_min_new_tokens: int = 20
    generation_temperature: float = 0.8
    generation_top_k: int = 50
    generation_top_p: float = 0.95
    generation_num_return_sequences: int = 1
    generation_batch_size: int = 2
    prompt_char_limit: int = 200
    hard_negative_ratio: float = 0.1
    hard_negative_threshold: float = 0.15
    max_hard_negatives: Optional[int] = None
    max_generated_history: Optional[int] = 5000
    generator_epochs: float = 10.0
    discriminator_epochs: float = 3.0


@dataclass(frozen=True)
class Config:
    """Top-level configuration container."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    tokenizers: TokenizerConfig = field(default_factory=TokenizerConfig)
    g0_training: G0TrainingConfig = field(default_factory=G0TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    daa_training: DaaTrainingConfig = field(default_factory=DaaTrainingConfig)
    dag_training: DagTrainingConfig = field(default_factory=DagTrainingConfig)
    binary_discriminator: BinaryDiscriminatorConfig = field(default_factory=BinaryDiscriminatorConfig)
    main_loop: MainLoopConfig = field(default_factory=MainLoopConfig)
    random_seed: int = 42


CONFIG = Config()

__all__ = ["CONFIG"]
