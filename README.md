# Author Imitation Project

This project implements an iterative generator-discriminator training loop for author imitation. The system trains GPT-2 generators to mimic a target author's writing style while using BERT-based discriminators to identify generated text. Using a technique similar tohard negative mining, we aim to iteratively improve the generator model by training it on generated text that the discriminator classifies as the target author.

## Overview

The project consists of several key components:

- **G_0**: Baseline GPT-2 generator trained on target author's corpus
- **D_aa**: Multi-class BERT discriminator for authorship attribution
- **D_ag**: Augmented discriminator that includes a "generated" class
- **Main Loop**: Iterative training where generators and discriminators co-evolve through hard negative mining

## Project Structure

```
author_imitation/
├── data/
│   ├── reuter/              # Raw Reuters C50 dataset
│   ├── processed/           # Processed datasets
│   └── generated/           # Generated text samples
├── models/                  # Trained models
├── results/                 # Evaluation results
└── src/
    ├── config.py            # Global configuration
    ├── preprocess.py        # Dataset preprocessing
    ├── g0_train.py          # Train baseline generator
    ├── daa_train.py         # Train authorship attribution discriminator
    ├── dag_preprocess.py    # Prepare augmented discriminator dataset
    ├── dag_train.py         # Train augmented discriminator
    ├── gpt_generate.py      # Generate text samples
    ├── bert_evaluate.py     # Evaluate generated texts
    └── main_loop.py         # Main iterative training loop
```

## Setup

### Installation

Install required packages:

```bash
pip install -r requirements.txt
```

### Data Preparation

Place the Reuters C50 dataset in `data/reuter/` with the following structure:

```
data/reuter/
├── C50train/
│   ├── Author1/
│   │   ├── file1.txt
│   │   └── ...
│   └── ...
└── C50test/
    ├── Author1/
    │   ├── file1.txt
    │   └── ...
    └── ...
```

## Configuration

The project uses a centralized configuration system in `src/config.py`. Key settings include:

- **Target Author**: Set `CONFIG.preprocess.target_author` (default: "AaronPressman")
- **Model Paths**: All paths are configured in `CONFIG.paths`
- **Training Hyperparameters**: Configured in respective config classes

To modify settings, edit `src/config.py` or override via command-line arguments.

## Usage

### Step 1: Preprocess Data

Preprocess the raw Reuters dataset to create training datasets for the generator and discriminator:

```bash
python src/preprocess.py --overwrite
```

**Options:**
- `--overwrite`: Remove existing processed datasets before writing new ones
- `--verbose`: Enable verbose logging

**Output:**
- Generator dataset: `data/processed/reuter/generator_g0/`
- Discriminator dataset: `data/processed/reuter/discriminator_daa/`
- Metadata: `data/processed/reuter/metadata/`

### Step 2: Train Pure Authorship Attribution Discriminator (D_aa)

Train a BERT classifier to distinguish between target author and distractors:

```bash
python src/daa_train.py --num-train-epochs 10 --classification-report
```

**Options:**
- `--dataset-dir PATH`: Path to discriminator dataset (default: from config)
- `--metadata-dir PATH`: Path to metadata directory (default: from config)
- `--output-dir PATH`: Directory to save model (default: `models/bert_Daa`)
- `--model-name NAME`: Base BERT model (default: "bert-base-uncased")
- `--num-train-epochs FLOAT`: Number of training epochs (default: 4.0)
- `--learning-rate FLOAT`: Learning rate (default: 2e-5)
- `--per-device-train-batch-size INT`: Batch size (default: 8)
- `--classification-report`: Generate detailed classification report
- `--overwrite-output-dir`: Overwrite existing output directory

**Output:** Model saved to `models/bert_Daa/` with training metrics and optional classification report


### Step 3: Train Baseline Generator (G_0)

Train the baseline GPT-2 generator on the target author's corpus:

```bash
python src/g0_train.py --num-train-epochs 10
```

**Options:**
- `--dataset-dir PATH`: Path to generator dataset (default: from config)
- `--output-dir PATH`: Directory to save model (default: `models/gpt2_G0`)
- `--model-name NAME`: Base GPT-2 model (default: "gpt2-large")
- `--num-train-epochs FLOAT`: Number of training epochs (default: 10.0)
- `--learning-rate FLOAT`: Learning rate (default: 5e-5)
- `--per-device-train-batch-size INT`: Batch size (default: 2)
- `--gradient-accumulation-steps INT`: Gradient accumulation (default: 16)
- `--fp16`: Enable FP16 mixed precision
- `--bf16`: Enable BF16 mixed precision
- `--overwrite-output-dir`: Overwrite existing output directory

**Another Example:**
```bash
python src/g0_train.py \
    --num-train-epochs 10 \
    --learning-rate 5e-5 \
    --per-device-train-batch-size 2 \
    --gradient-accumulation-steps 16 \
    --output-dir models/gpt2_G0_10e
```

**Output:** Model saved to `models/gpt2_G0/` (or specified directory)

### Step 4: Create Prompts

Create prompts from the test set author's corpus:

```bash
python src/create_prompts.py AaronPressman -n 10 -k 10 -m 10 -o data/prompts/100prompts_10words.txt
```

**Options:**

- `person_name`: Name of the person (directory name in C50test, e.g., "AaronPressman")
- `-n`: Number of words to extract (default: 10)
- `-k`: Number of people to process (default: 10)
- `-m`: Maximum number of text files to process from each person's directory (default: all files)
- `-o`: Output file (default: data/prompts/prompts.txt)
- `--data-dir`: Path to C50test directory (default: data/reuter/C50test)

**Output:** Text file with prompts

### Step 5: Generate Text Samples From Untrained Generator

Generate text samples using the trained generator:

```bash
python src/gpt_generate.py --model-name gpt2-large \
 --prompt-file data/prompts/100prompts_10words.txt \
 --output-file data/generated/gpt2l_100.jsonl
```

```bash
python src/gpt_generate.py --model-name gpt2-large --auto-prompt --num-auto-prompts 100 --output-file data/generated/gpt2l_100.jsonl
```

**Options:**
- `--model-dir PATH`: Path to fine-tuned GPT-2 model (required if not using `--model-name`)
- `--model-name NAME`: Use raw pre-trained GPT-2 model (e.g., "gpt2-large")
- `--auto-prompt`: Automatically generate prompts from target author texts
- `--num-auto-prompts INT`: Number of prompts when using `--auto-prompt` (default: 1)
- `--prompt TEXT`: Manual prompt text (can be repeated)
- `--prompt-file PATH`: File containing prompts (one per line)
- `--max-new-tokens INT`: Maximum tokens to generate (default: 200)
- `--min-new-tokens INT`: Minimum tokens to generate (default: 0)
- `--temperature FLOAT`: Sampling temperature (default: 0.8)
- `--top-k INT`: Top-k sampling (default: 50)
- `--top-p FLOAT`: Nucleus sampling (default: 0.95)
- `--output-file PATH`: Output JSONL file (default: stdout)

**Output:** JSONL file with generated samples containing `prompt`, `completion`, and `full_text` fields


### Step 6: Prepare Augmented Discriminator Dataset (D_ag)

Create a dataset for the augmented discriminator that includes a "generated" class:

```bash
python src/dag_preprocess.py \
    --generated-file data/generated/gpt2l_100.jsonl \
    --num-generated-samples 100 \
    --generated-train-ratio 0.5 \
    --output-dir data/processed/reuter/discriminator_dag
```

**Options:**
- `--daa-dataset-dir PATH`: Path to D_aa dataset (default: from config)
- `--metadata-dir PATH`: Path to metadata directory (default: from config)
- `--generated-file PATH`: Path to JSONL file with generated texts (required)
- `--num-generated-samples INT`: Number of generated samples to use (required)
- `--generated-train-ratio FLOAT`: Ratio for train/validation split (default: 0.5)
- `--output-dir PATH`: Directory to save D_ag dataset (default: from config)
- `--overwrite`: Remove existing dataset before writing
- `--seed INT`: Random seed for reproducibility

**Output:** Dataset saved to `data/processed/reuter/discriminator_dag/` with metadata in `data/processed/reuter/metadata/dag_label_mapping.json`

### Step 7: Train Augmented Discriminator (D_ag)

Train a BERT classifier with generated text as a negative class:

```bash
python src/dag_train.py --dataset-dir data/processed/reuter/discriminator_dag --output-dir models/bert_Dag --classification-report --num-train-epochs 10
```

**Options:**
- `--dataset-dir PATH`: Path to discriminator dataset (default: from config)
- `--metadata-dir PATH`: Path to metadata directory (default: from config)
- `--output-dir PATH`: Directory to save model (default: `models/bert_D0`)
- `--model-name NAME`: Base BERT model (default: "bert-base-uncased")
- `--num-train-epochs FLOAT`: Number of training epochs (default: 4.0)
- `--learning-rate FLOAT`: Learning rate (default: 2e-5)
- `--per-device-train-batch-size INT`: Batch size (default: 8)
- `--classification-report`: Generate detailed classification report
- `--overwrite-output-dir`: Overwrite existing output directory

### Step 8: Main Iterative Training Loop

Run the main loop that iteratively trains generators using hard negative mining (this will take a while):

```bash
python src/main_loop.py \
    --g0-model-dir models/gpt2_G0 \
    --dag-model-dir models/bert_Dag \
    --g0-dataset-dir data/processed/reuter/generator_g0 \
    --daa-dataset-dir data/processed/reuter/discriminator_daa \
    --metadata-dir data/processed/reuter/metadata \
    --num-iterations 5 \
    --num-sequences 300 \
    --prompt-num-words 10
```

**Required Arguments:**
- `--g0-model-dir PATH`: Path to trained G₀ model
- `--dag-model-dir PATH`: Path to trained D_ag model
- `--g0-dataset-dir PATH`: Path to original G_0 training dataset
- `--daa-dataset-dir PATH`: Path to original D_aa training dataset. Used to build prompts.
- `--metadata-dir PATH`: Path to metadata directory with label mappings

**Loop Configuration:**
- `--num-iterations INT`: Number of loop iterations (default: 10)
- `--num-sequences INT`: Number of sequences to generate per iteration (default: 300)
- `--prompt-num-words INT`: Number of words to use for prompts (default: 10)
- `--generation-max-new-tokens INT`: Maximum tokens to generate (default: 200)
- `--generation-min-new-tokens INT`: Minimum tokens to generate (default: 20)
- `--generation-temperature FLOAT`: Generation temperature (default: 0.8)
- `--hard-negative-threshold FLOAT`: Confidence threshold for hard negatives (default: 0.15)
- `--num-hard-negs-per-iteration INT`: Maximum number of hard negatives to use per iteration (default: 10)
- `--output-root PATH`: Root directory for saving iterations (default: `models/loop`)

**Output:** Each iteration creates:
- `models/loop/iteration_N/`: Directory for iteration N
  - `hard_negatives.jsonl`: Generated samples that fooled the discriminator
  - `classification_distribution.txt`: Distribution of predictions
  - `generator_dataset/`: Augmented generator dataset
  - `discriminator_dataset/`: Augmented discriminator dataset
  - `generators/G_N/`: Trained generator for iteration N

### Step 9: Evaluate the Generator

First create a dataset of generated texts from the latest generator:

```bash
python src/gpt_generate.py --model-dir models/iteration_N/generators/G_N --prompt-file data/prompts/100prompts_10words.txt --output-file data/generated/gN_100.jsonl
```

Evluate the generator using the holdout discriminator:

```bash
python src/bert_evaluate.py data/generated/gN_100.jsonl models/bert_Daa --output-file results/iteration_N/gN_100_distribution.txt --confidences-file results/iteration_N/gN_100_confidences.jsonl
```

**Options:**
- `input-file PATH`: Path to JSONL file containing generated texts
- `model-dir PATH`: Path to discriminator model
- `--output-file PATH`: Path to save the prediction distribution results
- `--confidences-file PATH`: Path to save per-text prediction confidences for each class
- `--batch-size INT`: Batch size for classification (default: 32)
- `--device CHOICE`: Device to run classification on (default: "auto")
- `--text-field CHOICE`: Which field from the JSONL to use as text (default: "completion" or "full_text" or "text")
- `--max-length INT`: Maximum sequence length for tokenization (Defaults to model config or CONFIG.tokenizers.bert_max_length.)
- `--logging-level CHOICE`: Logging verbosity level (default: "INFO")

**Output:**
- `results/iteration_N/gN_100_distribution.txt`: Prediction distribution results
- `results/iteration_N/gN_100_confidences.jsonl`: Per-text prediction confidences for each class

## Tips and Troubleshooting

### Memory Issues
- Reduce batch sizes: `--per-device-train-batch-size 1`
- Enable gradient accumulation: `--gradient-accumulation-steps 16`
- Use gradient checkpointing: `--gradient-checkpointing`
- Use mixed precision: `--fp16` or `--bf16`

### Training Speed
- Increase batch size if memory allows
- Use multiple GPUs if available
- Reduce number of samples per iteration in main loop

### Model Selection
- Use `gpt2-large` for better generation quality (default)
- Use `gpt2` or `gpt2-medium` for faster training
- Adjust learning rates based on model size

## Configuration Reference

Key configuration options in `src/config.py`:

- **Target Author**: `CONFIG.preprocess.target_author = "AaronPressman"`
- **Number of Distractors**: `CONFIG.preprocess.num_distractor_authors = 9`
- **GPT-2 Model**: `CONFIG.tokenizers.gpt2_model_name = "gpt2-large"`
- **BERT Model**: `CONFIG.tokenizers.bert_model_name = "bert-base-uncased"`
- **Block Size**: `CONFIG.tokenizers.gpt2_block_size = 512`
- **Max Length**: `CONFIG.tokenizers.bert_max_length = 512`
