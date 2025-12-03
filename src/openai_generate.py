#!/usr/bin/env python3
"""Generate text continuations using OpenAI API with style emulation.

This script reads prompts from a file and generates 200-token continuations
using OpenAI's API, with examples from AaronPressman's writing style to guide
the generation.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from config import PathsConfig


def load_prompts(prompts_file: Path) -> List[str]:
    """Load prompts from a text file, one per line."""
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def load_example_files(examples_dir: Path, num_examples: int = 5) -> List[str]:
    """Randomly select and load example text files."""
    txt_files = list(examples_dir.glob("*.txt"))
    if len(txt_files) < num_examples:
        raise ValueError(f"Not enough example files. Found {len(txt_files)}, need {num_examples}")
    
    selected_files = random.sample(txt_files, num_examples)
    examples = []
    
    for file_path in selected_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            examples.append(content)
    
    return examples


def build_system_prompt(examples: List[str]) -> str:
    """Build the system prompt with examples."""
    examples_text = "\n\n---\n\n".join(examples)
    
    prompt = """Task:

<SYS> You are an emulator designed to replicate the writing style of a human author.</SYS> Your task is to generate a 200-token continuation that seamlessly integrates with the provided human-authored snippet. Strive to make the continuation indistinguishable from the human-authored text.

Instructions:

The goal of this task is to mimic the author's writing style while paying meticulous attention to lexical richness and diversity, sentence structure, punctuation style, special character style, expressions and idioms, overall tone, emotion and mood, or any other relevant aspect of writing style established by the author.

Output Indicator:

As output, exclusively return the text completion without any accompanying explanations or comments.

Examples:

"""
    prompt += examples_text
    
    return prompt


def generate_completion(
    client: OpenAI,
    prompt: str,
    system_prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 200,
    temperature: float = 0.8
) -> str:
    """Generate a text completion using OpenAI API."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating completion for prompt: {prompt[:50]}...")
        print(f"Error: {e}")
        raise


def format_output(
    prompt: str,
    completion: str,
    generation_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Format the output to match the expected JSONL structure."""
    full_text = prompt + " " + completion
    
    return {
        "prompt": prompt,
        "samples": [{
            "prompt_text": prompt,
            "completion": completion,
            "full_text": full_text
        }],
        "generation_kwargs": generation_kwargs
    }


def main():
    config = PathsConfig()
    env_file = config.project_root / ".env"
    load_dotenv(env_file)
    
    prompts_file = config.data_root / "prompts" / "100_prompts_new.txt"
    examples_dir = config.raw_reuters_dir / "C50train" / "AaronPressman"
    output_file = config.data_root / "generated" / "openai_100_new.jsonl"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            f"OpenAI API key not found. Please set OPENAI_API_KEY in {env_file} "
            "or as an environment variable"
        )
    
    client = OpenAI(api_key=api_key)
    
    # Load prompts
    print(f"Loading prompts from {prompts_file}...")
    prompts = load_prompts(prompts_file)
    print(f"Loaded {len(prompts)} prompts")
    
    # Load example files
    print(f"Loading example files from {examples_dir}...")
    examples = load_example_files(examples_dir, num_examples=5)
    print(f"Loaded {len(examples)} example files")
    
    # Build system prompt with examples
    system_prompt = build_system_prompt(examples)
    
    # Generation parameters (matching the format from gpt2l_100_new.jsonl)
    generation_kwargs = {
        "max_new_tokens": 200,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "num_return_sequences": 1,
        "do_sample": True,
        "num_beams": 1,
        "no_repeat_ngram_size": 0,
        "length_penalty": 1.0,
        "early_stopping": False,
        "pad_token_id": None,
        "eos_token_id": None
    }
    
    # Generate completions for each prompt
    print(f"Generating completions...")
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Processing prompt {i}/{len(prompts)}: {prompt[:50]}...")
        system_prompt = build_system_prompt(examples)
        
        try:
            completion = generate_completion(
                client=client,
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=200,
                temperature=0.8
            )
            
            output = format_output(prompt, completion, generation_kwargs)
            results.append(output)
            
        except Exception as e:
            print(f"Failed to generate completion for prompt {i}: {e}")
            continue
    
    # Write results to JSONL file
    print(f"Writing results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Successfully generated {len(results)} completions")
    print(f"Output saved to {output_file}")


if __name__ == "__main__":
    main()

