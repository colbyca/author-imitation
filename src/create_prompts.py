"""Extract first n words from text files in Reuters C50test directories.

This script takes a person's name, extracts the first n words from every text file
in that person's directory and the next k people's directories, and writes each
extract as a new line in an output file.
"""

import argparse
from pathlib import Path
from typing import List


def get_first_n_words(text: str, n: int) -> str:
    """Extract the first n words from a text string."""
    words = text.split()
    return " ".join(words[:n])


def get_sorted_directories(base_dir: Path) -> List[Path]:
    """Get all author directories sorted alphabetically."""
    directories = [d for d in base_dir.iterdir() if d.is_dir()]
    return sorted(directories)


def find_person_directories(base_dir: Path, person_name: str, k: int) -> List[Path]:
    """Find the person's directory and the next k people's directories."""
    sorted_dirs = get_sorted_directories(base_dir)
    
    # Find the starting person's directory
    person_dir = None
    start_idx = None
    for idx, dir_path in enumerate(sorted_dirs):
        if dir_path.name == person_name:
            person_dir = dir_path
            start_idx = idx
            break
    
    if person_dir is None:
        raise ValueError(f"Person '{person_name}' not found in {base_dir}")
    
    # Get the next k directories (including the starting person)
    end_idx = min(start_idx + k, len(sorted_dirs))
    return sorted_dirs[start_idx:end_idx]


def extract_words_from_directory(dir_path: Path, n: int, max_texts: int | None = None) -> List[str]:
    """Extract first n words from text files in a directory.
    
    Args:
        dir_path: Directory containing text files
        n: Number of words to extract from each file
        max_texts: Maximum number of text files to process (None = all files)
    """
    extracts = []
    text_files = sorted(dir_path.glob("*.txt"))
    
    # Limit to first m texts if specified
    if max_texts is not None:
        text_files = text_files[:max_texts]
    
    for text_file in text_files:
        try:
            with open(text_file, "r", encoding="utf-8") as f:
                content = f.read()
                first_n_words = get_first_n_words(content, n)
                if first_n_words:  # Only add non-empty extracts
                    extracts.append(first_n_words)
        except Exception as e:
            print(f"Warning: Could not read {text_file}: {e}")
    
    return extracts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract first n words from text files in Reuters C50test directories"
    )
    parser.add_argument(
        "person_name",
        type=str,
        help="Name of the person (directory name in C50test, e.g., 'AaronPressman')"
    )
    parser.add_argument(
        "-n",
        "--num-words",
        type=int,
        required=True,
        help="Number of words to extract from each file"
    )
    parser.add_argument(
        "-k",
        "--num-people",
        type=int,
        required=True,
        help="Number of people's directories to process (including the starting person)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/prompts/prompts.txt",
        help="Output file path (default: data/prompts/prompts.txt)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/reuter/C50test",
        help="Path to C50test directory (default: data/reuter/C50test)"
    )
    parser.add_argument(
        "-m",
        "--max-texts",
        type=int,
        default=None,
        help="Maximum number of text files to process from each person's directory (default: all files)"
    )
    
    args = parser.parse_args()
    
    if args.data_dir:
        base_dir = Path(args.data_dir)
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        base_dir = project_root / "data" / "reuter" / "C50test"
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    
    # Find directories to process
    print(f"Looking for person '{args.person_name}' in {base_dir}...")
    directories = find_person_directories(base_dir, args.person_name, args.num_people)
    
    print(f"Found {len(directories)} directory(ies) to process:")
    for dir_path in directories:
        print(f"  - {dir_path.name}")
    
    # Extract words from all files in these directories
    all_extracts = []
    for dir_path in directories:
        print(f"Processing {dir_path.name}...")
        extracts = extract_words_from_directory(dir_path, args.num_words, args.max_texts)
        all_extracts.extend(extracts)
        print(f"  Extracted {len(extracts)} lines from {dir_path.name}")
    
    # Write to output file
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        for extract in all_extracts:
            f.write(extract + "\n")
    
    print(f"\nDone! Wrote {len(all_extracts)} lines to {output_path}")


if __name__ == "__main__":
    main()

