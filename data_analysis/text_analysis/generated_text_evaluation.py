import json
import os
import sys
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from itertools import islice

# Ensure NLTK resources (will download if not present)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

STOP_WORDS = set(stopwords.words('english'))


def tokenize_sentences(text):
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        return [text]


def tokenize_words(text):
    try:
        return nltk.word_tokenize(text.lower())
    except Exception:
        return text.lower().split()


def pos_tag_flat(tokenized):
    tagged = [nltk.pos_tag(tokens) for tokens in tokenized]
    return [tag for (_, tag) in sum(tagged, [])]


def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed lines
                continue
    return items


def safe_filename(s: str, maxlen=60):
    # Minimal sanitizer for filenames (not used for index-based labels)
    keep = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in s)
    keep = "_".join(keep.split())
    return keep[:maxlen]


def analyze_jsonl(jsonl_path):
    items = read_jsonl(jsonl_path)
    out_dir = f"jsonl_analysis_output_{jsonl_path}"
    analysis_dir = "text_reports"
    if len(items) == 0:
        print("No valid JSON objects found in:", jsonl_path)
        return

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    item_word_counts = {}
    item_sentence_lengths = {}
    global_word_counter = Counter()
    stop_word_counter = Counter()
    global_sentence_lengths = []
    all_tags = Counter()

    # Per-item analysis (index-based)
    for idx, entry in enumerate(items):
        # Use the `prompt_text` field per your instruction
        for sample in entry.get("samples", []):
            text = sample.get("prompt_text") + sample.get("completion")
            if not text:
                # skip empty
                continue

            sentences = tokenize_sentences(text)
            words = tokenize_words(text)
            non_stop_words = [w for w in words if w.isalpha() and w not in STOP_WORDS]
            stop_words = [w for w in words if w.isalpha() and w in STOP_WORDS]
            item_word_counts[idx] = Counter(non_stop_words)

            ptags = pos_tag_flat(tokenize_words(s) for s in sentences)
            tags = Counter(zip(*(islice(ptags, i, None) for i in range(2))))
            all_tags.update(tags)

            lengths = []
            for s in sentences:
                s_words = [w for w in tokenize_words(s) if w.isalpha()]
                l = len(s_words)
                lengths.append(l)
                global_sentence_lengths.append(l)

            item_sentence_lengths[idx] = lengths
            global_word_counter.update(words)
            stop_word_counter.update(stop_words)

            # Per-item IQR boxplot (only if there is at least one sentence length)
            if len(lengths) > 0:
                plt.figure()
                plt.title(f"IQR Sentence Length - item_{idx}")
                # Create a boxplot; using a single box vertical
                plt.boxplot(lengths, vert=True)
                out_png = os.path.join(out_dir, f"sentence_IQR_item_{idx}.png")
                plt.savefig(out_png, bbox_inches="tight")
                plt.close()

    # Write top-10 words per item
    top10_path = os.path.join(out_dir, "top10_words_per_item.txt")
    with open(top10_path, "w", encoding="utf-8") as f:
        for idx, counter in sorted(item_word_counts.items()):
            f.write(f"=== item_{idx} ===\n")
            top = counter.most_common(10)
            if top:
                f.write(", ".join(f"{w}: {c}" for w, c in top))
            else:
                f.write("(no words)")
            f.write("\n\n")

    # Global unique non-stop words
    uniq_path = os.path.join(out_dir, "stylistic_analysis.txt")
    with open(uniq_path, "w", encoding="utf-8") as f:
        f.write(f"Total unique words: {len(global_word_counter)}\n")
        f.write(f"Total word count: {sum(global_word_counter.values())}\n")
        f.write(f"Type Token Ratio (TTR): {len(global_word_counter)/sum(global_word_counter.values()):.2%}\n")
        f.write(f"Total stop-words count: {sum(stop_word_counter.values())}\n")
        f.write(f"Most common Parts of Speech n-grams:\n")
        num = 0
        for gram, count in all_tags.most_common(15):
            num += 1
            f.write(f"{num}. {gram}, {count}\n")

    # Global word frequency distribution (scatter)
    if len(global_word_counter) > 0:
        freq_counts = Counter(global_word_counter.values())
        x = sorted(freq_counts.keys())
        y = [freq_counts[k] for k in x]

        plt.figure()
        plt.scatter(x, y)
        plt.xlabel("Word Frequency")
        plt.ylabel("Number of Words")
        plt.title("Global Word Frequency Distribution")
        plt.savefig(os.path.join(out_dir, "global_word_freq_distribution.png"), bbox_inches="tight")
        plt.close()

    # Global sentence length histogram with a bin for every integer between min and max inclusive
    if len(global_sentence_lengths) > 0:
        min_l = int(min(global_sentence_lengths))
        max_l = int(max(global_sentence_lengths))
        bins = list(range(min_l, max_l + 2))  # +2 so that integers fall into discrete bins

        mean_len = np.mean(global_sentence_lengths)
        std_len = np.std(global_sentence_lengths)

        plt.figure()
        plt.hist(global_sentence_lengths, bins=bins)
        plt.axvline(mean_len, linestyle="--", label="Mean")
        plt.axvline(mean_len + std_len, linestyle="--", label="+1 SD")
        plt.axvline(mean_len - std_len, linestyle="--", label="-1 SD")
        plt.legend()
        plt.title("Global Sentence Length Distribution")
        plt.xlabel("Sentence Length (words)")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(out_dir, "global_sentence_length_distribution.png"), bbox_inches="tight")
        plt.close()

    # Generate PDF report containing:
    # - top10_words_per_item text block
    # - total_unique_non_stop_words text block
    # - all PNGs (per-item and global)
    pdf_path = os.path.join(analysis_dir, f"analysis_report_{jsonl_path}.pdf")
    pdf = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    def add_text_block(path, title):
        if os.path.exists(path):
            blk = [Paragraph(f"<b>{title}</b>", styles["Heading2"]), Spacer(1, 12)]
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip()
                    if line == "":
                        blk.append(Spacer(1, 6))
                    else:
                        blk.append(Paragraph(line, styles["BodyText"]))
            blk.append(Spacer(1, 12))
            story.append(KeepTogether(blk))

    add_text_block(top10_path, "Top 10 Words Per Item (index-based labels)")
    add_text_block(uniq_path, "Stylistic Analysis")

    # Add PNG images (sorted by name so item_0, item_1, ... come first)
    for fname in sorted(os.listdir(out_dir)):
        if fname.lower().endswith(".png"):
            fpath = os.path.join(out_dir, fname)
            story.append(
                KeepTogether([
                    Paragraph(fname, styles["Heading3"]),
                    Image(fpath, width=350, height=200),
                    Spacer(1, 18)
                ])
            )

    # Build PDF (if there's content)
    if story:
        pdf.build(story)

    print(f"Analysis complete. Outputs saved in folder: {out_dir}")
    print(f"PDF report: {pdf_path}")


if __name__ == "__main__":
    # default to the uploaded file path if user doesn't pass an argument
    if len(sys.argv) != 2:
        print("Usage: python generated_text_evaluation.py <jsonl_path>")
        sys.exit(1)
    jsonl_file = sys.argv[1]
    if not os.path.exists(jsonl_file):
        print("JSONL file not found:", jsonl_file)
        sys.exit(1)
    analyze_jsonl(jsonl_file)