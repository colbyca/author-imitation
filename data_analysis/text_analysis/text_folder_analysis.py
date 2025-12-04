import os
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.platypus import KeepTogether
from itertools import islice

# Ensure NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

STOP_WORDS = set(stopwords.words('english'))


def tokenize_sentences(text):
    return nltk.sent_tokenize(text)


def tokenize_words(text):
    return nltk.word_tokenize(text.lower())


def pos_tag_flat(tokenized):
    tagged = [nltk.pos_tag(tokens) for tokens in tokenized]
    return [tag for (_, tag) in sum(tagged, [])]


def analyze_folder(folder_path):
    out_dir = f"folder_analysis_output_{folder_path}"
    analysis_dir = "text_reports"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    file_word_counts = {}
    file_sentence_lengths = {}
    filtered_word_counts = Counter()
    file_filtered_word_counts = {}
    all_tags = Counter()

    global_word_counter = Counter()
    global_sentence_lengths = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            path = os.path.join(folder_path, filename)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()

            sentences = tokenize_sentences(text)
            words = tokenize_words(text)

            stop_words = [w for w in words if w.isalpha() and w in STOP_WORDS]
            non_stop_words = [w for w in words if w.isalpha() and w not in STOP_WORDS]
            file_word_counts[filename] = Counter(words)
            file_filtered_word_counts[filename] = Counter(non_stop_words)
            filtered_word_counts.update(stop_words)

            ptags = pos_tag_flat(tokenize_words(s) for s in sentences)
            tags = Counter(zip(*(islice(ptags, i, None) for i in range(2))))
            all_tags.update(tags)

            sentence_lengths = []
            for s in sentences:
                s_words = [w for w in tokenize_words(s) if w.isalpha()]
                sentence_lengths.append(len(s_words))
                global_sentence_lengths.append(len(s_words))

            file_sentence_lengths[filename] = sentence_lengths

            global_word_counter.update(words)

    # Save file-level top-10 words
    top10_path = os.path.join(out_dir, "top10_words_per_item.txt")
    with open(top10_path, 'w') as out:
        for fname, counter in file_filtered_word_counts.items():
            out.write(f"=== {fname} ===\n")
            loop_count = 0
            for word, count in counter.most_common(10):
                loop_count += 1
                out.write(f"{word}: {count}")
                if loop_count < 10:
                    out.write(", ")
            out.write("\n")

    # Sentence length IQR graphs per file
    for fname, lengths in file_sentence_lengths.items():
        q1 = np.percentile(lengths, 25)
        q3 = np.percentile(lengths, 75)

        plt.figure()
        plt.title(f"IQR Sentence Length - {fname}")
        plt.boxplot(lengths, vert=True)
        out_png = os.path.join(out_dir, f"sentence_length_IQR_{fname}.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()

    # Total different non-stop words
    uniq_path = os.path.join(out_dir, "stylistic_analysis.txt")
    with open(uniq_path, 'w') as out:
        out.write(f"Total unique words: {len(global_word_counter)}\n")
        out.write(f"Total word count: {sum(global_word_counter.values())}\n")
        out.write(f"Type Token Ratio (TTR): {len(global_word_counter)/sum(global_word_counter.values()):.2%}\n")
        out.write(f"Total stop-words count: {sum(filtered_word_counts.values())}\n")
        out.write(f"Most common Parts of Speech n-grams:\n")
        num = 0
        for gram, count in all_tags.most_common(15):
            num += 1
            out.write(f"{num}. {gram}, {count}\n")

    # Word frequency distribution graph
    freq_counts = Counter(global_word_counter.values())
    x = list(freq_counts.keys())
    y = list(freq_counts.values())

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel('Word Frequency')
    plt.ylabel('Number of Words')
    plt.title('Word Frequency Distribution')
    plt.savefig(os.path.join(out_dir, "global_word_freq_distribution.png"), bbox_inches="tight")
    plt.close()

    # Global sentence length distribution
    mean_len = np.mean(global_sentence_lengths)
    std_len = np.std(global_sentence_lengths)
    total_bins = range(min(global_sentence_lengths), max(global_sentence_lengths) + 2)
    plt.figure()
    plt.hist(global_sentence_lengths, total_bins)
    plt.axvline(mean_len, linestyle='--', label='Mean')
    plt.axvline(mean_len + std_len, linestyle='--', label='+1 SD')
    plt.axvline(mean_len - std_len, linestyle='--', label='-1 SD')
    plt.legend()
    plt.title('Global Sentence Length Distribution')
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(out_dir, "global_sentence_length_distribution.png"), bbox_inches="tight")
    plt.close()

    # Generate single PDF report
    pdf_path = os.path.join(analysis_dir, f"analysis_report_{folder_path}.pdf")
    pdf = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add text sections
    def add_text_file(path, title):
        if os.path.exists(path):
            block = []
            block.append(Paragraph(f"<b>{title}</b>", styles['Heading2']))
            block.append(Spacer(1, 12))
            with open(path, 'r') as f:
                for line in f:
                    block.append(Paragraph(line.strip(), styles['BodyText']))
            block.append(Spacer(1, 12))
            story.append(KeepTogether(block))

    add_text_file(top10_path, 'Top 10 Words Per File')
    add_text_file(uniq_path, 'Stylistic Analysis')
    # Add images
    for fname in sorted(os.listdir(out_dir)):
        if fname.endswith('.png'):
            fpath = os.path.join(out_dir, fname)
            story.append(
                KeepTogether([
                    Paragraph(fname, styles['Heading2']),
                    Image(fpath, width=250, height=180),
                    Spacer(1, 24),
                ])
            )

    pdf.build(story)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python text_folder_analysis.py <folder_path>")
        sys.exit(1)
    analyze_folder(sys.argv[1])
