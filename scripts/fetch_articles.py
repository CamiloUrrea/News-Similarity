"""Export news articles from the NLTK Reuters corpus to data/articles/.

Usage:
    python scripts/fetch_articles.py [--force]
"""

import argparse
import os
import re

import nltk
from nltk.corpus import reuters

TOPICS = ["crude", "grain", "acq"]
ARTICLES_PER_TOPIC = 3
MIN_WORDS, MAX_WORDS = 150, 500

# Keywords that show an article is really about its topic
KEYWORDS = {
    "crude": ["oil", "crude"],
    "grain": ["grain", "wheat", "corn"],
    "acq": ["acquire", "acquisition", "bid", "takeover", "stake", "offer"],
}
MIN_KEYWORDS = 4

# Skip numeric tables: max share of digit characters
MAX_DIGIT_RATIO = 0.05

# Output folder, relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTICLES_DIR = os.path.join(BASE_DIR, "data", "articles")


def ensure_reuters():
    """Download the Reuters corpus only if it is not available."""
    try:
        nltk.data.find("corpora/reuters")
    except LookupError:
        nltk.download("reuters", quiet=True)


def count_keywords(text, keywords):
    """Count whole-word, case-insensitive occurrences of the keywords in a text."""
    pattern = r"\b(?:" + "|".join(map(re.escape, keywords)) + r")\b"
    return len(re.findall(pattern, text.lower()))


def digit_ratio(text):
    """Return the share of characters in the text that are digits."""
    return sum(c.isdigit() for c in text) / len(text) if text else 0.0


def select_articles(topic):
    """Select the first matching Reuters articles for a topic.

    An article matches if its only category is the topic, it has
    between MIN_WORDS and MAX_WORDS words, it contains at least
    MIN_KEYWORDS topic keywords and at most MAX_DIGIT_RATIO of its
    characters are digits. File ids are scanned in alphabetical order
    so the result is reproducible.

    Args:
        topic: Reuters category name.

    Returns:
        A list of (fileid, text, keyword_count) tuples.

    Raises:
        ValueError: If not enough articles match.
    """
    selected = []
    for fileid in sorted(reuters.fileids(topic)):
        if reuters.categories(fileid) != [topic]:
            continue
        text = reuters.raw(fileid)
        if not MIN_WORDS <= len(text.split()) <= MAX_WORDS:
            continue
        if digit_ratio(text) > MAX_DIGIT_RATIO:
            continue
        keyword_count = count_keywords(text, KEYWORDS[topic])
        if keyword_count >= MIN_KEYWORDS:
            selected.append((fileid, text, keyword_count))
            if len(selected) == ARTICLES_PER_TOPIC:
                return selected

    raise ValueError(f"Only {len(selected)} matching articles found for topic '{topic}'")


def main():
    """Export the selected articles, refusing to overwrite unless --force."""
    parser = argparse.ArgumentParser(description="Export Reuters articles to data/articles/.")
    parser.add_argument("--force", action="store_true", help="overwrite existing .txt files")
    args = parser.parse_args()

    os.makedirs(ARTICLES_DIR, exist_ok=True)

    # Stop if there are already articles, unless forced
    existing = sorted(f for f in os.listdir(ARTICLES_DIR) if f.lower().endswith(".txt"))
    if existing and not args.force:
        print(f"{ARTICLES_DIR} already contains .txt files:")
        for name in existing:
            print(f"  {name}")
        print("No changes made. Use --force to overwrite.")
        return

    ensure_reuters()

    # Select all articles before writing anything
    rows = []
    for topic in TOPICS:
        for i, (fileid, text, keyword_count) in enumerate(select_articles(topic), start=1):
            rows.append((f"{topic}_{i:02d}.txt", fileid, text, keyword_count))

    for file_name, _, text, _ in rows:
        with open(os.path.join(ARTICLES_DIR, file_name), "w", encoding="utf-8") as f:
            f.write(text)

    # Summary table
    print(f"{'File':<14} {'Reuters fileid':<18} {'Words':>5} {'Keywords':>8}")
    for file_name, fileid, text, keyword_count in rows:
        print(f"{file_name:<14} {fileid:<18} {len(text.split()):>5} {keyword_count:>8}")
    print(f"\n{len(rows)} articles saved to {ARTICLES_DIR}")


if __name__ == "__main__":
    main()
