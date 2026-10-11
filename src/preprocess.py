"""Text preprocessing for news articles."""

import html
import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords

# Newswire boilerplate words, added to NLTK's English stop words
DOMAIN_STOPWORDS = {
    "said", "says", "also", "would", "could", "year", "years", "reuters",
    "mln", "pct", "dlrs", "billion", "today", "told", "new",
}


@lru_cache(maxsize=1)
def _english_stopwords():
    """Return NLTK's English stop words plus the domain ones (loaded once)."""
    return set(stopwords.words("english")) | DOMAIN_STOPWORDS


def preprocess_text(text):
    """Clean a text and split it into tokens.

    Steps: unescape HTML entities, remove text between angle brackets,
    lowercase, remove URLs, keep only letters and spaces, tokenize,
    remove English and domain stop words and drop tokens shorter
    than 3 characters.

    Args:
        text: Raw text string.

    Returns:
        A list of clean tokens.
    """
    text = html.unescape(text)

    # Remove text between angle brackets (e.g. tickers like <BUD.N>)
    text = re.sub(r"<[^>]*>", " ", text)

    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Keep only letters and whitespace
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = nltk.word_tokenize(text)

    # Remove stop words and short tokens
    stop_words = _english_stopwords()
    return [t for t in tokens if t not in stop_words and len(t) >= 3]


def preprocess_articles(articles):
    """Preprocess every article in a dictionary.

    Args:
        articles: Dict {file_name: text}, as returned by load_articles.

    Returns:
        A dict {file_name: list_of_tokens}.
    """
    return {name: preprocess_text(text) for name, text in articles.items()}
