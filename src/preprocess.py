"""Text preprocessing for news articles."""

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords


@lru_cache(maxsize=1)
def _english_stopwords():
    """Return the English stop words as a set (loaded once)."""
    return set(stopwords.words("english"))


def preprocess_text(text):
    """Clean a text and split it into tokens.

    Steps: lowercase, remove URLs, keep only letters and spaces,
    tokenize, remove English stop words and drop tokens shorter
    than 3 characters.

    Args:
        text: Raw text string.

    Returns:
        A list of clean tokens.
    """
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
