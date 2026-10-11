"""Feature extraction and similarity measures for news articles."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_term_vectors(tokens_by_article):
    """Build a term-frequency matrix from pre-tokenized articles.

    Args:
        tokens_by_article: Dict {file_name: list_of_tokens}.

    Returns:
        A tuple (matrix, names): the sparse term-frequency matrix
        (one row per article) and the file names in the same order.
    """
    names = list(tokens_by_article.keys())

    # Texts are already tokenized, so the analyzer returns them as-is
    vectorizer = CountVectorizer(analyzer=lambda tokens: tokens)
    matrix = vectorizer.fit_transform(tokens_by_article[name] for name in names)

    return matrix, names


def jaccard_similarity(tokens_a, tokens_b):
    """Compute the Jaccard similarity between two token lists.

    Args:
        tokens_a: First list of tokens.
        tokens_b: Second list of tokens.

    Returns:
        |A ∩ B| / |A ∪ B| as a float, or 0.0 if the union is empty.
    """
    set_a, set_b = set(tokens_a), set(tokens_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def compute_jaccard_matrix(tokens_by_article):
    """Compute the pairwise Jaccard similarity between all articles.

    Args:
        tokens_by_article: Dict {file_name: list_of_tokens}.

    Returns:
        A square DataFrame with file names as index and columns.
    """
    names = list(tokens_by_article.keys())
    data = [
        [jaccard_similarity(tokens_by_article[a], tokens_by_article[b]) for b in names]
        for a in names
    ]
    return pd.DataFrame(data, index=names, columns=names)


def compute_cosine_matrix(tokens_by_article):
    """Compute the pairwise cosine similarity between all articles.

    Args:
        tokens_by_article: Dict {file_name: list_of_tokens}.

    Returns:
        A square DataFrame with file names as index and columns.
    """
    matrix, names = build_term_vectors(tokens_by_article)
    return pd.DataFrame(cosine_similarity(matrix), index=names, columns=names)
