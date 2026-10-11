"""Pair ranking and Markdown report generation."""

import os

import numpy as np
import pandas as pd

# Written analysis inserted into the report, kept outside results/
INTERPRETATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "interpretation.md"
)

LIMITATIONS = [
    "Similarity is lexical (shared words), not semantic: synonyms and paraphrases are not detected.",
    "Preprocessing removes accented letters and the stop word \"won\".",
    "The dataset contains only a few articles.",
    "Results depend on the length of each text.",
    "Custom domain stop words were added for newswire boilerplate.",
]


def rank_pairs(matrix):
    """Rank every pair of articles by similarity.

    Args:
        matrix: Square similarity DataFrame with file names as index and columns.

    Returns:
        A DataFrame with columns article_a, article_b, similarity, one row
        per pair (upper triangle, no diagonal), sorted from most to least similar.
    """
    names = list(matrix.index)
    rows, cols = np.triu_indices(len(names), k=1)
    pairs = pd.DataFrame({
        "article_a": [names[i] for i in rows],
        "article_b": [names[j] for j in cols],
        "similarity": matrix.to_numpy()[rows, cols],
    })
    # Stable sort keeps alphabetical order among ties
    return pairs.sort_values("similarity", ascending=False, kind="stable").reset_index(drop=True)


def build_pairs_table(jaccard_matrix, cosine_matrix):
    """Combine the Jaccard and cosine rankings into one table.

    Args:
        jaccard_matrix: Square Jaccard similarity DataFrame.
        cosine_matrix: Square cosine similarity DataFrame.

    Returns:
        A DataFrame with columns article_a, article_b, jaccard, cosine,
        sorted by cosine similarity (descending).
    """
    jaccard = rank_pairs(jaccard_matrix).rename(columns={"similarity": "jaccard"})
    cosine = rank_pairs(cosine_matrix).rename(columns={"similarity": "cosine"})
    table = jaccard.merge(cosine, on=["article_a", "article_b"])
    return table.sort_values("cosine", ascending=False, kind="stable").reset_index(drop=True)


def top_and_bottom_pairs(matrix, n=3):
    """Return the n most similar and the n least similar pairs.

    Args:
        matrix: Square similarity DataFrame.
        n: Number of pairs to return for each group.

    Returns:
        A tuple (top, bottom) of DataFrames. Top is sorted from most similar,
        bottom from least similar.
    """
    ranked = rank_pairs(matrix)
    top = ranked.head(n).reset_index(drop=True)
    bottom = ranked.tail(n).iloc[::-1].reset_index(drop=True)
    return top, bottom


def _strip_ext(name):
    """Remove the .txt extension from a file name."""
    return name.removesuffix(".txt")


def _markdown_table(headers, rows):
    """Build a Markdown table, formatting floats with 3 decimals."""
    def fmt(value):
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _pairs_to_markdown(pairs):
    """Convert a pairs DataFrame (article_a, article_b, similarity) to a Markdown table."""
    rows = [
        (_strip_ext(r.article_a), _strip_ext(r.article_b), float(r.similarity))
        for r in pairs.itertuples()
    ]
    return _markdown_table(["Article A", "Article B", "Similarity"], rows)


def generate_report(jaccard_matrix, cosine_matrix, tokens_by_article, output_path):
    """Write a Markdown report summarizing the similarity results.

    Args:
        jaccard_matrix: Square Jaccard similarity DataFrame.
        cosine_matrix: Square cosine similarity DataFrame.
        tokens_by_article: Dict {file_name: list_of_tokens}.
        output_path: Path of the Markdown file to write.
    """
    lines = ["# Text Similarity in News Articles - Report", ""]

    # a) Dataset
    lines += ["## Dataset", "", f"Number of articles: {len(tokens_by_article)}", ""]
    dataset_rows = [
        (_strip_ext(name), len(tokens), len(set(tokens)))
        for name, tokens in tokens_by_article.items()
    ]
    lines += [_markdown_table(["Article", "Total tokens", "Unique tokens"], dataset_rows), ""]

    jaccard_top, jaccard_bottom = top_and_bottom_pairs(jaccard_matrix)
    cosine_top, cosine_bottom = top_and_bottom_pairs(cosine_matrix)

    # b) Most similar pairs
    lines += [
        "## Most Similar Pairs", "",
        "### Jaccard", "", _pairs_to_markdown(jaccard_top), "",
        "### Cosine", "", _pairs_to_markdown(cosine_top), "",
    ]

    # c) Least similar pairs
    lines += [
        "## Least Similar Pairs", "",
        "### Jaccard", "", _pairs_to_markdown(jaccard_bottom), "",
        "### Cosine", "", _pairs_to_markdown(cosine_bottom), "",
    ]

    # d) Metric comparison
    lines += ["## Metric Comparison", ""]
    pairs = build_pairs_table(jaccard_matrix, cosine_matrix)
    if pairs.empty:
        lines += ["Not enough articles to compare pairs.", ""]
    else:
        best_jaccard = (jaccard_top.article_a[0], jaccard_top.article_b[0])
        best_cosine = (cosine_top.article_a[0], cosine_top.article_b[0])
        jaccard_name = " - ".join(_strip_ext(n) for n in best_jaccard)
        cosine_name = " - ".join(_strip_ext(n) for n in best_cosine)
        if best_jaccard == best_cosine:
            lines.append(f"- The most similar pair is the same for both metrics: {jaccard_name}.")
        else:
            lines.append(
                f"- The most similar pair differs: {jaccard_name} (Jaccard) "
                f"vs {cosine_name} (cosine)."
            )

        # Pearson correlation is undefined with fewer than 2 pairs or constant values
        jaccard_values = pairs["jaccard"].to_numpy()
        cosine_values = pairs["cosine"].to_numpy()
        if len(pairs) < 2 or jaccard_values.std() == 0 or cosine_values.std() == 0:
            lines.append("- Pearson correlation between Jaccard and cosine: undefined.")
        else:
            corr = np.corrcoef(jaccard_values, cosine_values)[0, 1]
            lines.append(f"- Pearson correlation between Jaccard and cosine: {corr:.3f}")
        lines.append("")

    # e) Limitations
    lines += ["## Limitations", ""]
    lines += [f"- {item}" for item in LIMITATIONS]
    lines.append("")

    # f) Interpretation, read from interpretation.md if it exists
    if os.path.isfile(INTERPRETATION_PATH):
        with open(INTERPRETATION_PATH, encoding="utf-8") as f:
            interpretation = f.read().rstrip("\n")
    else:
        interpretation = "TODO: write after reviewing results"
    lines += ["## Interpretation", "", interpretation, ""]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
