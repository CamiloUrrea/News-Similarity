"""Entry point: load, preprocess and compare the news articles."""

import os

import nltk
import pandas as pd

from src.load_data import load_articles
from src.preprocess import preprocess_articles
from src.report import build_pairs_table, generate_report
from src.similarity import compute_cosine_matrix, compute_jaccard_matrix
from src.visualize import plot_similarity_heatmap

# Download required NLTK resources only if missing
NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
}

for name, path in NLTK_RESOURCES.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name, quiet=True)

# Input and output folders, relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTICLES_DIR = os.path.join(BASE_DIR, "data", "articles")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def main():
    """Load, preprocess and compare the articles, printing each step."""
    articles = load_articles(ARTICLES_DIR)

    print(f"Loaded {len(articles)} articles.\n")
    for name, text in articles.items():
        print(f"{name}: {text[:100]}")

    # Preprocess and compare token counts
    tokens_by_article = preprocess_articles(articles)

    print("\nPreprocessing results:\n")
    for name, tokens in tokens_by_article.items():
        raw_count = len(articles[name].split())
        print(f"{name}: {raw_count} tokens before, {len(tokens)} after")
        print(f"  First 15: {tokens[:15]}")

    # Pairwise similarity matrices
    jaccard_df = compute_jaccard_matrix(tokens_by_article)
    cosine_df = compute_cosine_matrix(tokens_by_article)

    # Show all columns without wrapping
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print("\nJaccard similarity matrix:\n")
    print(jaccard_df.round(3))
    print("\nCosine similarity matrix:\n")
    print(cosine_df.round(3))

    # Save results as CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    jaccard_df.to_csv(os.path.join(RESULTS_DIR, "jaccard_matrix.csv"))
    cosine_df.to_csv(os.path.join(RESULTS_DIR, "cosine_matrix.csv"))
    print(f"\nMatrices saved to {RESULTS_DIR}")

    # Heatmaps
    heatmaps = [
        (jaccard_df, "Jaccard Similarity Between News Articles", "jaccard_heatmap.png"),
        (cosine_df, "Cosine Similarity Between News Articles", "cosine_heatmap.png"),
    ]
    for matrix, title, file_name in heatmaps:
        output_path = os.path.join(RESULTS_DIR, file_name)
        plot_similarity_heatmap(matrix, title, output_path)
        print(f"Heatmap saved to {output_path}")

    # Pairs ranking and report
    ranking_path = os.path.join(RESULTS_DIR, "pairs_ranking.csv")
    build_pairs_table(jaccard_df, cosine_df).to_csv(ranking_path, index=False)
    print(f"Pairs ranking saved to {ranking_path}")

    report_path = os.path.join(RESULTS_DIR, "report.md")
    generate_report(jaccard_df, cosine_df, tokens_by_article, report_path)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
