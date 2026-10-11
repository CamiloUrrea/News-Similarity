"""Utilities for loading news articles from disk."""

import os


def load_articles(folder_path):
    """Load all .txt articles from a folder.

    Args:
        folder_path: Path to the folder containing the .txt files.

    Returns:
        A dict mapping each file name to its text content,
        in alphabetical order by file name.

    Raises:
        FileNotFoundError: If the folder does not exist.
        ValueError: If the folder contains no .txt files.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Collect .txt files in alphabetical order
    file_names = sorted(
        name for name in os.listdir(folder_path)
        if name.lower().endswith(".txt")
    )

    if not file_names:
        raise ValueError(f"No .txt files found in: {folder_path}")

    articles = {}
    for name in file_names:
        with open(os.path.join(folder_path, name), encoding="utf-8") as f:
            articles[name] = f.read()

    return articles
