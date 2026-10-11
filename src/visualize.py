"""Heatmap visualization of similarity matrices."""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_similarity_heatmap(matrix, title, output_path):
    """Plot a similarity matrix as a heatmap and save it as a PNG.

    Args:
        matrix: Square DataFrame with file names as index and columns.
        title: Title of the figure.
        output_path: Path of the PNG file to write.
    """
    # Axis labels without the .txt extension
    labels = [name.removesuffix(".txt") for name in matrix.index]

    # Scale figure size with the number of articles
    size = max(6, 0.9 * len(labels))
    plt.figure(figsize=(size, size))

    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Similarity"},
    )
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
