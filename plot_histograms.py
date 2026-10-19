from pathlib import Path
from math import ceil

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np


def _save_histograms(series: np.ndarray, label: str, output: Path) -> None:
    plot_columns = min(3, len(series))
    plot_rows = ceil(len(series) / plot_columns)
    figure = plt.figure(figsize=(4 * plot_columns, 3 * plot_rows))
    for index, values in enumerate(series):
        plt.subplot(plot_rows, plot_columns, index + 1)
        plt.hist(values, bins=6, color="#2878b5", edgecolor="white")
        plt.title(f"{label} {index}")
        plt.xlabel("Значение")
        plt.ylabel("Количество")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def plot_histograms(matrix: np.ndarray, output_directory: str | Path) -> tuple[Path, Path]:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("Матрица должна быть двумерной и непустой.")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    row_path = output_directory / "row_histograms.png"
    column_path = output_directory / "column_histograms.png"
    _save_histograms(matrix, "Строка", row_path)
    _save_histograms(matrix.T, "Столбец", column_path)
    return row_path, column_path


if __name__ == "__main__":
    from grader_contracts.numpy_tasks import RandomMatrixInput
    from numpy_tasks import matrix_statistics

    statistics = matrix_statistics(RandomMatrixInput(rows=4, columns=6, seed=42))
    for path in plot_histograms(statistics.matrix, Path(__file__).parent / "histograms"):
        print(path.name)
