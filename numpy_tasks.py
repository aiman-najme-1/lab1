import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = np.asarray(data.matrices), np.asarray(data.vectors)
    if (matrices.ndim != 3 or vectors.ndim != 3
            or matrices.shape[1] != matrices.shape[2]
            or vectors.shape != (matrices.shape[0], matrices.shape[1], 1)):
        raise ValueError("Ожидаются формы (p, n, n) и (p, n, 1).")
    return (matrices @ vectors).sum(axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = np.asarray(data.matrix), data.threshold
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    if matrix.ndim != 2:
        raise ValueError("Ожидается двумерная матрица.")
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    if matrix.ndim != 2:
        raise ValueError("Ожидается двумерная матрица.")
    return [np.unique(column).tolist() for column in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    if rows <= 0 or columns <= 0:
        raise ValueError("Размеры матрицы должны быть положительными.")
    if std < 0:
        raise ValueError("Стандартное отклонение должно быть неотрицательным.")
    generator = np.random.default_rng(seed)
    matrix = generator.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    if rows < 0 or columns < 0:
        raise ValueError("Размеры матрицы должны быть неотрицательными.")
    row_indices, column_indices = np.indices((rows, columns))
    return np.where((row_indices + column_indices) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    if image_height <= 0 or image_width <= 0 or width < 0 or height < 0:
        raise ValueError("Размеры изображения должны быть положительными, а размеры прямоугольника — неотрицательными.")
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    left, top = (image_width - width) // 2, (image_height - height) // 2
    right, bottom = left + width, top + height
    image[max(0, top):min(image_height, bottom),
          max(0, left):min(image_width, right)] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    if min(image_height, image_width, semi_axis_x, semi_axis_y) <= 0:
        raise ValueError("Размеры изображения и полуоси эллипса должны быть положительными.")
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    y, x = np.ogrid[:image_height, :image_width]
    center_x, center_y = (image_width - 1) / 2, (image_height - 1) / 2
    inside = ((x - center_x) / semi_axis_x) ** 2 + ((y - center_y) / semi_axis_y) ** 2 <= 1
    image[inside] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = np.asarray(data.values, dtype=float), data.window
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Ожидается непустой одномерный временной ряд.")
    if not isinstance(window, (int, np.integer)) or not 1 <= window <= values.size:
        raise ValueError("Размер окна должен быть целым числом от 1 до длины ряда.")
    middle = values[1:-1]
    maxima = np.flatnonzero((middle > values[:-2]) & (middle > values[2:])) + 1
    minima = np.flatnonzero((middle < values[:-2]) & (middle < values[2:])) + 1
    moving_average = np.convolve(values, np.ones(window) / window, mode="valid")
    return TimeSeriesStatistics(
        mean=float(values.mean()),
        variance=float(values.var()),
        std=float(values.std()),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = np.asarray(data.labels), data.class_count
    if labels.ndim != 1:
        raise ValueError("Ожидается одномерный вектор меток.")
    if labels.size and (not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)):
        raise ValueError("Метки должны быть неотрицательными целыми числами.")
    required_classes = int(labels.max()) + 1 if labels.size else 0
    if class_count is None:
        class_count = required_classes
    if (not isinstance(class_count, (int, np.integer))
            or class_count < required_classes or class_count < 0):
        raise ValueError("Количество классов должно быть неотрицательным целым числом, включающим все метки.")
    result = np.zeros((labels.size, class_count), dtype=int)
    if labels.size:
        result[np.arange(labels.size), labels] = 1
    return result
