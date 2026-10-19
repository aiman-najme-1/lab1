import unittest

import numpy as np

import numpy_tasks as tasks
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


class SumProdTests(unittest.TestCase):
    def test_two_matrix_vector_products(self):
        matrices = np.array([[[1, 2], [3, 4]], [[2, 0], [0, 3]]])
        vectors = np.array([[[5], [6]], [[7], [8]]])
        result = tasks.sum_prod(MatrixVectorBatchInput(matrices, vectors))
        np.testing.assert_array_equal(result, [[31], [63]])
        np.testing.assert_array_equal(matrices, [[[1, 2], [3, 4]], [[2, 0], [0, 3]]])
        np.testing.assert_array_equal(vectors, [[[5], [6]], [[7], [8]]])

    def test_single_pair_with_fractional_values(self):
        result = tasks.sum_prod(MatrixVectorBatchInput([[[0.5]]], [[[3.0]]]))
        np.testing.assert_array_equal(result, [[1.5]])

    def test_empty_batch_has_zero_sum(self):
        data = MatrixVectorBatchInput(np.empty((0, 2, 2)), np.empty((0, 2, 1)))
        np.testing.assert_array_equal(tasks.sum_prod(data), [[0], [0]])

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            tasks.sum_prod(MatrixVectorBatchInput(np.ones((2, 2, 2)), np.ones((1, 2, 1))))


class BinarizeTests(unittest.TestCase):
    def test_strict_threshold_and_input_unchanged(self):
        matrix = np.array([[-1.0, 0.5, 0.51], [2.0, 0.0, 0.5]])
        original = matrix.copy()
        result = tasks.binarize(BinarizeInput(matrix))
        np.testing.assert_array_equal(result, [[0, 0, 1], [1, 0, 0]])
        np.testing.assert_array_equal(matrix, original)
        self.assertFalse(np.shares_memory(matrix, result))

    def test_custom_negative_threshold(self):
        result = tasks.binarize(BinarizeInput([[-3, -2, -1]], -2))
        np.testing.assert_array_equal(result, [[0, 0, 1]])

    def test_empty_matrix_keeps_shape(self):
        self.assertEqual(tasks.binarize(BinarizeInput(np.empty((0, 3)))).shape, (0, 3))


class UniqueValuesTests(unittest.TestCase):
    def test_unique_rows_and_columns_sorted(self):
        matrix = np.array([[3, 1, 3], [2, 1, 2], [3, 4, 2]])
        original = matrix.copy()
        self.assertEqual(tasks.unique_rows(MatrixInput(matrix)), [[1, 3], [1, 2], [2, 3, 4]])
        self.assertEqual(tasks.unique_columns(MatrixInput(matrix)), [[2, 3], [1, 4], [2, 3]])
        np.testing.assert_array_equal(matrix, original)

    def test_fractional_values_and_single_cell(self):
        self.assertEqual(tasks.unique_rows(MatrixInput([[0.5, -1.0, 0.5]])), [[-1.0, 0.5]])
        self.assertEqual(tasks.unique_columns(MatrixInput([[7]])), [[7]])

    def test_empty_rows_and_columns(self):
        self.assertEqual(tasks.unique_rows(MatrixInput(np.empty((0, 3)))), [])
        self.assertEqual(tasks.unique_columns(MatrixInput(np.empty((0, 3)))), [[], [], []])
        self.assertEqual(tasks.unique_rows(MatrixInput(np.empty((2, 0)))), [[], []])
        self.assertEqual(tasks.unique_columns(MatrixInput(np.empty((2, 0)))), [])

    def test_flat_input_is_rejected(self):
        for function in (tasks.unique_rows, tasks.unique_columns):
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function(MatrixInput([1, 2, 3]))


class RandomStatisticsTests(unittest.TestCase):
    def test_seeded_sample_and_axis_statistics(self):
        result = tasks.matrix_statistics(RandomMatrixInput(2, 3, seed=42))
        self.assertIsInstance(result, MatrixStatistics)
        
        expected = [[0.30471707975443135, -1.0399841062404955, 0.7504511958064572],
                    [0.9405647163912139, -1.9510351886538364, -1.302179506862318]]
        np.testing.assert_allclose(result.matrix, expected)
        np.testing.assert_allclose(result.row_means, [0.005061389773464375, -0.7708833263749802])
        np.testing.assert_allclose(result.column_means,
                                   [0.6226408980728226, -1.4955096474471659, -0.2758641555279304])
        np.testing.assert_allclose(result.row_variances, [0.5791731947383117, 1.5346961508429742])
        np.testing.assert_allclose(result.column_variances,
                                   [0.10107555425414545, 0.20750351869163003, 1.0533232003846276])

    def test_same_seed_reproduces_and_different_seed_changes(self):
        first = tasks.matrix_statistics(RandomMatrixInput(3, 2, mean=5, std=2, seed=9))
        second = tasks.matrix_statistics(RandomMatrixInput(3, 2, mean=5, std=2, seed=9))
        different = tasks.matrix_statistics(RandomMatrixInput(3, 2, mean=5, std=2, seed=10))
        np.testing.assert_array_equal(first.matrix, second.matrix)
        self.assertFalse(np.array_equal(first.matrix, different.matrix))

    def test_custom_mean_and_scale(self):
        result = tasks.matrix_statistics(RandomMatrixInput(1, 1, mean=10, std=2, seed=42))
        np.testing.assert_allclose(result.matrix, [[10.609434159508863]])
        np.testing.assert_array_equal(result.row_variances, [0])
        np.testing.assert_array_equal(result.column_variances, [0])

    def test_zero_standard_deviation(self):
        result = tasks.matrix_statistics(RandomMatrixInput(2, 3, mean=3.25, std=0))
        np.testing.assert_array_equal(result.matrix, [[3.25, 3.25, 3.25], [3.25, 3.25, 3.25]])
        np.testing.assert_array_equal(result.row_means, [3.25, 3.25])
        np.testing.assert_array_equal(result.column_means, [3.25, 3.25, 3.25])
        np.testing.assert_array_equal(result.row_variances, [0, 0])
        np.testing.assert_array_equal(result.column_variances, [0, 0, 0])

    def test_invalid_dimensions_and_standard_deviation(self):
        cases = (RandomMatrixInput(0, 2), RandomMatrixInput(2, -1), RandomMatrixInput(1, 1, std=-1))
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                tasks.matrix_statistics(data)


class ChessTests(unittest.TestCase):
    def test_readme_example(self):
        np.testing.assert_array_equal(tasks.chess(ChessInput(3, 4, 0, 1)),
                                      [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])

    def test_single_cell_and_fractional_colors(self):
        np.testing.assert_array_equal(tasks.chess(ChessInput(1, 1, 2.5, -1.5)), [[2.5]])
        np.testing.assert_array_equal(tasks.chess(ChessInput(2, 1, 2.5, -1.5)), [[2.5], [-1.5]])

    def test_zero_dimension(self):
        self.assertEqual(tasks.chess(ChessInput(0, 4, 1, 2)).shape, (0, 4))
        self.assertEqual(tasks.chess(ChessInput(3, 0, 1, 2)).shape, (3, 0))

    def test_negative_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            tasks.chess(ChessInput(-1, 2, 0, 1))


class RectangleTests(unittest.TestCase):
    def test_centered_rectangle(self):
        result = tasks.draw_rectangle(RectangleInput(3, 1, 3, 5, (255, 0, 0), (0, 10, 20)))
        self.assertEqual(result.shape, (3, 5, 3))
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result[1],
                                      [[0, 10, 20], [255, 0, 0], [255, 0, 0], [255, 0, 0], [0, 10, 20]])
        np.testing.assert_array_equal(result[0], [[0, 10, 20]] * 5)
        np.testing.assert_array_equal(result[2], [[0, 10, 20]] * 5)

    def test_floor_offsets_for_mixed_parity(self):
        result = tasks.draw_rectangle(RectangleInput(1, 1, 4, 4, (255, 255, 255), (0, 0, 0)))
        expected = np.zeros((4, 4, 3), dtype=np.uint8)
        expected[1, 1] = (255, 255, 255)
        np.testing.assert_array_equal(result, expected)

    def test_oversized_rectangle_is_clipped(self):
        result = tasks.draw_rectangle(RectangleInput(7, 5, 2, 3, (1, 2, 3), (0, 0, 0)))
        np.testing.assert_array_equal(result, [[(1, 2, 3)] * 3] * 2)

    def test_zero_width_leaves_background(self):
        result = tasks.draw_rectangle(RectangleInput(0, 2, 2, 3, (255, 0, 0), (0, 0, 0)))
        np.testing.assert_array_equal(result, np.zeros((2, 3, 3)))

    def test_invalid_dimensions(self):
        for width, height, image_height, image_width in ((-1, 2, 3, 3), (1, 1, 0, 3)):
            with self.subTest(width=width, image_height=image_height), self.assertRaises(ValueError):
                tasks.draw_rectangle(RectangleInput(width, height, image_height, image_width,
                                                    (1, 1, 1), (0, 0, 0)))


class EllipseTests(unittest.TestCase):
    def test_unit_circle_includes_boundary_pixels(self):
        result = tasks.draw_ellipse(EllipseInput(1, 1, 3, 3, (255, 0, 0), (0, 0, 0)))
        self.assertEqual(result.shape, (3, 3, 3))
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result[:, :, 0], [[0, 255, 0], [255, 255, 255], [0, 255, 0]])
        np.testing.assert_array_equal(result[:, :, 1:], np.zeros((3, 3, 2)))

    def test_even_image_uses_geometric_pixel_center(self):
        result = tasks.draw_ellipse(EllipseInput(1, 1, 4, 4, (1, 2, 3), (0, 0, 0)))
        np.testing.assert_array_equal(result[:, :, 0],
                                      [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])

    def test_unequal_axes_and_background_color(self):
        result = tasks.draw_ellipse(EllipseInput(2, 1, 3, 5, (50, 60, 70), (10, 20, 30)))
        np.testing.assert_array_equal(result[:, :, 0],
                                      [[10, 10, 50, 10, 10], [50, 50, 50, 50, 50], [10, 10, 50, 10, 10]])
        np.testing.assert_array_equal(result[0, 0], [10, 20, 30])
        np.testing.assert_array_equal(result[1, 2], [50, 60, 70])

    def test_single_pixel_image(self):
        result = tasks.draw_ellipse(EllipseInput(1, 1, 1, 1, (2, 3, 4), (0, 0, 0)))
        np.testing.assert_array_equal(result, [[[2, 3, 4]]])

    def test_invalid_dimensions_or_axes(self):
        for axis_x, axis_y, height, width in ((0, 1, 3, 3), (1, -1, 3, 3), (1, 1, 0, 3)):
            with self.subTest(axis_x=axis_x, axis_y=axis_y, height=height), self.assertRaises(ValueError):
                tasks.draw_ellipse(EllipseInput(axis_x, axis_y, height, width, (1, 1, 1), (0, 0, 0)))


class TimeSeriesTests(unittest.TestCase):
    def test_statistics_extrema_and_moving_average(self):
        values = np.array([1, 3, 2, 4, 0])
        result = tasks.analyze_time_series(TimeSeriesInput(values, 2))
        self.assertIsInstance(result, TimeSeriesStatistics)
        self.assertEqual(result.mean, 2.0)
        self.assertEqual(result.variance, 2.0)
        self.assertAlmostEqual(result.std, 2 ** 0.5)
        np.testing.assert_array_equal(result.local_maxima_indices, [1, 3])
        np.testing.assert_array_equal(result.local_minima_indices, [2])
        np.testing.assert_allclose(result.moving_average, [2, 2.5, 3, 2])
        np.testing.assert_array_equal(values, [1, 3, 2, 4, 0])

    def test_plateaus_and_endpoints_are_not_extrema(self):
        result = tasks.analyze_time_series(TimeSeriesInput([4, 2, 2, 4, 4, 1], 3))
        self.assertEqual(result.local_maxima_indices.size, 0)
        self.assertEqual(result.local_minima_indices.size, 0)
        np.testing.assert_allclose(result.moving_average, [8 / 3, 8 / 3, 10 / 3, 3])

    def test_single_value(self):
        result = tasks.analyze_time_series(TimeSeriesInput([7], 1))
        self.assertEqual((result.mean, result.variance, result.std), (7.0, 0.0, 0.0))
        self.assertEqual(result.local_maxima_indices.size, 0)
        self.assertEqual(result.local_minima_indices.size, 0)
        np.testing.assert_array_equal(result.moving_average, [7])

    def test_window_one_and_full_length(self):
        first = tasks.analyze_time_series(TimeSeriesInput([1, -1, 3], 1))
        full = tasks.analyze_time_series(TimeSeriesInput([1, -1, 3], 3))
        np.testing.assert_array_equal(first.moving_average, [1, -1, 3])
        np.testing.assert_allclose(full.moving_average, [1])

    def test_two_values_have_no_extrema(self):
        result = tasks.analyze_time_series(TimeSeriesInput([1, 2], 2))
        self.assertEqual(result.local_maxima_indices.size, 0)
        self.assertEqual(result.local_minima_indices.size, 0)
        np.testing.assert_allclose(result.moving_average, [1.5])

    def test_invalid_series_or_window(self):
        cases = (TimeSeriesInput([], 1), TimeSeriesInput([[1, 2]], 1),
                 TimeSeriesInput([1, 2], 0), TimeSeriesInput([1, 2], 3), TimeSeriesInput([1, 2], 1.5))
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                tasks.analyze_time_series(data)


class OneHotTests(unittest.TestCase):
    def test_readme_example(self):
        labels = np.array([0, 2, 3, 0])
        np.testing.assert_array_equal(tasks.one_hot(OneHotInput(labels)),
                                      [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
        np.testing.assert_array_equal(labels, [0, 2, 3, 0])

    def test_explicit_class_count_includes_unused_classes(self):
        result = tasks.one_hot(OneHotInput([1, 0], 4))
        np.testing.assert_array_equal(result, [[0, 1, 0, 0], [1, 0, 0, 0]])

    def test_zero_class_label(self):
        np.testing.assert_array_equal(tasks.one_hot(OneHotInput([0, 0])), [[1], [1]])

    def test_empty_labels(self):
        self.assertEqual(tasks.one_hot(OneHotInput([])).shape, (0, 0))
        self.assertEqual(tasks.one_hot(OneHotInput([], 4)).shape, (0, 4))
        self.assertEqual(tasks.one_hot(OneHotInput([], 0)).shape, (0, 0))

    def test_invalid_labels_or_class_count(self):
        cases = (OneHotInput([-1, 0]), OneHotInput([0.5, 1]), OneHotInput([[0, 1]]),
                 OneHotInput([2], 2), OneHotInput([], -1), OneHotInput([0], 1.5))
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                tasks.one_hot(data)


if __name__ == "__main__":
    unittest.main()
