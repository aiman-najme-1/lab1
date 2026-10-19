import unittest

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from python_basics import (
    count_one_bits,
    count_vowels,
    has_unique_characters,
    is_balanced_number,
    mse,
    multiplicative_persistence,
    prime_factorization,
    pyramid,
)


class CountVowelsTests(unittest.TestCase):
    def test_mixed_case_and_repeated_vowels(self):
        self.assertEqual(count_vowels(TextInput("AEiou Aa")), 7)

    def test_no_vowels(self):
        self.assertEqual(count_vowels(TextInput("rhythm")), 0)

    def test_empty_text(self):
        self.assertEqual(count_vowels(TextInput("")), 0)


class UniqueCharactersTests(unittest.TestCase):
    def test_distinct_characters(self):
        self.assertTrue(has_unique_characters(TextInput("Python!")))

    def test_repeated_character(self):
        self.assertFalse(has_unique_characters(TextInput("letter")))

    def test_case_is_significant(self):
        self.assertTrue(has_unique_characters(TextInput("Aa")))

    def test_empty_text(self):
        self.assertTrue(has_unique_characters(TextInput("")))


class CountOneBitsTests(unittest.TestCase):
    def test_examples(self):
        for number, expected in [(1, 1), (7, 3), (8, 1), (42, 3), (255, 8)]:
            with self.subTest(number=number):
                self.assertEqual(count_one_bits(PositiveIntegerInput(number)), expected)

    def test_large_integer(self):
        self.assertEqual(count_one_bits(PositiveIntegerInput((1 << 200) + 1)), 2)


class PersistenceTests(unittest.TestCase):
    def test_examples_from_readme(self):
        for number, expected in [(39, 3), (4, 0), (999, 4)]:
            with self.subTest(number=number):
                self.assertEqual(multiplicative_persistence(PositiveIntegerInput(number)), expected)

    def test_digit_zero_stops_after_one_step(self):
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(105)), 1)

    def test_smallest_positive_integer(self):
        self.assertEqual(multiplicative_persistence(PositiveIntegerInput(1)), 0)


class MseTests(unittest.TestCase):
    def test_identical_vectors(self):
        self.assertEqual(mse(VectorPairInput([1, -2, 3], [1, -2, 3])), 0.0)

    def test_nonzero_errors(self):
        self.assertAlmostEqual(mse(VectorPairInput([1, 2, 3], [2, 4, 6])), 14 / 3)

    def test_single_element(self):
        self.assertEqual(mse(VectorPairInput([-2], [3])), 25.0)

    def test_tuples_are_supported(self):
        self.assertEqual(mse(VectorPairInput((0, 2), (2, 0))), 4.0)

    def test_different_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            mse(VectorPairInput([1, 2], [1]))

    def test_empty_vectors_are_rejected(self):
        with self.assertRaises(ValueError):
            mse(VectorPairInput([], []))


class PrimeFactorizationTests(unittest.TestCase):
    def test_example_from_readme(self):
        self.assertEqual(prime_factorization(PositiveIntegerInput(86240)), "(2**5)(5)(7**2)(11)")

    def test_prime_number(self):
        self.assertEqual(prime_factorization(PositiveIntegerInput(13)), "(13)")

    def test_one_has_no_prime_factors(self):
        self.assertEqual(prime_factorization(PositiveIntegerInput(1)), "")

    def test_prime_powers(self):
        self.assertEqual(prime_factorization(PositiveIntegerInput(1024)), "(2**10)")
        self.assertEqual(prime_factorization(PositiveIntegerInput(81)), "(3**4)")

    def test_distinct_factors_are_sorted(self):
        self.assertEqual(prime_factorization(PositiveIntegerInput(2310)), "(2)(3)(5)(7)(11)")


class PyramidTests(unittest.TestCase):
    def test_small_pyramids(self):
        for cube_count, height in [(1, 1), (5, 2), (14, 3), (30, 4), (55, 5), (91, 6)]:
            with self.subTest(cube_count=cube_count):
                self.assertEqual(pyramid(PositiveIntegerInput(cube_count)), height)

    def test_impossible_counts(self):
        for cube_count in [2, 4, 6, 13, 15, 90, 92]:
            with self.subTest(cube_count=cube_count):
                self.assertEqual(pyramid(PositiveIntegerInput(cube_count)), "It is impossible")

    def test_very_large_pyramid_and_adjacent_count(self):
        height = 10**20
        cube_count = height * (height + 1) * (2 * height + 1) // 6
        self.assertEqual(pyramid(PositiveIntegerInput(cube_count)), height)
        self.assertEqual(pyramid(PositiveIntegerInput(cube_count + 1)), "It is impossible")


class BalancedNumberTests(unittest.TestCase):
    def test_examples_from_readme(self):
        self.assertTrue(is_balanced_number(PositiveIntegerInput(1234006)))
        self.assertFalse(is_balanced_number(PositiveIntegerInput(123456)))

    def test_one_and_two_digits_have_empty_sides(self):
        for number in [1, 7, 10, 12, 99]:
            with self.subTest(number=number):
                self.assertTrue(is_balanced_number(PositiveIntegerInput(number)))

    def test_single_middle_digit_is_excluded(self):
        self.assertTrue(is_balanced_number(PositiveIntegerInput(191)))
        self.assertTrue(is_balanced_number(PositiveIntegerInput(12321)))
        self.assertFalse(is_balanced_number(PositiveIntegerInput(192)))

    def test_two_middle_digits_are_excluded(self):
        self.assertTrue(is_balanced_number(PositiveIntegerInput(1231)))
        self.assertTrue(is_balanced_number(PositiveIntegerInput(120003)))
        self.assertFalse(is_balanced_number(PositiveIntegerInput(1234)))


if __name__ == "__main__":
    unittest.main()
