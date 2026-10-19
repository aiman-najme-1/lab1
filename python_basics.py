from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    return sum(character in "aeiou" for character in text.lower())


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("Векторы должны иметь одинаковую длину.")
    if len(predicted) == 0:
        raise ValueError("Векторы не должны быть пустыми.")
    squared_error = sum((x - y) ** 2 for x, y in zip(predicted, expected))
    return squared_error / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = []
    divisor = 2
    while divisor * divisor <= number:
        exponent = 0
        while number % divisor == 0:
            number //= divisor
            exponent += 1
        if exponent == 1:
            factors.append(f"({divisor})")
        elif exponent > 1:
            factors.append(f"({divisor}**{exponent})")
        
        divisor = 3 if divisor == 2 else divisor + 2
    if number > 1:
        factors.append(f"({number})")
    return "".join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value

    def square_sum(k: int) -> int:
        return k * (k + 1) * (2 * k + 1) // 6

    
    lower, upper = 1, 1
    while square_sum(upper) < cube_count:
        upper *= 2
    while lower <= upper:
        middle = (lower + upper) // 2
        total = square_sum(middle)
        if total == cube_count:
            return middle
        if total < cube_count:
            lower = middle + 1
        else:
            upper = middle - 1
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = str(number)
    side_length = (len(digits) - 1) // 2
    left = digits[:side_length]
    right = digits[len(digits) - side_length:]
    return sum(int(digit) for digit in left) == sum(int(digit) for digit in right)
