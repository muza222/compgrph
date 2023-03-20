import example


def test_example() -> None:
    assert example.sample_product() == \
           [{'a': 1, 'b': 2, 'a * b': 2},
            {'a': 2, 'b': 3, 'a * b': 6},
            {'a': 1, 'b': 3, 'a * b': 3},
            {'a': 4, 'b': 1, 'a * b': 4}]
