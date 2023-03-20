import copy
import pytest
from pytest import approx
import dataclasses
import typing as tp

from compgraph import operations as ops


class _Key:
    def __init__(self, *args: str) -> None:
        self._items = args

    def __call__(self, d: tp.Mapping[str, tp.Any]) -> tuple[str, ...]:
        return tuple(str(d.get(key)) for key in self._items)


@dataclasses.dataclass
class MapCase:
    mapper: ops.Mapper
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    mapper_item: int = 0
    mapper_ground_truth_items: tuple[int, ...] = (0,)


MAP_CASES = [
    MapCase(
        mapper=ops.DummyMapper(),
        data=[
            {'test_id': 1, 'text': 'one two three'},
            {'test_id': 2, 'text': 'testing out stuff'}
        ],
        ground_truth=[
            {'test_id': 1, 'text': 'one two three'},
            {'test_id': 2, 'text': 'testing out stuff'}
        ],
        cmp_keys=('test_id', 'text')
    ),
    MapCase(
        mapper=ops.LowerCase(column='text'),
        data=[
            {'test_id': 1, 'text': 'camelCaseTest'},
            {'test_id': 2, 'text': 'UPPER_CASE_TEST'},
            {'test_id': 3, 'text': 'wEiRdTeSt'}
        ],
        ground_truth=[
            {'test_id': 1, 'text': 'camelcasetest'},
            {'test_id': 2, 'text': 'upper_case_test'},
            {'test_id': 3, 'text': 'weirdtest'}
        ],
        cmp_keys=('test_id', 'text')
    ),
    MapCase(
        mapper=ops.FilterPunctuation(column='text'),
        data=[
            {'test_id': 1, 'text': 'Hello, world!'},
            {'test_id': 2, 'text': 'Test. with. a. lot. of. dots.'},
            {'test_id': 3, 'text': r'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'}
        ],
        ground_truth=[
            {'test_id': 1, 'text': 'Hello world'},
            {'test_id': 2, 'text': 'Test with a lot of dots'},
            {'test_id': 3, 'text': ''}
        ],
        cmp_keys=('test_id', 'text')
    ),
    MapCase(
        mapper=ops.Split(column='text'),
        data=[
            {'test_id': 1, 'text': 'one two three'},
            {'test_id': 2, 'text': 'tab\tsplitting\ttest'},
            {'test_id': 3, 'text': 'more\nlines\ntest'},
            {'test_id': 4, 'text': 'tricky\u00A0test'}
        ],
        ground_truth=[
            {'test_id': 1, 'text': 'one'},
            {'test_id': 1, 'text': 'three'},
            {'test_id': 1, 'text': 'two'},

            {'test_id': 2, 'text': 'splitting'},
            {'test_id': 2, 'text': 'tab'},
            {'test_id': 2, 'text': 'test'},

            {'test_id': 3, 'text': 'lines'},
            {'test_id': 3, 'text': 'more'},
            {'test_id': 3, 'text': 'test'},

            {'test_id': 4, 'text': 'test'},
            {'test_id': 4, 'text': 'tricky'}
        ],
        cmp_keys=('test_id', 'text'),
        mapper_ground_truth_items=(0, 1, 2)
    ),
    MapCase(
        mapper=ops.Product(columns=['speed', 'time'], result_column='distance'),
        data=[
            {'test_id': 1, 'speed': 5, 'time': 10},
            {'test_id': 2, 'speed': 60, 'time': 2},
            {'test_id': 3, 'speed': 3, 'time': 15},
            {'test_id': 4, 'speed': 100, 'time': 0.5},
            {'test_id': 5, 'speed': 48, 'time': 15},
        ],
        ground_truth=[
            {'test_id': 1, 'speed': 5, 'time': 10, 'distance': 50},
            {'test_id': 2, 'speed': 60, 'time': 2, 'distance': 120},
            {'test_id': 3, 'speed': 3, 'time': 15, 'distance': 45},
            {'test_id': 4, 'speed': 100, 'time': 0.5, 'distance': 50},
            {'test_id': 5, 'speed': 48, 'time': 15, 'distance': 720},
        ],
        cmp_keys=('test_id', 'speed', 'time', 'distance')
    ),
    MapCase(
        mapper=ops.Filter(condition=lambda row: row['f'] ^ row['g']),
        data=[
            {'test_id': 1, 'f': 0, 'g': 0},
            {'test_id': 2, 'f': 0, 'g': 1},
            {'test_id': 3, 'f': 1, 'g': 0},
            {'test_id': 4, 'f': 1, 'g': 1}
        ],
        ground_truth=[
            {'test_id': 2, 'f': 0, 'g': 1},
            {'test_id': 3, 'f': 1, 'g': 0}
        ],
        cmp_keys=('test_id', 'f', 'g'),
        mapper_ground_truth_items=tuple()
    ),
    MapCase(
        mapper=ops.Project(columns=['value']),
        data=[
            {'test_id': 1, 'junk': 'x', 'value': 42},
            {'test_id': 2, 'junk': 'y', 'value': 1},
            {'test_id': 3, 'junk': 'z', 'value': 144}
        ],
        ground_truth=[
            {'value': 42},
            {'value': 1},
            {'value': 144}
        ],
        cmp_keys=('value',)
    )
]


@pytest.mark.parametrize('case', MAP_CASES)
def test_mapper(case: MapCase) -> None:
    mapper_data_row = copy.deepcopy(case.data[case.mapper_item])
    mapper_ground_truth_rows = [copy.deepcopy(case.ground_truth[i]) for i in case.mapper_ground_truth_items]

    key_func = _Key(*case.cmp_keys)

    mapper_result = case.mapper(mapper_data_row)
    assert isinstance(mapper_result, tp.Iterator)
    assert sorted(mapper_ground_truth_rows, key=key_func) == sorted(mapper_result, key=key_func)

    result = ops.Map(case.mapper)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(case.ground_truth, key=key_func) == sorted(result, key=key_func)


@dataclasses.dataclass
class ReduceCase:
    reducer: ops.Reducer
    reducer_keys: tuple[str, ...]
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    reduce_data_items: tuple[int, ...] = (0,)
    reduce_ground_truth_items: tuple[int, ...] = (0,)


REDUCE_CASES = [
    ReduceCase(
        reducer=ops.Mean(column='score'),
        reducer_keys=('match_id',),
        data=[
            {'match_id': 1, 'player_id': 1, 'score': 42},
            {'match_id': 1, 'player_id': 2, 'score': 7},
            {'match_id': 1, 'player_id': 3, 'score': 0},
            {'match_id': 1, 'player_id': 4, 'score': 39},

            {'match_id': 2, 'player_id': 5, 'score': 15},
            {'match_id': 2, 'player_id': 6, 'score': 39},
            {'match_id': 2, 'player_id': 7, 'score': 27},
            {'match_id': 2, 'player_id': 8, 'score': 7}
        ],
        ground_truth=[
            {'match_id': 1, 'score': approx(22, 1e-3)},
            {'match_id': 2, 'score': approx(22, 1e-3)}
        ],
        cmp_keys=('test_id', 'text'),
        reduce_data_items=(0, 1, 2, 3),
        reduce_ground_truth_items=(0,)
    )

]


@pytest.mark.parametrize('case', REDUCE_CASES)
def test_reducer(case: ReduceCase) -> None:
    reducer_data_rows = [copy.deepcopy(case.data[i]) for i in case.reduce_data_items]
    reducer_ground_truth_rows = [copy.deepcopy(case.ground_truth[i]) for i in case.reduce_ground_truth_items]

    key_func = _Key(*case.cmp_keys)

    reducer_result = case.reducer(case.reducer_keys, iter(reducer_data_rows))
    assert isinstance(reducer_result, tp.Iterator)
    assert sorted(reducer_ground_truth_rows, key=key_func) == sorted(reducer_result, key=key_func)

    result = ops.Reduce(case.reducer, case.reducer_keys)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(case.ground_truth, key=key_func) == sorted(result, key=key_func)
