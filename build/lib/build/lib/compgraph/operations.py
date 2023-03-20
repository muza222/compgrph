import datetime
from abc import abstractmethod, ABC
import typing as tp
import itertools
import heapq
import math
import re

TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]

EARTH_RADIUS_KM: int = 6373
SEC_IN_HOUR: int = 3600


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


class Read(Operation):
    def __init__(self, filename: str, parser: tp.Callable[[str], TRow]) -> None:
        self.filename = filename
        self.parser = parser

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        with open(self.filename) as f:
            for line in f:
                yield self.parser(line)


class ReadIterFactory(Operation):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self.name]():
            yield row


# Operations


class Mapper(ABC):
    """Base class for mappers"""

    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in rows:
            yield from self.mapper.__call__(row)


class Reducer(ABC):
    """Base class for reducers"""

    full = False

    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        if self.reducer.full:
            yield from self.reducer.__call__(tuple(self.keys), rows)
        else:
            for k, group in itertools.groupby(rows, key=lambda x: [x[key] for key in self.keys]):
                yield from self.reducer.__call__(tuple(self.keys), group)


class Joiner(ABC):
    """Base class for joiners"""

    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


def get_key(x: tp.Any, k: tp.Any) -> tp.Any:
    res = []
    for i in k:
        res.append(x[i])
    return tuple(res)


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        left_data = itertools.groupby(rows, key=lambda x: get_key(x, self.keys))
        right_data = itertools.groupby(args[0], key=lambda x: get_key(x, self.keys))
        key1, group1 = next(left_data)
        key2, group2 = next(right_data)
        a_stopped = False
        b_stopped = False
        empty: TRow = dict()
        while not a_stopped or not b_stopped:
            if a_stopped:
                yield from self.joiner.__call__(self.keys, [empty], group2)
                for k2, g2 in right_data:
                    yield from self.joiner.__call__(self.keys, [empty], g2)
                break
            elif b_stopped:
                yield from self.joiner.__call__(self.keys, group1, [empty])
                for k1, g1 in left_data:
                    yield from self.joiner.__call__(self.keys, g1, [empty])
                break

            if key1 == key2:
                yield from self.joiner.__call__(self.keys, group1, group2)
                try:
                    key1, group1 = next(left_data)
                except StopIteration:
                    a_stopped = True
                try:
                    key2, group2 = next(right_data)
                except StopIteration:
                    b_stopped = True
                    continue
                continue

            if key2 < key1:
                while key2 < key1:
                    for i in self.joiner.__call__(self.keys, [empty], group2):
                        yield i
                    try:
                        key2, group2 = next(right_data)
                    except StopIteration:
                        b_stopped = True
                        continue
                continue

            if key1 < key2:
                while key2 > key1:
                    for i in self.joiner.__call__(self.keys, group1, [empty]):
                        yield i
                    try:
                        key1, group1 = next(left_data)
                    except StopIteration:
                        a_stopped = True
                        continue


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""

    def __init__(self) -> None:
        self.full = True

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        all_rows: tp.Any = dict()
        for row in rows:
            values: tp.Any = []
            for key in group_key:
                values.append(row[key])
            values = tuple(values)
            if values not in all_rows:
                all_rows[values] = True
                yield row


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column
        self.all = set(r'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~')

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = ''.join(filter(lambda c: c not in self.all, row[self.column]))
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = row[self.column].lower()
        yield row


class Idf(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, docs_cnt_column: str, rows_cnt_column: str, res_column: str):
        """
        :param column: name of column to process
        """
        self.docs_cnt_column = docs_cnt_column
        self.rows_cnt_column = rows_cnt_column
        self.res_column = res_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.res_column] = math.log(row[self.rows_cnt_column] / row[self.docs_cnt_column])
        yield row


class Log(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str, res_column: str):
        """
        :param column: name of column to process
        """
        self.column = column
        self.res_column = res_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.res_column] = math.log(row[self.column])
        yield row


class DayHour(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str, res_day_column: str, res_hour_column: str):
        """
        :param column: name of column to process
        """
        self.column = column
        self.res_day_column = res_day_column
        self.res_hour_column = res_hour_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        t: datetime.datetime
        try:
            t = datetime.datetime.strptime(row[self.column], "%Y%m%dT%H%M%S.%f")
        except ValueError:
            t = datetime.datetime.strptime(row[self.column], "%Y%m%dT%H%M%S")
        row[self.res_day_column] = t.strftime('%A')[:3]
        row[self.res_hour_column] = t.hour
        yield row


class TimeDelta(Mapper):
    """calculate time delta in seconds"""

    def __init__(self, start_column: str, end_column: str, res_column: str):
        """
        :param column: name of column to process
        """
        self.start_column = start_column
        self.end_column = end_column
        self.res_column = res_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        t1: tp.Any = row[self.start_column]
        t2: tp.Any = row[self.end_column]
        try:
            t1 = datetime.datetime.strptime(t1, "%Y%m%dT%H%M%S.%f")
        except ValueError:
            t1 = datetime.datetime.strptime(t1, "%Y%m%dT%H%M%S")
        try:
            t2 = datetime.datetime.strptime(t2, "%Y%m%dT%H%M%S.%f")
        except ValueError:
            t2 = datetime.datetime.strptime(t2, "%Y%m%dT%H%M%S")
        row[self.res_column] = (t2 - t1).total_seconds()
        yield row


class Distance(Mapper):
    """calculate time delta in seconds"""

    def __init__(self, point_a: str, point_b: str, res_column: str):
        """
        :param column: name of column to process
        """
        self.point_a = point_a
        self.point_b = point_b
        self.res_column = res_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        a: tp.Any = row[self.point_a]
        b: tp.Any = row[self.point_b]
        delta: tp.Any = math.radians(a[0]) - math.radians(b[0])
        length: tp.Any = (math.sin(math.radians(a[1])) * math.sin(math.radians(b[1]))) + \
                         (math.cos(math.radians(a[1])) * math.cos(math.radians(b[1])) * math.cos(delta))
        row[self.res_column] = math.acos(length) * EARTH_RADIUS_KM
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: tp.Optional[str] = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        if separator is None:
            self.separator = re.compile(r"\s+")
        else:
            self.separator = re.compile(separator)

    def __call__(self, row: TRow) -> TRowsGenerator:

        last_found = 0
        for m in re.finditer(self.separator, row[self.column]):
            ans = row.copy()
            ans[self.column] = row[self.column][last_found:m.start()]
            yield ans
            last_found = m.end()

        if last_found != len(row[self.column]):
            ans = row.copy()
            ans[self.column] = ans[self.column][last_found:]
            yield ans


class Product(Mapper):
    """Calculates product of multiple columns"""

    def __init__(self, columns: tp.Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        res = 1
        for i in self.columns:
            res *= row[i]
        row[self.result_column] = res
        yield row


class Applyer(Mapper):
    """Calculates product of multiple columns"""

    def __init__(self, columns: tp.Sequence[str], func: tp.Any, result_columns: tp.Sequence[str]) -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_columns = result_columns
        self.func = func

    def __call__(self, row: TRow) -> TRowsGenerator:
        args = []
        for arg_index in self.columns:
            args.append(row[arg_index])
        result = self.func(*args)
        for res, col in zip(result, self.result_columns):
            row[col] = res
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""

    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        ans = dict()
        for x in self.columns:
            y = row[x]
            ans.update({x: y})
        yield ans


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""

    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.full = False
        self.column_max = column
        self.n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        h: tp.Any = []
        for row in rows:
            heapq.heappush(h, (row[self.column_max], tuple(row.items())))
            if len(h) > self.n:
                heapq.heappop(h)

        yield from [dict(x[1]) for x in h]


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""

    def __init__(self, words_column: str, result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.full = True
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        all_words: tp.Any = dict()
        cnt_words: tp.Any = dict()
        for row in rows:
            values: tp.Any = []
            for key in group_key:
                values.append(row[key])
            values = tuple(values)

            if values not in all_words:
                all_words[values] = dict()
                cnt_words[values] = 0

            word = row[self.words_column]

            if word not in all_words[values]:
                all_words[values][word] = 0

            all_words[values][word] += 1
            cnt_words[values] += 1

        for k in all_words:
            for w in all_words[k]:
                ans = dict()
                for col, val in zip(group_key, k):
                    ans[col] = val
                ans[self.words_column] = w
                ans[self.result_column] = all_words[k][w] / cnt_words[k]
                yield ans


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.full = True
        self.column = column

    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        cnt: tp.Any = dict()
        for row in rows:
            values: tp.Any = []
            for key in group_key:
                values.append(row[key])
            values = tuple(values)
            if values not in cnt:
                cnt[values] = 0
            cnt[values] += 1
        for k in cnt:
            ans: tp.Any = dict()
            for col, val in zip(group_key, k):
                ans[col] = val
            ans[self.column] = cnt[k]
            yield ans


class CountTF(Reducer):
    """Calculate frequency of values in column with given count"""

    def __init__(self, words_column: str, count_column: str = 'count', result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param count_column: name for column with counts
        :param result_column: name for result column
        """
        self.full = False
        self.words_column = words_column
        self.count_column = count_column
        self.result_column = result_column

    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        words_counter: dict[str, int] = {}
        ans: TRow = {}
        n = 0
        for row in rows:
            if not ans:
                for key in group_key:
                    ans[key] = row[key]
            n += row[self.count_column]
            if row[self.words_column] in words_counter.keys():
                words_counter[row[self.words_column]] += row[self.count_column]
            else:
                words_counter[row[self.words_column]] = row[self.count_column]
        for key, value in words_counter.items():
            ans_row: TRow = {}
            for k in ans.keys():
                ans_row[k] = ans[k]
            ans_row[self.words_column] = key
            ans_row[self.result_column] = value / n
            yield ans_row


class Sum(Reducer):
    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.full = True
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        summ: tp.Any = dict()
        for row in rows:
            values: tp.Any = []
            for key in group_key:
                values.append(row[key])
            values = tuple(values)
            if values not in summ:
                summ[values] = 0
            summ[values] += row[self.column]
        for k in summ:
            ans: tp.Any = dict()
            for col, val in zip(group_key, k):
                ans[col] = val
            ans[self.column] = summ[k]
            yield ans


# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        empty: TRow = {}
        rows = list(rows_b)
        if rows != [empty]:
            for row_a in rows_a:
                if not row_a:
                    continue
                for row_b in rows:
                    ans: TRow = {}
                    for col in row_a.keys():
                        if col in row_b.keys() and row_a[col] != row_b[col]:
                            ans[col + self._a_suffix] = row_a[col]
                            ans[col + self._b_suffix] = row_b[col]
                        else:
                            ans[col] = row_a[col]
                    for col in row_b.keys():
                        if col not in row_a.keys():
                            ans[col] = row_b[col]
                    yield ans


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows = list(rows_b)
        if not rows:
            empty: TRow = {}
            rows.append(empty)
        for row_a in rows_a:
            for row_b in rows:
                ans: TRow = {}
                for col in row_a.keys():
                    ans[col] = row_a[col]
                for col in row_b.keys():
                    if col not in row_a.keys():
                        ans[col] = row_b[col]
                yield ans


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows = list(rows_b)
        for row_a in rows_a:
            if row_a == {}:
                continue
            for row_b in rows:
                ans: TRow = {}
                for col in row_a.keys():
                    ans[col] = row_a[col]
                for col in row_b.keys():
                    if col not in row_a.keys():
                        ans[col] = row_b[col]
                yield ans


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows = list(rows_a)
        for row_b in rows_b:
            if row_b == {}:
                continue
            for row_a in rows:
                ans: TRow = {}
                for col in row_a.keys():
                    ans[col] = row_a[col]
                for col in row_b.keys():
                    if col not in row_a.keys():
                        ans[col] = row_b[col]
                yield ans


class Mean(Reducer):
    def __init__(self, column: str) -> None:
        self.column = column
        self.full = True

    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        summ: tp.Any = dict()
        cnt: tp.Any = dict()
        for row in rows:
            values: tp.Any = []
            for key in group_key:
                values.append(row[key])
            values = tuple(values)
            if values not in cnt:
                cnt[values] = 0
                summ[values] = 0
            summ[values] += row[self.column]
            cnt[values] += 1
        for k in summ:
            ans: tp.Any = dict()
            for col, val in zip(group_key, k):
                ans[col] = val
            ans[self.column] = summ[k] / cnt[k]
            yield ans


class MeanSpeed(Reducer):
    def __init__(self, len_column: str, time_column: str, result_column: str) -> None:
        self.len_column = len_column
        self.time_column = time_column
        self.result_column = result_column
        self.full = False

    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        full_time: float = float(0)
        full_len: float = float(0)
        ans: tp.Any = dict()
        for row in rows:
            cur_time = row[self.time_column]
            cur_len = row[self.len_column]
            full_time += cur_time
            full_len += cur_len

            if len(ans) == 0:
                for key in group_key:
                    ans[key] = row[key]
        ans[self.result_column] = full_len / full_time * SEC_IN_HOUR
        yield ans
