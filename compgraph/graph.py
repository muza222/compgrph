import copy
import typing as tp

from . import operations as ops
from .external_sort import ExternalSort


class Graph:
    """Computational graph implementation"""

    def __init__(self) -> None:
        self._operations: list[ops.Operation] = []
        self._join_list: list['Graph'] = []

    @staticmethod
    def graph_from_iter(name: str) -> 'Graph':
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        res: Graph = Graph()
        res._operations = [ops.ReadIterFactory(name=name)]
        return res

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> 'Graph':
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        res: Graph = Graph()
        res._operations = [ops.Read(filename=filename, parser=parser)]
        return res

    def map(self, mapper: ops.Mapper) -> 'Graph':
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        res: Graph = copy.deepcopy(self)
        res._operations.append(ops.Map(mapper=mapper))
        return res

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        res: Graph = copy.deepcopy(self)
        res._operations.append(ops.Reduce(keys=keys, reducer=reducer))
        return res

    def sort(self, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        res: Graph = copy.deepcopy(self)
        res._operations.append(ExternalSort(keys=keys))
        return res

    def join(self, joiner: ops.Joiner, join_graph: 'Graph', keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        res: Graph = copy.deepcopy(self)
        res._operations.append(ops.Join(joiner=joiner, keys=keys))
        res._join_list.append(join_graph)
        return res

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        """Single method to start execution; data sources passed as kwargs"""
        assert len(self._operations) > 0 and (isinstance(self._operations[0], ops.ReadIterFactory) or
                                              isinstance(self._operations[0], ops.Read)), 'Wrong first operation'
        table = self._operations[0](**kwargs)
        cur_joiner = 0

        for operation in self._operations[1:]:
            if isinstance(operation, ops.Join):
                right_table = self._join_list[cur_joiner].run(**kwargs)
                cur_joiner += 1

                table = operation(table, right_table)
            else:
                table = operation(table)

        yield from table
