from compgraph import graph
from compgraph import operations
import typing as tp


# https://ru.wikihow.com/умножать


def sample_product() -> tp.Any:
    data = [
        {'a': 1, 'b': 2},
        {'a': 2, 'b': 3},
        {'a': 1, 'b': 3},
        {'a': 4, 'b': 1}]
    product_graph = graph.Graph.graph_from_iter('data') \
        .map(operations.Product(['a', 'b'], 'a * b'))
    return list(product_graph.run(data=lambda: iter(data)))
