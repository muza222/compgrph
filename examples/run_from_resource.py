import json
import click
import sys

from compgraph import algorithms
from compgraph import operations
import typing as tp


def parser(line: str) -> tp.Any:
    return json.loads(line)


@click.command()
@click.argument("task")
@click.argument("input_1")
@click.argument("input_2", required=False)
@click.argument("out_file_name", required=False)
def cli(task: tp.Any, input_1: tp.Any, input_2: tp.Any = None, out_file_name: tp.Any = None) -> None:
    if out_file_name is not None:
        out_file: tp.Any = open(out_file_name, 'w')
    else:
        out_file = sys.stdout
    if task == "word_count":
        graph = algorithms.word_count_graph(input_stream_name="f1")
        for line in graph.run(f1=operations.Read(input_1, parser)):
            print(json.dumps(line), file=out_file)
    elif task == "inverted_index_graph":
        graph = algorithms.inverted_index_graph(input_stream_name="f1")
        for line in graph.run(f1=operations.Read(input_1, parser)):
            print(json.dumps(line), file=out_file)

    elif task == "pmi_graph":
        graph = algorithms.pmi_graph(input_stream_name="f1")
        for line in graph.run(f1=operations.Read(input_1, parser)):
            print(json.dumps(line), file=out_file)
    elif task == "yandex_maps_graph":
        graph = algorithms.yandex_maps_graph(
            input_stream_name_length="f1",
            input_stream_name_time="f2"
        )
        for line in graph.run(f1=operations.Read(input_1, parser),
                              f2=operations.Read(input_2, parser)):
            print(json.dumps(line), file=out_file)
    else:
        raise RuntimeError("unknown algorithm")


if __name__ == "__main__":
    cli()
