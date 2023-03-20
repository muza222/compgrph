from copy import deepcopy

from . import Graph, operations

EARTH_RADIUS_KM: int = 6373


def word_count_graph(input_stream_name: str, text_column: str = 'text', count_column: str = 'count') -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    graph = Graph.graph_from_iter(input_stream_name)
    return graph \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .sort([count_column, text_column])


def inverted_index_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
                         result_column: str = 'tf_idf') -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    graph = Graph.graph_from_iter(input_stream_name)

    splitted = deepcopy(graph) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))

    tf = deepcopy(splitted) \
        .sort([doc_column]) \
        .reduce(operations.TermFrequency(text_column, 'tf'), [doc_column])

    idf = splitted \
        .sort([text_column, doc_column]) \
        .reduce(operations.FirstReducer(), [text_column, doc_column]) \
        .sort([text_column]) \
        .reduce(operations.Count('docs_cnt'), [text_column]) \
        .join(operations.InnerJoiner(), graph.reduce(operations.Count('rows_cnt'), []), []) \
        .map(operations.Idf('docs_cnt', 'rows_cnt', 'idf'))

    return tf \
        .sort([text_column]) \
        .join(operations.InnerJoiner(), idf, [text_column]) \
        .map(operations.Applyer(['idf', 'tf'], lambda x, y: (x * y,), [result_column])) \
        .map(operations.Project([doc_column, text_column, result_column])) \
        .sort([text_column]) \
        .reduce(operations.TopN(result_column, 3), [text_column])


def pmi_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
              result_column: str = 'pmi') -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""
    graph = Graph.graph_from_iter(input_stream_name)

    filtered = graph \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([doc_column, text_column]) \
        .reduce(operations.Count("word_in_document_cnt"), [doc_column, text_column]) \
        .map(operations.Filter(lambda row: len(row[text_column]) > 4 and row["word_in_document_cnt"] >= 2))

    doc_tf = deepcopy(filtered) \
        .reduce(operations.CountTF(text_column, "word_in_document_cnt", "doc_tf"), [doc_column]) \
        .sort([text_column])

    total_tf = filtered.sort([text_column]) \
        .reduce(operations.CountTF(text_column, "word_in_document_cnt", "total_tf"), []) \
        .sort([text_column])

    return doc_tf.join(operations.InnerJoiner(), total_tf, [text_column]) \
        .map(operations.Applyer(['total_tf'], lambda x: (1 / x,), ['total_tf'])) \
        .map(operations.Product(['doc_tf', 'total_tf'], 'pmi')) \
        .map(operations.Log('pmi', 'pmi')) \
        .sort([doc_column]) \
        .reduce(operations.TopN(result_column, 10), [doc_column]) \
        .map(operations.Project([result_column, doc_column, text_column])) \
        .map(operations.Applyer([result_column], lambda x: (1 / x,), [result_column])) \
        .sort([doc_column, result_column, text_column]) \
        .map(operations.Applyer([result_column], lambda x: (1 / x,), [result_column]))


def yandex_maps_graph(input_stream_name_time: str, input_stream_name_length: str,
                      enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                      edge_id_column: str = 'edge_id', start_coord_column: str = 'start',
                      end_coord_column: str = 'end',
                      weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                      speed_result_column: str = 'speed') -> Graph:

    """Constructs graph which measures average speed in km/h depending on the weekday and hours"""

    graph = Graph.graph_from_iter(input_stream_name_length)

    graph_distance = graph \
        .map(operations.Distance(start_coord_column, end_coord_column, 'len')) \
        .sort([edge_id_column])

    graph_time = Graph.graph_from_iter(input_stream_name_time)

    graph_with_date = graph_time \
        .sort([edge_id_column]) \
        .map(operations.TimeDelta(enter_time_column, leave_time_column, 'time_delta')) \
        .map(operations.DayHour(enter_time_column, weekday_result_column, hour_result_column)) \
        .join(operations.InnerJoiner(), graph_distance, [edge_id_column]) \
        .sort([edge_id_column]) \
        .reduce(operations.MeanSpeed('len', 'time_delta', speed_result_column),
                [weekday_result_column, hour_result_column]) \
        .map(operations.Project([weekday_result_column, hour_result_column, speed_result_column])) \
        .reduce(operations.Mean(speed_result_column), [hour_result_column, weekday_result_column])

    return graph_with_date
