import pandas as pd


def queries_from_jsonl(file: str):
    queries = pd.read_json(file, lines=True, orient='records')
    queries['qid'] = [str(i) for i in range(len(queries))]
    return queries
