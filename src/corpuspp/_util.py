import pandas as pd


def queries_from_jsonl(file: str):
    return pd.read_json(file, lines=True, orient='records')
