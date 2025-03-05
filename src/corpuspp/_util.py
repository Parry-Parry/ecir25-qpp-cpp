import pandas as pd


def convert_unicode(text):
    try:
        return text.encode('utf-8').decode('unicode_escape')
    except UnicodeDecodeError:
        return text


def queries_from_jsonl(file: str):
    queries = pd.read_json(file, lines=True, orient='records')
    queries['query'] = queries['query'].apply(convert_unicode)
    queries['qid'] = [str(i) for i in range(len(queries))]
    return queries
