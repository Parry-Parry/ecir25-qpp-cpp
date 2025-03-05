import os
import pandas as pd
import ir_datasets as irds
import pyterrier as pt
from fire import Fire

from . import _retrieval as retrievers
from ._util import queries_from_jsonl, filter_alnum_spaces


def main(
        index_path: str,
        output_directory: str,
        ir_dataset: str = None,
        query_path: str = None,
        checkpoint: str = None,
        batch_size: int = 128,
        threads: int = 4,
        retriever: str = 'dense',
        depth: int = 1000,
):
    assert query_path is not None or ir_dataset is not None, "Either query_path or ir_dataset must be provided"
    if f"{retriever}_retriever" not in retrievers.__all__:
        raise ValueError(f"Invalid retriever: {retriever}")
    retriever_obj = getattr(retrievers, f"{retriever}_retriever")(
        index_path=index_path,
        checkpoint=checkpoint,
        batch_size=batch_size,
        threads=threads
        )
    pipe = retriever_obj % depth

    if query_path is not None:
        queries = queries_from_jsonl(query_path)
    else:
        dataset = irds.load(ir_dataset)
        queries = pd.DataFram(dataset.queries_iter()).rename(columns={'query_id': 'qid', 'text': 'query'})
    if retriever == 'lexical':
        queries['query'] = queries['query'].apply(filter_alnum_spaces)
    print(queries.head())
    result = pipe.transform(queries)
    index_path_basename = os.path.basename(index_path)
    output_file = os.path.join(output_directory, f"{retriever}.{index_path_basename}.{depth}.tsv.gz")
    if len(result) == 0:
        print("No results to write")
        return
    pt.io.write_results(result, output_file)
    print(f"Results written to {output_file}")
    return 0


if __name__ == '__main__':
    Fire(main)
