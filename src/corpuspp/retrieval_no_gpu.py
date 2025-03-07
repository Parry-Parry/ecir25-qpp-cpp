import os
import pandas as pd
import ir_datasets as irds
import pyterrier as pt
from fire import Fire
import ast
import numpy as np

from . import _retrieval as retrievers


def main(
        index_path: str,
        output_directory: str,
        query_path: str = None,
        batch_size: int = 128,
        threads: int = 4,
        retriever: str = 'dense_no_retrieval',
        depth: int = 1000,
):
    assert query_path is not None, "query_path must be provided"
    if f"{retriever}_retriever" not in retrievers.__all__:
        raise ValueError(f"Invalid retriever: {retriever}")
    retriever_obj = getattr(retrievers, f"{retriever}_retriever")(
        index_path=index_path,
        checkpoint='NA',
        batch_size=batch_size,
        threads=threads
        )
    pipe = retriever_obj % depth
    if query_path is not None:
        queries = pd.read_csv(query_path, sep='\t')
        queries['query_vec'] = queries['query_vec'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
    else:
        dataset = irds.load(ir_dataset)
        queries = pd.DataFrame(dataset.queries_iter()).rename(columns={'query_id': 'qid', 'text': 'query'})
    result = pipe.transform(queries)
    index_path_basename = os.path.basename(index_path)
    retriever = 'sparse' if 'sparse' in retriever else 'dense'
    output_file = os.path.join(output_directory, f"{retriever}.{index_path_basename}.{depth}.tsv.gz")
    if len(result) == 0:
        print("No results to write")
        return
    pt.io.write_results(result, output_file)
    print(f"Results written to {output_file}")
    return 0


if __name__ == '__main__':
    Fire(main)
