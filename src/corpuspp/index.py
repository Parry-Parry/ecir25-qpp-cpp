import pyterrier as pt
from fire import Fire

from . import _index as indexers


def main(
        ir_dataset: str,
        index_path: str,
        checkpoint: str = None,
        batch_size: int = 128,
        threads: int = 4,
        retriever: str = 'dense',
):
    if f"{retriever}_indexer" not in indexers.__all__:
        raise ValueError(f"Invalid retriever: {retriever}")
    indexer_obj = getattr(indexers, f"{retriever}_indexer")(
        index_path=index_path,
        checkpoint=checkpoint,
        batch_size=batch_size,
        threads=threads
        )
    dataset = pt.get_dataset(f"irds:{ir_dataset}")
    indexer_obj.index(dataset.get_corpus_iter())

    print(f"Index written to {index_path}")
    return 0


if __name__ == '__main__':
    Fire(main)
