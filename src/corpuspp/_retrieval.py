from more_itertools import chunked
import pandas as pd

def dense_retriever(index_path: str, checkpoint: str, batch_size: int = 128, **kwargs):
    from pyterrier_dr import HgfBiEncoder, FlexIndex

    index = FlexIndex(index_path)
    model = HgfBiEncoder.from_pretrained(checkpoint, batch_size=batch_size, verbose=True)

    return model >> index.np_retriever()


def dense_no_index_retriever(index_path: str, checkpoint: str, batch_size: int = 128, **kwargs):
    from pyterrier_dr import HgfBiEncoder, FlexIndex

    model = HgfBiEncoder.from_pretrained(checkpoint, batch_size=batch_size, verbose=True)

    return model


def sparse_retriever(index_path: str, checkpoint: str = 'naver/splade-cocondenser-ensembledistil', batch_size: int = 128, threads: int = 4, **kwargs):
    from pyt_splade import Splade
    from pyterrier_pisa import PisaIndex

    index = PisaIndex(index_path, threads=threads, stemmer='none').quantized()
    splade = Splade(model=checkpoint)

    return splade.query_encoder(verbose=True, batch_size=batch_size) >> index


def lexical_retriever(index_path: str, threads: int = 4, **kwargs):
    from pyterrier_pisa import PisaIndex

    index = PisaIndex(index_path, threads=threads)

    return index.bm25()


def _batched_wrapper(queries: pd.DataFrame, func, batch_size: int = 128):
    for batch in chunked(queries.iterrows(), batch_size):
        batch_df = pd.DataFrame([row[1] for row in batch])  # Convert back to DataFrame
        yield func(batch_df)


__all__ = ['dense_retriever', 'dense_no_index_retriever', 'sparse_retriever', 'lexical_retriever', '_batched_wrapper']
