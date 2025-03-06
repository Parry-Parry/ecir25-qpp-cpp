def dense_retriever(index_path: str, checkpoint: str, batch_size: int = 128, **kwargs):
    from pyterrier_dr import HgfBiEncoder, FlexIndex

    index = FlexIndex(index_path)
    model = HgfBiEncoder.from_pretrained(checkpoint, batch_size=batch_size, verbose=True)

    return model >> index.torch_retriever(qbatch=16)


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


__all__ = ['dense_retriever', 'sparse_retriever', 'lexical_retriever']
