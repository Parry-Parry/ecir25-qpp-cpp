def dense_indexer(index_path: str, checkpoint: str, batch_size: int = 128, **kwargs):
    from pyterrier_dr import HgfBiEncoder, FlexIndex

    index = FlexIndex(index_path)
    model = HgfBiEncoder.from_pretrained(checkpoint, batch_size=batch_size, verbose=True)

    return model >> index


def sparse_indexer(index_path: str, checkpoint: str = 'naver/splade-cocondenser-ensembledistil', batch_size: int = 128, threads: int = 4, **kwargs):
    from pyt_splade import Splade
    from pyterrier_pisa import PisaIndex

    index = PisaIndex(index_path, threads=threads, stemmer='none')
    splade = Splade(model=checkpoint)

    return splade.doc_encoder(verbose=True) >> index.toks_indexer()


def lexical_indexer(index_path: str, threads: int = 4, **kwargs):
    from pyterrier_pisa import PisaIndex

    index = PisaIndex(index_path, threads=threads)

    return index


__all__ = ['dense_indexer', 'sparse_indexer', 'lexical_indexer']
