"""Builder for the BookOOD dataset (Amazon-Book preprocessed pickles).

Schema produced by ``data_preprocessing_book.build_amazon_book`` is identical
to MovieLens (``uid, iid, label, timestamp, his, his_title, title, genres,
flag, not_cold``), so we reuse ``MovieOODDataset`` as the underlying record
reader rather than duplicating it. The builder is registered separately so
``cfg.datasets`` can route to ``book_ood`` while keeping the movie config
intact.
"""

from __future__ import annotations

from sigllm.common.registry import registry
from sigllm.datasets.base.rec_base_dataset_builder import RecBaseDatasetBuilder
from sigllm.datasets.movie.movie_ood_dataset import MovieOODDataset


@registry.register_builder("book_ood")
class BookOODBuilder(RecBaseDatasetBuilder):
    """Construct BookOOD dataset splits using the MovieOOD record reader."""

    train_dataset_cls = MovieOODDataset
