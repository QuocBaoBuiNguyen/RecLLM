"""Shared DataLoader factories for Q-Former alignment pipelines.

Used by both stage 1 representation training and stage 2 generative
pretraining so the dataset/collate/loader plumbing stays in one place.
"""

import os
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from sigllm.datasets.qformer.qformer_alignment_dataset import QFormerAlignmentDataset


FilterFn = Callable[[QFormerAlignmentDataset], Dataset]


def qformer_collate(batch):
    """Stack tensors, keep other fields as lists."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out


def build_qformer_loader(
    cfg,
    filename: str,
    shuffle: bool,
    filter_fn: Optional[FilterFn] = None,
) -> DataLoader:
    """Build a single Q-Former DataLoader from a samples pickle.

    Parameters
    ----------
    cfg
        Config namespace with ``batch_size`` and ``num_workers`` attributes.
    filename
        Path to a ``.pkl`` file produced by ``QFormerAlignmentBuilder``.
    shuffle
        Whether to shuffle the loader.
    filter_fn
        Optional callable that receives the loaded ``QFormerAlignmentDataset``
        and returns a (possibly subsetted) ``Dataset``. Use this in stage 2
        to keep only ``item_text`` samples.
    """
    dataset: Dataset = QFormerAlignmentDataset(filename=filename)
    if filter_fn is not None:
        dataset = filter_fn(dataset)
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=shuffle,
        collate_fn=qformer_collate,
        num_workers=int(cfg.num_workers),
    )


def build_qformer_loaders(
    cfg,
    data_dir: str,
    train_filename: str = "train_qformer_ood2.pkl",
    val_filename: str = "valid_qformer_ood2.pkl",
    test_filename: Optional[str] = "test_qformer_ood2.pkl",
    filter_fn: Optional[FilterFn] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build train/val/test loaders for the Q-Former alignment task.

    Pass ``test_filename=None`` if the calling stage has no test split
    (stage 2 generative pretraining); the returned ``test_loader`` is
    ``None`` in that case.
    """
    train_loader = build_qformer_loader(
        cfg, filename=os.path.join(data_dir, train_filename), shuffle=True, filter_fn=filter_fn
    )
    val_loader = build_qformer_loader(
        cfg, filename=os.path.join(data_dir, val_filename), shuffle=False, filter_fn=filter_fn
    )
    test_loader = None
    if test_filename is not None:
        test_loader = build_qformer_loader(
            cfg, filename=os.path.join(data_dir, test_filename), shuffle=False, filter_fn=filter_fn
        )
    return train_loader, val_loader, test_loader
