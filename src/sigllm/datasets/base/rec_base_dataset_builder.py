"""Base builder for recommendation datasets."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from torch import dist

from sigllm.common.config import Config
from sigllm.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from sigllm.common.logging_utils import NotebookLogger

LOGGER = NotebookLogger.rich_logger("sigllm.rec_base_dataset_builder")

def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)

TRAIN_ITEM_IDS_CACHE = "train_item_ids.npy"


def load_train_item_ids(storage_path) -> set:
    """Item ids that appear in ``train_ood2.pkl``, i.e. the items the frozen MF
    teacher actually received gradients for.

    Cached as a small sidecar ``.npy`` next to the pickles: Amazon-Book's
    ``train_ood2.pkl`` is 505 MB and eval-only runs never load the train split
    otherwise, so paying that read on every run (and in every DDP rank) is not
    acceptable. Delete the sidecar to force a rebuild after re-preprocessing.
    """

    storage = Path(storage_path)
    cache_path = storage / TRAIN_ITEM_IDS_CACHE
    if cache_path.exists():
        ids = np.load(cache_path)
        log_step("Loaded train item ids (cached)", f"{cache_path} | {ids.size} items")
        return set(ids.tolist())

    train_path = storage / "train_ood2.pkl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"mark_cold_items=True needs {train_path} (or a prebuilt "
            f"{cache_path}) to know which items the MF teacher was trained on."
        )

    log_step("Building train item ids", f"reading {train_path} (one-off)")
    ids = np.unique(pd.read_pickle(train_path)["iid"].to_numpy())
    if is_main_process():
        try:
            np.save(cache_path, ids)
            log_step("Cached train item ids", f"{cache_path} | {ids.size} items")
        except OSError as exc:  # read-only mount etc. — cache is an optimization
            log_step("Could not cache train item ids", str(exc))
    return set(ids.tolist())


class RecBaseDatasetBuilder(ABC):
    """Abstract base for dataset builders."""
    train_dataset_cls = None

    def __init__(self, dataset_config) -> None:
        self.dataset_config = dataset_config
    
    def build_datasets(self, evaluate_only=False):
        """Construct dataset instances for training/validation/test."""

        if is_dist_avail_and_initialized():
            dist.barrier()
            
        dataset_cls = self.train_dataset_cls

        build_info = self.dataset_config.build_info
        storage_path = build_info.storage

        if storage_path is None:
            log_step("Warning", f"storage path {storage_path} does not exist.") 

        datasets = dict()

        # Cold-ITEM gating (P0). When on, every split learns which items the MF
        # teacher was actually trained on, so the model can swap a learned
        # "no-CF" token in for untrained embeddings instead of injecting them as
        # if they were valid. Must be paired with model.cold_item_token=True.
        mark_cold_items = bool(
            getattr(build_info, "get", lambda *a: False)("mark_cold_items", False)
        )
        train_item_ids = load_train_item_ids(storage_path) if mark_cold_items else None

        if not evaluate_only:
            datasets["train"] = dataset_cls(
                dataset_config=self.dataset_config,
                filename="train_ood2.pkl",
                train_item_ids=train_item_ids,
            )

            datasets["valid"] = dataset_cls(
                dataset_config=self.dataset_config,
                filename="valid_ood2.pkl",
                train_item_ids=train_item_ids,
            )
            datasets["test"] = dataset_cls(
                dataset_config=self.dataset_config,
                filename="test_ood2.pkl",
                train_item_ids=train_item_ids,
            )
        else:
            datasets["test"] = dataset_cls(
                dataset_config=self.dataset_config,
                filename="test_ood2.pkl",
                train_item_ids=train_item_ids,
            )
            warm_cold_filename = "test_warm_cold_ood2.pkl"
            warm_cold_path = Path(storage_path) / warm_cold_filename
            if warm_cold_path.exists():
                datasets["test_warm"] = dataset_cls(
                    dataset_config=self.dataset_config,
                    filename=warm_cold_filename,
                    subset="warm",
                    train_item_ids=train_item_ids,
                )
                datasets["test_cold"] = dataset_cls(
                    dataset_config=self.dataset_config,
                    filename=warm_cold_filename,
                    subset="cold",
                    train_item_ids=train_item_ids,
                )
            else:
                log_step("Skipping warm/cold subsets", f"file not found: {warm_cold_path}")

        return datasets
