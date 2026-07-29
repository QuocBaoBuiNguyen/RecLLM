"""Build Q-Former alignment dataset pickles for Stage 1 / Stage 2.

Run this script once before training. It reads the raw
``{train,valid,test}_ood2.pkl`` files under ``datasets[*].path`` and emits
``{train,valid,test}_qformer_ood2.pkl`` in the same directory. Stage 1
representation and Stage 2 generative pipelines then load these pickles via
``sigllm.datasets.qformer.qformer_loader``.

Build parameters come from ``run.qformer_stage1`` in the YAML config
(``seed``, ``item_pair_window``, ``max_item_item_pairs``,
``max_user_item_pairs``, ``include_user_item``).

Usage
-----
    python -m sigllm.pipelines.multimodal.build_qformer_dataset \
        --cfg-path configs/config.yaml
"""

import argparse
import os
from typing import Optional

from sigllm.common import NotebookLogger
from sigllm.common.config import Config
from sigllm.datasets.qformer.qformer_alignment_builder import QFormerAlignmentBuilder


LOGGER = NotebookLogger.rich_logger("sigllm.build_qformer_dataset")


SPLITS = (
    ("train", "train_ood2.pkl", "train_qformer_ood2.pkl"),
    ("valid", "valid_ood2.pkl", "valid_qformer_ood2.pkl"),
    ("test", "test_ood2.pkl", "test_qformer_ood2.pkl"),
)


def log_step(title: str, detail: Optional[str] = None) -> None:
    LOGGER.info(title if detail is None else f"{title} | {detail}")


def build_qformer_pkls(cfg) -> None:
    """Build all Q-Former alignment pickles from a parsed ``Config``.

    Parameters
    ----------
    cfg
        Parsed ``sigllm.common.config.Config`` object. Must expose
        ``run_cfg.qformer_stage1`` and ``datasets_cfg``.
    """
    stage1_cfg = cfg.run_cfg.get("qformer_stage1")
    if stage1_cfg is None:
        raise KeyError("Missing 'run.qformer_stage1' section in configuration.")

    first_dataset_key = list(cfg.datasets_cfg.keys())[0]
    data_dir = cfg.datasets_cfg[first_dataset_key].path

    seed = int(stage1_cfg.seed)
    item_pair_window = int(stage1_cfg.get("item_pair_window", 2))
    max_item_item_pairs = stage1_cfg.get("max_item_item_pairs", None)
    max_user_item_pairs = stage1_cfg.get("max_user_item_pairs", None)
    include_user_item = bool(stage1_cfg.get("include_user_item", False))
    # SHARED history cap: read from the dataset config so Stage 1's history
    # pooling is pretrained on exactly the sequence length Stage 3's
    # MovieOODDataset feeds at run time (one key, no 50-vs-10 drift).
    dataset_cfg = cfg.datasets_cfg[first_dataset_key]
    max_history_length = int(dataset_cfg.get("max_history_length", 10))
    # Same SeLLa-parity filter MovieOODDataset applies at Stage 3: without it,
    # Stage 1 trains on short-history rows (and the item_item pairs derived from
    # them) that no downstream stage ever sees.
    min_positive_history = int(dataset_cfg.get("min_positive_history", 1))
    # Item-level holdout for the item_text objective (0.0/0.0 = legacy: every
    # split gets item_text for all of its items, which leaks train pairs into
    # val under a temporal interaction split).
    item_text_valid_frac = float(stage1_cfg.get("item_text_valid_frac", 0.0))
    item_text_test_frac = float(stage1_cfg.get("item_text_test_frac", 0.0))
    item_text_split_seed = int(stage1_cfg.get("item_text_split_seed", seed))

    log_step(
        "Build config",
        (
            f"data_dir={data_dir}, seed={seed}, item_pair_window={item_pair_window}, "
            f"max_item_item_pairs={max_item_item_pairs}, "
            f"max_user_item_pairs={max_user_item_pairs}, "
            f"include_user_item={include_user_item}, "
            f"max_history_length={max_history_length}, "
            f"min_positive_history={min_positive_history}, "
            f"item_text_holdout=({item_text_valid_frac}, {item_text_test_frac}, "
            f"seed={item_text_split_seed})"
        ),
    )

    for split_name, input_name, output_name in SPLITS:
        input_path = os.path.join(data_dir, input_name)
        output_path = os.path.join(data_dir, output_name)
        QFormerAlignmentBuilder.build_qformer_alignment_samples(
            input_pkl_path=input_path,
            output_path=output_path,
            seed=seed,
            item_pair_window=item_pair_window,
            max_item_item_pairs=max_item_item_pairs,
            max_user_item_pairs=max_user_item_pairs,
            include_user_item=include_user_item,
            max_history_length=max_history_length,
            item_text_split=split_name,
            item_text_valid_frac=item_text_valid_frac,
            item_text_test_frac=item_text_test_frac,
            item_text_split_seed=item_text_split_seed,
            min_positive_history=min_positive_history,
        )
        log_step("Built Q-Former pkl", f"{input_path} -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Q-Former alignment dataset pickles for Stage 1 / Stage 2"
    )
    parser.add_argument(
        "--cfg-path",
        default="configs/config.yaml",
        type=str,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override config settings in key=value format.",
    )
    return parser.parse_args()


def main():
    cfg = Config(parse_args())
    build_qformer_pkls(cfg)


if __name__ == "__main__":
    main()
