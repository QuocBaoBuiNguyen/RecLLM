"""Amazon-Book preprocessing for SigLLM.

Unlike the ML-1M pipeline (``data_preprocessing.py``) which builds the OOD2
splits from raw ``ratings.dat`` files, the Amazon-Book splits come from an
upstream collaborator as pre-built pickles bundled in
``data/raw/amazon_book.zip``. This module therefore reduces to two tasks:

1. Extract ``train/valid/test/valid_small_ood2.pkl`` from the zip into
   ``data/processed/amazon-book/`` (idempotent).
2. Inject a synthetic ``genres`` column (constant ``"Books"``) on each row
   so the downstream Q-Former alignment builder
   (``qformer_alignment_builder.py``) — which currently requires a
   pipe-joined ``genres`` field — works without code changes.

Output schema matches MovieLens exactly:
    uid, iid, label, timestamp, his, his_title, title, genres, flag, not_cold

Run as:
    python -m sigllm.datasets.data_preprocessing_book
"""

import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from sigllm.common import NotebookLogger

LOGGER = NotebookLogger.rich_logger("sigllm.data_prep_book")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ZIP_PATH = os.environ.get(
    "SIGLLM_BOOK_ZIP",
    str(_REPO_ROOT / "data" / "raw" / "amazon_book.zip"),
)
_DEFAULT_OUT_DIR = os.environ.get(
    "SIGLLM_BOOK_PROCESSED_DIR",
    str(_REPO_ROOT / "data" / "processed" / "amazon-book"),
)

_EXPECTED_PKLS = (
    "train_ood2.pkl",
    "valid_ood2.pkl",
    "test_ood2.pkl",
    "valid_small_ood2.pkl",
)

_GENRES_FILL = "Books"


def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""
    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)


def _extract_zip(zip_path: str, out_dir: str) -> None:
    """Extract the four expected pickles from ``zip_path`` into ``out_dir``.

    The supplied zip stores files under a top-level ``book/`` prefix
    (``book/train_ood2.pkl``, …). We flatten that prefix on extract so the
    output directory layout matches ML-1M.
    """
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        log_step("Zip members", ", ".join(members))
        for member in members:
            name = os.path.basename(member)
            if name not in _EXPECTED_PKLS:
                continue
            target = os.path.join(out_dir, name)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            log_step("Extracted", f"{member} -> {target}")


def _ensure_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Add a constant ``genres`` column if missing.

    Amazon-Book pickles ship without genres. The downstream Q-Former
    alignment builder requires a non-empty pipe-joined ``genres`` field
    per item, so we fill with the constant ``"Books"``. This makes the
    item-text template render as ``"Title: <title>. Genres: Books."`` —
    a single bucket carries no per-item signal but satisfies the schema
    contract; the bulk of the item-text discriminator is the title.
    """
    if "genres" in df.columns:
        return df
    df = df.copy()
    df["genres"] = _GENRES_FILL
    return df


def build_amazon_book(
    zip_path: str = _DEFAULT_ZIP_PATH,
    out_dir: str = _DEFAULT_OUT_DIR,
    force_extract: bool = False,
) -> tuple:
    """Materialise Amazon-Book OOD2 splits in ``out_dir``.

    Workflow
    --------
    1. If any of the expected pickles is missing in ``out_dir`` (or
       ``force_extract=True``), extract them from ``zip_path``.
    2. Load each pickle, add a constant ``genres`` column if missing,
       and re-save in place.
    3. Return ``(train_df, valid_df, test_df, valid_small_df)``.

    Parameters
    ----------
    zip_path
        Path to ``amazon_book.zip`` containing ``book/*_ood2.pkl``.
    out_dir
        Output directory for ``*_ood2.pkl`` files. Created if missing.
    force_extract
        When True, re-extract from the zip even if the target files
        already exist (useful if the upstream artifact changed).

    Returns
    -------
    (train_df, valid_df, test_df, valid_small_df)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    missing = [name for name in _EXPECTED_PKLS if not (out_path / name).exists()]
    if missing or force_extract:
        log_step(
            "[1/3] Extract zip",
            f"zip={zip_path} missing={missing} force={force_extract}",
        )
        if not os.path.exists(zip_path):
            raise FileNotFoundError(
                f"Amazon-Book zip not found at {zip_path}. "
                "Place the upstream amazon_book.zip there or set "
                "$SIGLLM_BOOK_ZIP."
            )
        _extract_zip(zip_path, out_dir)
    else:
        log_step("[1/3] Skip extract", f"all 4 pickles present in {out_dir}")

    log_step("[2/3] Inject genres column", f"fill='{_GENRES_FILL}' if missing")
    dfs = {}
    for name in _EXPECTED_PKLS:
        path = out_path / name
        df = pd.read_pickle(path)
        before_cols = list(df.columns)
        df = _ensure_genres(df)
        if list(df.columns) != before_cols:
            df.to_pickle(path)
            log_step(f"  Added genres -> {name}", f"shape={df.shape}")
        else:
            log_step(f"  Genres already present -> {name}", f"shape={df.shape}")
        dfs[name] = df

    train_ = dfs["train_ood2.pkl"]
    valid_ = dfs["valid_ood2.pkl"]
    test_ = dfs["test_ood2.pkl"]
    valid_small_ = dfs["valid_small_ood2.pkl"]

    log_step(
        "[3/3] Final split sizes",
        f"train={len(train_):,}, valid={len(valid_):,}, "
        f"test={len(test_):,}, valid_small={len(valid_small_):,}",
    )
    log_step(
        "ID ranges",
        f"max_uid={int(train_['uid'].max())}, "
        f"max_iid={int(train_['iid'].max())}",
    )
    log_step("Train columns", ", ".join(train_.columns))

    return train_, valid_, test_, valid_small_


if __name__ == "__main__":
    if "src" not in sys.path and not sys.path[0].endswith("src"):
        sys.path.insert(0, "src")

    train_df, valid_df, test_df, valid_small_df = build_amazon_book()

    log_step(
        "Preprocessing complete",
        f"train={len(train_df)}, valid={len(valid_df)}, "
        f"test={len(test_df)}, valid_small={len(valid_small_df)}",
    )
