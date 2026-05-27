import copy
import gzip
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sigllm.common import NotebookLogger

LOGGER = NotebookLogger.rich_logger("sigllm.data_prep_book")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RAW_DIR = os.environ.get(
    "SIGLLM_BOOK_RAW_DIR", str(_REPO_ROOT / "data" / "raw" / "amazon-book")
)
_DEFAULT_OUT_DIR = os.environ.get(
    "SIGLLM_BOOK_PROCESSED_DIR", str(_REPO_ROOT / "data" / "processed" / "amazon-book")
)


def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""
    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)


def preview_row_for_log(df: pd.DataFrame, max_list_items: int = 5) -> str:
    """Return a compact preview of one row that is about to be serialized."""
    if df.empty:
        return "<empty dataframe>"

    row = df.iloc[0].to_dict()
    preview = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, list):
            suffix = " ..." if len(value) > max_list_items else ""
            preview[key] = f"{value[:max_list_items]}{suffix} (len={len(value)})"
        else:
            preview[key] = value
    return str(preview)


def _parse_gzip_json(path: str):
    """Yield dicts from a gzipped JSON-lines file."""
    with gzip.open(path, "rb") as f:
        for line in f:
            yield json.loads(line)


def _category_to_genres(category) -> str:
    """Convert Amazon ``category`` field (list of strings) to a pipe-joined
    string in the same shape as MovieLens ``genres`` (e.g. ``"Books|Fiction|Mystery"``).
    Single-string or NaN inputs are handled defensively.
    """
    if isinstance(category, list):
        return "|".join(str(c).strip() for c in category if str(c).strip())
    if isinstance(category, str):
        return category.strip()
    return ""


def deal_with_each_u(x, u):
    """Build sequential interaction samples for a single user.

    Mirrors ``data_preprocessing.deal_with_each_u`` but uses ``genres``
    derived from Amazon ``category`` (pipe-joined string) so downstream
    Q-Former / movie pipeline can read books with the same schema.
    """
    items = np.array(x.iid)
    labels = np.array(x.label)
    titles = np.array(x.title)
    genres = np.array(x.genres)
    timestamp = np.array(x.timestamp)
    flags = np.array(x.flag)
    his = [0]  # adding a '0' by default
    his_title = [""]
    results = []
    for i in range(items.shape[0]):
        results.append(
            (
                u,
                items[i],
                timestamp[i],
                np.array(his),
                copy.copy(his_title),
                titles[i],
                labels[i],
                genres[i],
                flags[i],
            )
        )
        if labels[i] > 0:
            his.append(items[i])
            his_title.append(titles[i])
    return results


def build_amazon_book(
    raw_dir: str = _DEFAULT_RAW_DIR,
    out_dir: str = _DEFAULT_OUT_DIR,
    reviews_file: str = "Books_5.json.gz",
    meta_file: str = "meta_Books.json.gz",
    year_filter: int = 2017,
    min_user_interactions: int = 20,
    min_item_interactions: int = 20,
    label_threshold: int = 5,
    train_months: Optional[list] = None,
    valid_test_month: int = 12,
) -> tuple:
    """Build Amazon-Book sequential dataset.

    Mirrors ``build_ml1m`` in shape so the rest of the SigLLM pipeline
    (movie_ood dataset / Q-Former builder / Stage 1-3 trainers) works on
    books with no code changes downstream of the pickle files.

    Output sample (same columns as ML-1M):
    uid  iid  label  timestamp   his        his_title             title              genres                    flag  not_cold
    1    10   1      1496174400  [0]        [""]                  Some Book Title     Books|Fiction             -1    1
    1    25   0      1496347200  [0, 10]    ["", "Some Book ..."]  Another Title       Books|Mystery             -1    1

    Saved files in out_dir:
    - train_ood2.pkl: rows with flag = -1
    - valid_ood2.pkl: rows with flag = 0
    - test_ood2.pkl: rows with flag = 1
    - valid_small_ood2.pkl: 50% sample from valid_ood2.pkl
    - users_map.pkl / items_map.pkl: id remapping dictionaries

    Parameters
    ----------
    raw_dir
        Directory containing ``Books_5.json.gz`` and ``meta_Books.json.gz``
        from https://nijianmo.github.io/amazon/index.html.
    out_dir
        Output directory for ``*_ood2.pkl`` files.
    year_filter
        Single calendar year to restrict the dataset (default 2017,
        matches the ``processing_ood.ipynb`` reference).
    min_user_interactions / min_item_interactions
        Filter users/items with fewer interactions than the threshold
        within ``year_filter`` (default 20 each).
    label_threshold
        Binary label cutoff: ``rating >= label_threshold`` → positive.
        Default 5 matches ``processing_ood.ipynb`` (stricter than ML-1M).
    train_months / valid_test_month
        Temporal split inside ``year_filter``. Default: train on months
        1-11, validation+test on month 12 split 50/50 by timestamp.

    Returns
    -------
    (train_df, valid_df, test_df, users_map, items_map)
    """
    if train_months is None:
        train_months = list(range(1, valid_test_month))  # default: 1..11

    reviews_path = os.path.join(raw_dir, reviews_file)
    meta_path = os.path.join(raw_dir, meta_file)

    log_step(
        "[1/10] Load raw tables",
        f"reviews={reviews_path}, meta={meta_path}",
    )
    df = pd.DataFrame.from_dict({i: d for i, d in enumerate(_parse_gzip_json(reviews_path))}, orient="index")
    log_step("Raw reviews shape", str(df.shape))

    df_meta = pd.DataFrame.from_dict(list(_parse_gzip_json(meta_path)))
    log_step("Raw metadata shape", str(df_meta.shape))

    # Filter unformatted metadata rows (mojibake-like patterns flagged by the
    # presence of literal "getTime" strings in title — same as the reference
    # notebook). Strict equality on the empty string isn't enough.
    df_meta = df_meta.fillna("")
    bad_mask = df_meta["title"].astype(str).str.contains("getTime")
    df_meta = df_meta[~bad_mask].copy()
    log_step("Metadata after dropping malformed titles", str(df_meta.shape))

    log_step("[2/10] Project + rename columns", "ratings -> uid/iid/rating/timestamp")
    rating = df[["asin", "reviewerID", "overall", "unixReviewTime"]].copy()
    rating.columns = ["asin", "user", "rating", "timestamp"]

    meta_cols = [c for c in ("asin", "category", "title", "brand", "price") if c in df_meta.columns]
    meta_data = df_meta[meta_cols].copy()
    meta_data = meta_data.drop_duplicates(subset=["asin"], keep="last")
    log_step("Metadata projected", f"shape={meta_data.shape}, columns={meta_cols}")

    log_step("[3/10] Merge ratings + metadata", "right-join on asin")
    data = rating.merge(meta_data, on="asin", how="right")
    data = data.dropna()
    log_step("After merge + dropna", str(data.shape))

    rating_ = data.copy()
    # Match downstream column order: uid first, iid second (note: source notebook
    # uses ['iid','uid',...]; we keep ['uid','iid',...] for consistency with
    # ML-1M pipeline).
    new_cols = ["iid", "uid", "rating", "timestamp"]
    for c in ("category", "title", "brand", "price"):
        if c in rating_.columns:
            new_cols.append(c)
    rating_.columns = new_cols

    # Normalise category -> genres (pipe-joined string) so downstream code
    # that reads ``genres`` (e.g. MovieOODDataset and the Q-Former item-text
    # alignment builder) works without modification.
    if "category" in rating_.columns:
        rating_["genres"] = rating_["category"].apply(_category_to_genres)
    else:
        rating_["genres"] = ""

    log_step("[4/10] Restrict to single year", f"year={year_filter}")
    rating_["time"] = pd.to_datetime(rating_.timestamp, unit="s").map(lambda x: x.year)
    rating_ = rating_[rating_["time"] == year_filter].copy()
    log_step("After year filter", str(rating_.shape))

    # Now expand "time" from year -> month (1..12) so subsequent slicing uses
    # months as the temporal bucket within the chosen year.
    rating_["time"] = pd.to_datetime(rating_.timestamp, unit="s").map(lambda x: x.month)
    rating_ = rating_[rating_["time"].isin(range(1, 13))].copy()

    log_step(
        "[5/10] Filter active users/items",
        f"min_user={min_user_interactions}, min_item={min_item_interactions}",
    )
    user_info = rating_.groupby("uid").agg({"rating": "count"})
    item_info = rating_.groupby("iid").agg({"rating": "count"})
    active_users = user_info[user_info["rating"] > min_user_interactions].index
    active_items = item_info[item_info["rating"] > min_item_interactions].index
    log_step(
        "Active counts (after threshold)",
        f"users={active_users.shape[0]:,}, items={active_items.shape[0]:,}",
    )

    rating_ = rating_[rating_["uid"].isin(active_users)]
    rating_ = rating_[rating_["iid"].isin(active_items)]
    log_step("Post-filter shape", str(rating_.shape))

    rating_ = rating_.reset_index(drop=True)

    log_step("[6/10] Create binary label", f"rating >= {label_threshold} ⇒ positive=1")
    rating_["label"] = rating_["rating"].apply(lambda x: 1 if x >= label_threshold else 0)
    label_stats = rating_.label.describe()
    log_step(
        "Label stats",
        f"mean={label_stats['mean']:.3f}, std={label_stats['std']:.3f}, positives={label_stats['mean']*100:.1f}%",
    )

    # Remap raw Amazon IDs to compact ints starting at 1 (so 0 stays free as
    # the history-padding sentinel, matching MovieLens convention).
    users = rating_.uid.unique()
    items = rating_.iid.unique()
    users_map = dict(zip(users, np.arange(users.shape[0]) + 1))
    items_map = dict(zip(items, np.arange(items.shape[0]) + 1))
    users_map[0] = 0
    items_map[0] = 0
    rating_["uid"] = rating_["uid"].map(users_map)
    rating_["iid"] = rating_["iid"].map(items_map)
    log_step(
        "[7/10] Remap ids",
        f"users={len(users_map)-1:,}, items={len(items_map)-1:,}",
    )

    log_step(
        "[8/10] Define temporal split",
        f"train_months={train_months}, valid_test_month={valid_test_month}",
    )
    rating_train = rating_[rating_["time"].isin(train_months)].copy()
    rating_valid_test = rating_[rating_["time"] == valid_test_month].copy()
    rating_valid_test = rating_valid_test.sort_values(by="timestamp")
    n_half = rating_valid_test.shape[0] // 2
    rating_valid = rating_valid_test.iloc[:n_half].copy()
    rating_test = rating_valid_test.iloc[n_half:].copy()
    log_step(
        "Split sizes",
        f"train={rating_train.shape[0]:,}, valid={rating_valid.shape[0]:,}, test={rating_test.shape[0]:,}",
    )

    rating_valid_f = rating_valid
    rating_test_f = rating_test
    log_step(
        "Validation/Test label rate",
        f"valid={rating_valid_f.label.mean():.3f}, test={rating_test_f.label.mean():.3f}",
    )

    rating_train = rating_train.copy()
    rating_train["flag"] = pd.DataFrame(
        np.ones(rating_train.shape[0]) * -1, index=rating_train.index
    )
    rating_valid_f["flag"] = pd.DataFrame(
        np.zeros(rating_valid_f.shape[0]), index=rating_valid_f.index
    )
    rating_test_f["flag"] = pd.DataFrame(
        np.ones(rating_test_f.shape[0]), index=rating_test_f.index
    )
    log_step("Annotate split flag", "train=-1, valid=0, test=1")

    data = pd.concat([rating_train, rating_valid_f, rating_test_f], axis=0, ignore_index=True)
    data = data.sort_values(by=["uid", "timestamp"])
    log_step("Concatenate & sort", "order by (uid, timestamp)")

    u_inter_all = data.groupby("uid").agg(
        {"iid": list, "label": list, "title": list, "genres": list, "timestamp": list, "flag": list}
    )
    log_step("Grouped interactions", f"users={u_inter_all.shape[0]:,}")

    log_step("[9/10] Build sequential samples", "deal_with_each_u")
    results = []
    for u in u_inter_all.index:
        results.extend(deal_with_each_u(u_inter_all.loc[u], u))

    u_, i_, time_, label_, his_, his_title, title_, genres_, flag_ = [], [], [], [], [], [], [], [], []
    for re_ in results:
        u_.append(re_[0])
        i_.append(re_[1])
        time_.append(re_[2])
        his_.append(re_[3])
        his_title.append(re_[4])
        title_.append(re_[5])
        label_.append(re_[6])
        genres_.append(re_[7])
        flag_.append(re_[8])

    data = pd.DataFrame(
        {
            "uid": u_,
            "iid": i_,
            "label": label_,
            "timestamp": time_,
            "his": his_,
            "his_title": his_title,
            "title": title_,
            "genres": genres_,
            "flag": flag_,
        }
    )

    log_step("Sequential dataset", f"rows={data.shape[0]:,}")

    # History indices are already remapped because we remapped uid/iid BEFORE
    # the sequential build above. The lookup data["his"].apply(...) used in the
    # ML-1M version is a no-op here, but we keep the explicit map call to keep
    # the column dtype consistent (list of ints) regardless of how upstream
    # numpy handled it.
    data["his"] = data["his"].apply(lambda x: [int(k) for k in x])
    hist_lens = data["his"].apply(len)
    log_step(
        "History stats",
        f"avg={hist_lens.mean():.2f}, max={hist_lens.max()}, min={hist_lens.min()}",
    )

    final_stats = data.label.describe()
    log_step(
        "Final label stats",
        f"mean={final_stats['mean']:.3f}, count={int(final_stats['count']):,}",
    )

    train_ = data[data["flag"].isin([-1])].copy()
    valid_ = data[data["flag"].isin([0])].copy()
    test_ = data[data["flag"].isin([1])].copy()
    log_step(
        "Final dataset shapes",
        f"train={len(train_):,}, valid={len(valid_):,}, test={len(test_):,}",
    )

    train_user = set(train_["uid"].unique())
    train_item = set(train_["iid"].unique())
    valid_["not_cold"] = (
        valid_["uid"].isin(train_user) & valid_["iid"].isin(train_item)
    ).astype("int")
    test_["not_cold"] = (
        test_["uid"].isin(train_user) & test_["iid"].isin(train_item)
    ).astype("int")
    train_["not_cold"] = 1
    log_step(
        "Cold-start flags",
        f"valid warm={valid_['not_cold'].sum():,}, test warm={test_['not_cold'].sum():,}",
    )

    log_step("Train pickle columns", ", ".join(train_.columns))
    log_step("Train pickle sample", preview_row_for_log(train_))
    log_step("Valid pickle columns", ", ".join(valid_.columns))
    log_step("Valid pickle sample", preview_row_for_log(valid_))
    log_step("Test pickle columns", ", ".join(test_.columns))
    log_step("Test pickle sample", preview_row_for_log(test_))

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train_ood2.pkl")
    valid_path = os.path.join(out_dir, "valid_ood2.pkl")
    test_path = os.path.join(out_dir, "test_ood2.pkl")
    train_.to_pickle(train_path)
    valid_.to_pickle(valid_path)
    test_.to_pickle(test_path)

    valid_small = valid_.sample(frac=0.5, random_state=2023)
    valid_small_path = os.path.join(out_dir, "valid_small_ood2.pkl")
    valid_small.to_pickle(valid_small_path)

    users_map_path = os.path.join(out_dir, "users_map.pkl")
    items_map_path = os.path.join(out_dir, "items_map.pkl")
    with open(users_map_path, "wb") as f:
        pickle.dump(users_map, f)
    with open(items_map_path, "wb") as f:
        pickle.dump(items_map, f)

    log_step(
        "[10/10] Saved artifacts",
        f"train={train_path}, valid={valid_path}, test={test_path}, valid_small={valid_small_path}",
    )

    return train_, valid_, test_, users_map, items_map


if __name__ == "__main__":
    """Run Amazon-Book preprocessing pipeline as a standalone script."""

    if "src" not in sys.path:
        src_path = "src"
        if not sys.path[0].endswith("src"):
            sys.path.insert(0, src_path)

    train_df, valid_df, test_df, users_map, items_map = build_amazon_book()

    log_step(
        "✓ Preprocessing complete",
        f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}",
    )
