import copy
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sigllm.common import NotebookLogger

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RAW_DIR = os.environ.get("SIGLLM_RAW_DIR", str(_REPO_ROOT / "data" / "raw" / "ml-1m"))
_DEFAULT_OUT_DIR = os.environ.get("SIGLLM_PROCESSED_DIR", str(_REPO_ROOT / "data" / "processed" / "ml-1m"))

LOGGER = NotebookLogger.rich_logger("sigllm.data_prep")


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


def deal_with_each_u(x, u):
    """Build sequential interaction samples for a single user."""
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
            (u, items[i], timestamp[i], np.array(his), copy.copy(his_title), titles[i], labels[i], genres[i], flags[i])
        )
        if labels[i] > 0:
            his.append(items[i])
            his_title.append(titles[i])
    return results


def build_ml1m(
    raw_dir: str = _DEFAULT_RAW_DIR,
    out_dir: str = _DEFAULT_OUT_DIR,
    train_slot: list = None,
    valid_slot: list = None,
    test_slot: list = None,
) -> tuple:
    """
    Build ML-1M sequential dataset.

    Output sample:
    uid  iid  label   timestamp   his        his_title                  title                genres                    flag
    1    10   1       978300760   [0]        [""]                       Toy Story (1995)     Animation|Children        -1
    1    25   0       978302109   [0, 10]    ["", "Toy Story (1995)"]   Jumanji (1995)       Adventure|Children        -1
    1    33   1       978301968   [0, 10]    ["", "Toy Story (1995)"]   Grumpier Old Men     Comedy|Romance            0

    Columns: uid, iid, label, timestamp, his, his_title, title, genres, flag
    Saved files in out_dir:
    - train_ood2.pkl: rows with flag = -1
    - valid_ood2.pkl: rows with flag = 0
    - test_ood2.pkl: rows with flag = 1
    - valid_small_ood2.pkl: 50% sample from valid_ood2.pkl
    - users_map.pkl / items_map.pkl: id remapping dictionaries

    Returns: (train_df, valid_df, test_df, users_map, items_map)
    """
    if train_slot is None:
        train_slot = list(range(14, 24))
    if valid_slot is None:
        valid_slot = list(range(24, 29))
    if test_slot is None:
        test_slot = list(range(29, 34))
    
    log_step("[1/10] Load raw tables", "ratings + movies + users")
    rating = pd.read_csv(
        f"{raw_dir}/ratings.dat",
        header=None,
        sep="::",
        names=["uid", "iid", "rating", "timestamp"],
        engine="python",
    )
    item_info = pd.read_csv(
        f"{raw_dir}/movies.dat",
        sep="::",
        encoding="latin-1",
        engine="python",
        names=["iid", "title", "genres"],
    )
    user_info = pd.read_csv(
        f"{raw_dir}/users.dat",
        sep="::",
        encoding="latin-1",
        engine="python",
        names=["uid", "gender", "age", "occupation", "zipcode"],
        )
    log_step(
        "Raw shapes",
        f"ratings={rating.shape}, movies={item_info.shape}, users={user_info.shape}",
    )

    log_step("[2/10] Merge metadata", "content-aware enrichment (title/genres)")
    rating = pd.merge(rating, item_info, on="iid", how="inner")
    log_step("Merged table", f"shape={rating.shape}")

    log_step("[3/10] Inspect time span", "temporal coverage")
    date_min = pd.to_datetime(rating.timestamp, unit="s").min()
    date_max = pd.to_datetime(rating.timestamp, unit="s").max()
    log_step("Time extrema", f"{date_min.date()} -> {date_max.date()}")

    months_span = (date_max.year - date_min.year) * 12 + (date_max.month - date_min.month)
    log_step("Total months", str(months_span))

    log_step("[4/10] Build time slots", "prevent leakage via temporal split")
    rating["time"] = pd.to_datetime(rating.timestamp, unit="s").map(
        lambda x: (x.year - date_min.year) * 12 + x.month
    )
    rating["time"] = rating["time"] - rating["time"].min()

    time_bins = np.sort(rating.time.unique())
    log_step("Time bins", f"count={time_bins.shape[0]}, range=({time_bins.min()}, {time_bins.max()})")

    log_step("Plot time histogram", "volume vs slot for drift check")
    rating.groupby("time").agg({"rating": "count"}).reset_index().plot(x="time", kind="bar")

    log_step(f"[5/10] Define temporal windows", f"train={train_slot}, valid={valid_slot}, test={test_slot}")

    log_step("[6/10] Create binary label", "rating >= 4 ⇒ positive=1")
    # Example after merge and time-slot assignment:
    #   uid  iid  rating   timestamp   title                genres                    time
    #   1    10   5        978300760   Toy Story (1995)     Animation|Children        14
    #   1    25   3        978302109   Jumanji (1995)       Adventure|Children        14
    #   1    33   4        978301968   Grumpier Old Men     Comedy|Romance            24
    #   2    18   4        978824268   Heat (1995)          Action|Crime|Thriller     14
    #   2    41   5        978824291   Sabrina (1995)       Comedy|Romance            29
    #
    # After binary label transform (rating >= 4 -> 1 else 0):
    #   uid  iid  rating  label  time
    #   1    10   5       1      14
    #   1    25   3       0      14
    #   1    33   4       1      24
    #   2    18   4       1      14
    #   2    41   5       1      29
    rating["label"] = rating["rating"].apply(lambda x: 1 if x >= 4 else 0)

    label_stats = rating.label.describe()
    log_step(
        "Label stats",
        f"mean={label_stats['mean']:.3f}, std={label_stats['std']:.3f}, positives={label_stats['mean']*100:.1f}%",
    )

    uid_bounds = (rating.uid.min(), rating.uid.max())
    iid_bounds = (rating.iid.min(), rating.iid.max())
    log_step("ID ranges", f"uid={uid_bounds}, iid={iid_bounds}")

    rating_train = rating[rating["time"].isin(train_slot)].copy()
    rating_valid = rating[rating["time"].isin(valid_slot)].copy()
    rating_test = rating[rating["time"].isin(test_slot)].copy()
    # Example temporal split:
    # train_slot = [14, ..., 23], valid_slot = [24, ..., 28], test_slot = [29, ..., 33]
    # rating_train:
    #   uid  iid  label  time
    #   1    10   1      14
    #   1    25   0      14
    #   2    18   1      14
    # rating_valid:
    #   uid  iid  label  time
    #   1    33   1      24
    # rating_test:
    #   uid  iid  label  time
    #   2    41   1      29

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
    log_step("Validation columns", ", ".join(rating_valid_f.columns))

    rating_train = rating_train.copy()

    rating_train["flag"] = pd.DataFrame(np.ones(rating_train.shape[0]) * -1, index=rating_train.index)
    rating_valid_f["flag"] = pd.DataFrame(np.zeros(rating_valid_f.shape[0]), index=rating_valid_f.index)
    rating_test_f["flag"] = pd.DataFrame(np.ones(rating_test_f.shape[0]), index=rating_test_f.index)
    # Example split flag:
    # train rows -> flag = -1
    # valid rows -> flag = 0
    # test rows  -> flag = 1
    log_step("[7/10] Annotate split flag", "train=-1, valid=0, test=1")

    data = pd.concat([rating_train, rating_valid_f, rating_test_f], axis=0, ignore_index=True)
    data = data.sort_values(by=["uid", "timestamp"])
    # Example after concat + sort by (uid, timestamp):
    #   uid  iid  label   timestamp   flag
    #   1    10   1       978300760   -1
    #   1    25   0       978302109   -1
    #   1    33   1       978301968    0
    #   2    18   1       978824268   -1
    #   2    41   1       978824291    1
    log_step("[8/10] Concatenate & sort", "order by (uid, timestamp)")

    u_inter_all = data.groupby("uid").agg(
        {"iid": list, "label": list, "title": list, "genres": list, "timestamp": list, "flag": list}
    )
    # Example grouped interactions by user:
    # u_inter_all.loc[1] =
    #   iid        [10, 25, 33]
    #   label      [1, 0, 1]
    #   title      ["Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men"]
    #   genres     ["Animation|Children", "Adventure|Children", "Comedy|Romance"]
    #   timestamp  [978300760, 978302109, 978301968]
    #   flag       [-1, -1, 0]
    log_step("Grouped interactions", f"users={u_inter_all.shape[0]:,}")

    flag_values = data.flag.unique()
    log_step("Flag uniqueness", str(flag_values))

    log_step("[9/10] Build sequential samples", "deal_with_each_u")
    results = []
    for u in u_inter_all.index:
        results.extend(deal_with_each_u(u_inter_all.loc[u], u))
    # Example output from deal_with_each_u(u_inter_all.loc[1], 1):
    # [
    #   (1, 10, 978300760, [0], [""] , "Toy Story (1995)", 1, "Animation|Children", -1),
    #   (1, 25, 978302109, [0, 10], ["", "Toy Story (1995)"], "Jumanji (1995)", 0, "Adventure|Children", -1),
    #   (1, 33, 978301968, [0, 10], ["", "Toy Story (1995)"], "Grumpier Old Men", 1, "Comedy|Romance", 0),
    # ]
    # Note: history only grows when previous label == 1.

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
    # Example sequential dataframe before id remap:
    #   uid  iid  label   timestamp   his        his_title                  title                genres                    flag
    #   1    10   1       978300760   [0]        [""]                       Toy Story (1995)     Animation|Children        -1
    #   1    25   0       978302109   [0, 10]    ["", "Toy Story (1995)"]   Jumanji (1995)       Adventure|Children        -1
    #   1    33   1       978301968   [0, 10]    ["", "Toy Story (1995)"]   Grumpier Old Men     Comedy|Romance            0

    log_step("Sequential dataset", f"rows={data.shape[0]:,}")

    users = data.uid.unique()
    items = data.iid.unique()
    users_map = dict(zip(users, np.arange(users.shape[0]) + 1))
    items_map = dict(zip(items, np.arange(items.shape[0]) + 1))

    users_map[0] = 0
    items_map[0] = 0

    data["uid"] = data["uid"].map(users_map)
    data["iid"] = data["iid"].map(items_map)
    # Example id remap:
    # users_map = {1: 1, 2: 2, 0: 0}
    # items_map = {10: 1, 25: 2, 33: 3, 18: 4, 41: 5, 0: 0}
    # before: uid=1, iid=33
    # after : uid=1, iid=3
    log_step("[10/10] Remap ids", f"users={len(users_map)-1:,}, items={len(items_map)-1:,}")

    data["his"] = data["his"].apply(lambda x: [items_map[k] for k in x])
    # Example history remap:
    # before: his = [0, 10]
    # after : his = [0, 1]
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

    log_step("Next steps", "negative sampling + OOD tagging + serialize")

    # Split back to train/valid/test
    train_ = data[data["flag"].isin([-1])].copy()
    valid_ = data[data["flag"].isin([0])].copy()
    test_ = data[data["flag"].isin([1])].copy()
    # Example final split after sequential build:
    # train_:
    #   uid  iid  label  his       flag
    #   1    1    1      [0]       -1
    #   1    2    0      [0, 1]    -1
    # valid_:
    #   uid  iid  label  his       flag
    #   1    3    1      [0, 1]     0
    # test_:
    #   uid  iid  label  his       flag
    #   2    5    1      [0, 4]     1

    log_step("Final dataset shapes", f"train={len(train_):,}, valid={len(valid_):,}, test={len(test_):,}")

    # Cold-start annotations
    train_user = set(train_["uid"].unique())
    train_item = set(train_["iid"].unique())
    valid_["not_cold"] = (
        valid_["uid"].isin(train_user) & valid_["iid"].isin(train_item)
    ).astype("int")
    test_["not_cold"] = (
        test_["uid"].isin(train_user) & test_["iid"].isin(train_item)
    ).astype("int")
    train_["not_cold"] = 1
    # Example not_cold:
    # train_user = {1, 2}
    # train_item = {1, 2, 4}
    # valid row (uid=1, iid=3) -> not_cold = 0 because iid=3 not in train_item
    # test row  (uid=2, iid=5) -> not_cold = 0 because iid=5 not in train_item
    # train rows are always not_cold = 1
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
        "Saved artifacts",
        f"train={train_path}, valid={valid_path}, test={test_path}, valid_small={valid_small_path}",
    )

    return train_, valid_, test_, users_map, items_map


if __name__ == "__main__":
    """Run preprocessing pipeline as standalone script or on Colab."""
    
    # Ensure src is importable when running from repo root
    if "src" not in sys.path:
        src_path = "src"
        if not sys.path[0].endswith("src"):
            sys.path.insert(0, src_path)
    
    # Build dataset
    train_df, valid_df, test_df, users_map, items_map = build_ml1m()
    
    log_step("✓ Preprocessing complete", f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    print("\nDataset ready for training!")

