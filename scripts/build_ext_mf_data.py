"""Build an EXTENDED MF-pretraining file that adds the discarded pre-train
"burst" (months 0-13) to the current train window (14-23).

Why
---
The MF/CF teacher is currently trained on `train_ood2.pkl` = months 14-23 only
(3.4% of all interactions), which caps uAUC (MF-alone uAUC ~0.671). Months 0-13
are temporally BEFORE the train window, so using them as extra CF pretraining is
leakage-free. `scripts/diagnose_burst_overlap.py` confirmed the burst densifies
test users' CF history ~15x and rescues ~98% of cold test users.

What this produces
------------------
`train_ext_ood2.pkl` with columns [uid, iid, label], where:
  * rows  = every raw interaction in months 0..END_MONTH (default 23),
  * label = 1 if rating >= 4 else 0 (identical rule to data_preprocessing.py),
  * ids   = remapped through the EXISTING users_map/items_map so the id space
            (and therefore user_num/item_num) is byte-for-byte the pipeline's.

Rows whose user OR item is not in the maps (burst-only entities absent from the
14-33 eval space) are dropped — they can never be referenced at eval, so keeping
them would only inflate the id space. valid_ood2 / test_ood2 are NOT touched:
the only variable that changes is the MF training pool.

Usage
-----
    python scripts/build_ext_mf_data.py \
        --raw-dir /content/SigLLM/data/raw/ml-1m \
        --processed-dir /content/SigLLM/data/processed/ml-1m
    # then train MF on it:
    #   run.rec_baseline.train_file = train_ext_ood2.pkl   (see train_rec_baseline.py)
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

# End of the extended CF-pretraining window (inclusive). 23 = last train month;
# months 24-33 (valid/test) are deliberately excluded to avoid leakage.
DEFAULT_END_MONTH = 23


def load_and_bin(raw_dir):
    """Load raw ratings and reproduce the month-bin `time` column exactly as
    data_preprocessing.py:134-137."""
    rating = pd.read_csv(
        f"{raw_dir}/ratings.dat",
        header=None,
        sep="::",
        names=["uid", "iid", "rating", "timestamp"],
        engine="python",
    )
    dt = pd.to_datetime(rating.timestamp, unit="s")
    date_min = dt.min()
    rating["time"] = dt.map(lambda x: (x.year - date_min.year) * 12 + x.month)
    rating["time"] = rating["time"] - rating["time"].min()
    return rating


def main():
    ap = argparse.ArgumentParser(description="Build extended MF-pretraining pickle (months 0..END).")
    ap.add_argument("--raw-dir", default="/content/SigLLM/data/raw/ml-1m")
    ap.add_argument("--processed-dir", default="/content/SigLLM/data/processed/ml-1m")
    ap.add_argument("--end-month", type=int, default=DEFAULT_END_MONTH,
                    help="inclusive last month to include (default 23 = end of train window)")
    ap.add_argument("--out", default=None,
                    help="output path (default: <processed-dir>/train_ext_ood2.pkl)")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.processed_dir, "train_ext_ood2.pkl")

    with open(os.path.join(args.processed_dir, "users_map.pkl"), "rb") as f:
        users_map = pickle.load(f)
    with open(os.path.join(args.processed_dir, "items_map.pkl"), "rb") as f:
        items_map = pickle.load(f)

    rating = load_and_bin(args.raw_dir)
    ext = rating[rating.time <= args.end_month].copy()
    ext["label"] = (ext["rating"] >= 4).astype(int)

    # Remap into the pipeline's id space; drop entities not present in the maps.
    ext["uid"] = ext["uid"].map(users_map)
    ext["iid"] = ext["iid"].map(items_map)
    before = len(ext)
    ext = ext.dropna(subset=["uid", "iid"])
    dropped = before - len(ext)
    ext["uid"] = ext["uid"].astype(int)
    ext["iid"] = ext["iid"].astype(int)
    ext = ext[["uid", "iid", "label"]].reset_index(drop=True)

    ext.to_pickle(out_path)

    # Report vs the original train pool so the density gain is explicit.
    def summarize(df, name):
        print(f"  {name:18s} rows={len(df):>9,}  users={df.uid.nunique():>6,}  "
              f"items={df.iid.nunique():>6,}  pos_rate={df.label.mean():.3f}  "
              f"median_inter/user={df.groupby('uid').size().median():.1f}")

    print(f"Extended MF-pretraining file written: {out_path}")
    print(f"  window: months 0..{args.end_month}   dropped {dropped:,} rows (user/item not in id space)")
    summarize(ext, "train_ext (0-23)")
    try:
        orig = pd.read_pickle(os.path.join(args.processed_dir, "train_ood2.pkl"))[["uid", "iid", "label"]]
        summarize(orig, "train_ood2 (14-23)")
        print(f"  -> {len(ext) / max(len(orig), 1):.1f}x more interactions for MF")
    except Exception as e:  # noqa: BLE001
        print(f"  (could not load train_ood2.pkl for comparison: {e})")


if __name__ == "__main__":
    main()
