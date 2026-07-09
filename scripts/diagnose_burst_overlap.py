"""Diagnose whether the discarded pre-train "burst" (months 0-13) would raise
the CF ceiling if added as extra MF pretraining data.

Background
----------
`data_preprocessing.build_ml1m` keeps only months 14-33 for the OOD split
(train=14-23, valid=24-28, test=29-33) and THROWS AWAY months 0-13 — the huge
collection burst visible in the ratings-over-time histogram. The MF teacher is
therefore trained on a sparse slice, which caps uAUC (MF-alone uAUC ~0.671).

Adding months 0-13 as *extra CF pretraining* is temporally valid (0-13 precedes
the train window 14-23 -> no leakage). But it only helps if the test users/items
actually appear in that burst. This script measures that overlap so you know,
BEFORE retraining anything, whether the burst can rescue cold test entities.

It reconstructs the exact time-binning from raw ratings.dat (matching
data_preprocessing.py) so no processed files are required. `cold` follows the
strict definition in preprocess_test_cold_warm.py: uid not in train users AND
iid not in train items. `warm` threshold: interaction count > min_*_inter.

Usage
-----
    python scripts/diagnose_burst_overlap.py \
        --raw-dir /content/SigLLM/data/raw/ml-1m \
        [--processed-dir /content/SigLLM/data/processed/ml-1m]  # optional sanity check
"""

import argparse

import numpy as np
import pandas as pd


# Windows must match data_preprocessing.build_ml1m defaults.
HIST_SLOT = list(range(0, 14))    # discarded burst (pre-train history)
TRAIN_SLOT = list(range(14, 24))
VALID_SLOT = list(range(24, 29))
TEST_SLOT = list(range(29, 34))


def load_and_bin(raw_dir):
    """Load raw ratings and reproduce the month-bin `time` column exactly as
    data_preprocessing.py does."""
    rating = pd.read_csv(
        f"{raw_dir}/ratings.dat",
        header=None,
        sep="::",
        names=["uid", "iid", "rating", "timestamp"],
        engine="python",
    )
    dt = pd.to_datetime(rating.timestamp, unit="s")
    date_min = dt.min()
    # (year - year_min)*12 + month, then rebased to start at 0 — identical to
    # data_preprocessing.py:134-137.
    rating["time"] = dt.map(lambda x: (x.year - date_min.year) * 12 + x.month)
    rating["time"] = rating["time"] - rating["time"].min()
    return rating


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def section(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main():
    ap = argparse.ArgumentParser(description="Measure burst<->test overlap for CF pretraining.")
    ap.add_argument("--raw-dir", default="/content/SigLLM/data/raw/ml-1m")
    ap.add_argument("--processed-dir", default=None,
                    help="optional; if given, sanity-checks test row count against test_ood2.pkl")
    ap.add_argument("--min-user-inter", type=int, default=3)
    ap.add_argument("--min-item-inter", type=int, default=3)
    args = ap.parse_args()

    rating = load_and_bin(args.raw_dir)

    hist = rating[rating.time.isin(HIST_SLOT)]
    train = rating[rating.time.isin(TRAIN_SLOT)]
    test = rating[rating.time.isin(TEST_SLOT)]

    section("Window sizes (interactions)")
    total = len(rating)
    for name, df, slot in [
        ("HIST (discarded 0-13)", hist, HIST_SLOT),
        ("TRAIN (14-23)", train, TRAIN_SLOT),
        ("VALID (24-28)", rating[rating.time.isin(VALID_SLOT)], VALID_SLOT),
        ("TEST (29-33)", test, TEST_SLOT),
    ]:
        print(f"  {name:24s} {len(df):>9,}  ({pct(len(df), total)} of all)")
    print(f"  {'TOTAL':24s} {total:>9,}")
    print(f"  -> burst is {len(hist) / max(len(train), 1):.1f}x the current train pool")

    if args.processed_dir:
        try:
            t_pkl = pd.read_pickle(f"{args.processed_dir}/test_ood2.pkl")
            print(f"  [sanity] reconstructed TEST rows={len(test):,} vs test_ood2.pkl rows={len(t_pkl):,}")
        except Exception as e:  # noqa: BLE001
            print(f"  [sanity] could not load test_ood2.pkl: {e}")

    # Interaction counts per user/item in current vs extended CF pool.
    train_u = train.groupby("uid").size()
    train_i = train.groupby("iid").size()
    ext = pd.concat([hist, train])                 # months 0-23 (no leakage into test)
    ext_u = ext.groupby("uid").size()
    ext_i = ext.groupby("iid").size()

    train_users, train_items = set(train_u.index), set(train_i.index)
    hist_users, hist_items = set(hist.uid.unique()), set(hist.iid.unique())

    test_users = set(test.uid.unique())
    test_items = set(test.iid.unique())

    section("TEST users — does the burst rescue the cold ones?")
    warm_now = {u for u in test_users if u in train_users}
    cold_now = test_users - warm_now
    rescued = {u for u in cold_now if u in hist_users}
    print(f"  test users total ............. {len(test_users):>6,}")
    print(f"  in TRAIN pool (have CF hist) . {len(warm_now):>6,}  ({pct(len(warm_now), len(test_users))})")
    print(f"  cold now (no train hist) ..... {len(cold_now):>6,}  ({pct(len(cold_now), len(test_users))})")
    print(f"    -> of those, in BURST ...... {len(rescued):>6,}  ({pct(len(rescued), len(cold_now))} of cold)  <-- rescuable")
    print(f"    -> still unseen even w/ burst {len(cold_now - rescued):>6,}  (irreducible cold)")

    section("TEST items — same view")
    iwarm_now = {i for i in test_items if i in train_items}
    icold_now = test_items - iwarm_now
    irescued = {i for i in icold_now if i in hist_items}
    print(f"  test items total ............. {len(test_items):>6,}")
    print(f"  in TRAIN pool ................ {len(iwarm_now):>6,}  ({pct(len(iwarm_now), len(test_items))})")
    print(f"  cold now ..................... {len(icold_now):>6,}  ({pct(len(icold_now), len(test_items))})")
    print(f"    -> of those, in BURST ...... {len(irescued):>6,}  ({pct(len(irescued), len(icold_now))} of cold)  <-- rescuable")
    print(f"    -> still unseen even w/ burst {len(icold_now - irescued):>6,}  (irreducible cold)")

    section("CF density for TEST users (interaction count in pool)")
    def dens(counts, users):
        vals = np.array([counts.get(u, 0) for u in users])
        warm = int((vals > args.min_user_inter).sum())
        return vals, warm
    cur_vals, cur_warm = dens(train_u.to_dict(), test_users)
    ext_vals, ext_warm = dens(ext_u.to_dict(), test_users)
    print(f"  metric                        current(14-23)   extended(0-23)")
    print(f"  median interactions/user ...  {np.median(cur_vals):>12.1f}   {np.median(ext_vals):>13.1f}")
    print(f"  mean   interactions/user ...  {np.mean(cur_vals):>12.1f}   {np.mean(ext_vals):>13.1f}")
    print(f"  users above warm thr (>{args.min_user_inter}) .  {cur_warm:>12,}   {ext_warm:>13,}"
          f"   (+{ext_warm - cur_warm})")

    section("Strict-cold TEST interactions (uid AND iid unseen in train)")
    tu = test.uid.values
    ti = test.iid.values
    cold_mask = np.array([(u not in train_users) and (i not in train_items) for u, i in zip(tu, ti)])
    ext_users_all = set(ext_u.index)
    ext_items_all = set(ext_i.index)
    still_cold_mask = np.array([(u not in ext_users_all) and (i not in ext_items_all)
                                for u, i in zip(tu, ti)])
    n_cold = int(cold_mask.sum())
    n_still = int(still_cold_mask.sum())
    print(f"  strict-cold interactions now ......... {n_cold:>7,}  ({pct(n_cold, len(test))} of test)")
    print(f"  still strict-cold after adding burst . {n_still:>7,}  ({pct(n_still, len(test))} of test)")
    print(f"  -> rescued by burst .................. {n_cold - n_still:>7,}")

    section("VERDICT")
    user_gain = pct(len(rescued), len(cold_now)) if cold_now else "0%"
    warm_gain = ext_warm - cur_warm
    print(f"  Cold test users rescuable by burst: {len(rescued)} ({user_gain} of cold users)")
    print(f"  Test users crossing warm threshold: +{warm_gain}")
    print(f"  Strict-cold interactions rescued:   {n_cold - n_still}")
    if len(rescued) > 0.15 * max(len(cold_now), 1) or warm_gain > 0.10 * max(len(test_users), 1):
        print("  => WORTH TRYING: burst adds real CF history to a meaningful slice of test.")
    else:
        print("  => LIKELY LOW PAYOFF: test entities are genuinely unseen in the burst;")
        print("     the OOD ceiling is intrinsic, not a data-volume artifact.")


if __name__ == "__main__":
    main()
