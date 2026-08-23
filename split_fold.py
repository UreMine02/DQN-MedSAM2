#!/usr/bin/env python
"""Pre-compute the grouped train/val fold assignment and write it to ./data/splits.

Folds used to be drawn inside every ``Dataset.__init__`` from ``-split_seed``. That is
reproducible only as long as nobody touches the manifests: adding one volume to a
manifest reshuffles every fold, so a number measured last month stops being comparable
with one measured today, silently. Materialising the assignment once turns it into a
committed artefact -- a run cites a CSV, and a diff on that CSV is the only thing that
can move a patient between folds.

Run once per (seed, n_folds) pair::

    python split_fold.py --n-folds 5 --seed 0

Only the ``*Tr.csv`` manifests are folded. The ``*Ts.csv`` test manifests are already a
fixed hold-out and are never part of the train/val rotation.

The split itself is a plain random k-fold. It needs no class stratification: each
manifest holds exactly one task, and within a task every scan carries the same label
set, so dealing scans out at random already gives every fold the same class mix.

The one thing it is not free to do is split on manifest *rows*. A row is a
``(volume, obj_id)`` pair, so the unit is ``group_key``: a BTCV volume's 13 organ rows
-- and both of a Sarcoma patient's structure files, ``STS_001_Mass`` and
``STS_001_Edema`` -- have to land on the same side or the same anatomy trains and
validates.
"""

import argparse
import glob
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

# Importing func_3d.dataset would run its package __init__, which reaches func_3d.utils
# and calls cfg.parse_args() at import time -- that parser would reject this script's own
# flags before argparse below ever sees them. splits.py has no intra-package imports, so
# loading it straight off disk gets the shared helpers without the side effect.
_SPLITS_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "func_3d", "dataset", "splits.py")
_spec = importlib.util.spec_from_file_location("_splits", _SPLITS_PY)
_splits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_splits)
fold_csv_path, group_key = _splits.fold_csv_path, _splits.group_key


def read_train_manifests(data_root):
    """Concat every ``*Tr.csv`` under ``data_root``. One manifest, one task."""
    paths = sorted(glob.glob(os.path.join(data_root, "**", "*Tr.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no *Tr.csv manifests under {data_root}")

    print(f"read {len(paths)} training manifests from {data_root}")
    return pd.concat([pd.read_csv(p, index_col=0) for p in paths], ignore_index=True)


def collapse_to_groups(train_df):
    """One row per group, tagged with its task and how many manifest rows it owns."""
    grouped = train_df.assign(group=group_key(train_df["gt_path"]).values).groupby("group")

    groups = pd.DataFrame({
        "task": grouped["task"].agg(lambda s: "|".join(sorted(set(s)))),
        "n_rows": grouped.size(),
    }).reset_index()

    # Every manifest holds a single task, so a group spanning two of them means group_key
    # collided across datasets -- and the whole split is then keyed on a lie. Catch it
    # here rather than letting it surface as a mysterious leak.
    straddling = groups[groups["task"].str.contains("|", regex=False)]
    if not straddling.empty:
        raise ValueError(
            "these groups span more than one task, so group_key is not unique per dataset:\n"
            + straddling[["group", "task"]].to_string(index=False)
        )

    return groups


def assign_folds(groups, n_folds, seed):
    """Deal each task's shuffled groups round-robin over the folds; returns the fold column.

    Per task, because a fold that held only Livers would be useless as a val split for a
    run training on Hearts. Within a task no further stratification is needed: every scan
    there carries the same label set, so a uniform shuffle already balances the classes.
    """
    rng = np.random.default_rng(seed)
    fold = np.empty(len(groups), dtype=int)

    for task in sorted(groups["task"].unique()):
        # Sort before shuffling: the assignment must depend on the seed alone, never on
        # the order rows happened to appear in the manifest.
        members = np.sort(groups.index[groups["task"] == task].to_numpy())
        shuffled = rng.permutation(members)
        # When a task's group count is not a multiple of n_folds the remainder has to go
        # somewhere; a random starting fold stops it landing on fold 0 for every task at
        # once, which would leave fold 0 systematically the largest.
        start = int(rng.integers(n_folds))
        fold[shuffled] = (np.arange(len(shuffled)) + start) % n_folds

        if len(members) < n_folds:
            print(
                f"  WARNING: task {task!r} has only {len(members)} group(s) for "
                f"{n_folds} folds -- {n_folds - len(members)} fold(s) hold none of it"
            )

    return fold


def report(assignment, train_df, n_folds):
    """Print groups-per-fold per task, plus any class a fold cannot supply a support for."""
    print("\ngroups per fold")
    table = assignment.pivot_table(
        index="task", columns="fold", values="group", aggfunc="count", fill_value=0
    ).reindex(columns=range(n_folds), fill_value=0)
    table["total"] = table.sum(axis=1)
    print(table.to_string())

    print("\nmanifest rows per fold")
    rows = assignment.pivot_table(
        index="task", columns="fold", values="n_rows", aggfunc="sum", fill_value=0
    ).reindex(columns=range(n_folds), fill_value=0)
    rows["total"] = rows.sum(axis=1)
    print(rows.to_string())

    # Each fold is used as a val split exactly once, and the other n_folds-1 folds are
    # then both the train queries and the support pool. A class that survives in only one
    # fold leaves the run that holds that fold out with no support volume at all.
    per_row = train_df.assign(group=group_key(train_df["gt_path"]).values).merge(
        assignment[["group", "fold"]], on="group", how="left"
    )
    coverage = per_row.groupby(["task", "obj_id"])["fold"].nunique()
    thin = coverage[coverage < 2]
    if not thin.empty:
        print("\nWARNING: these classes live in fewer than 2 folds, so holding one fold out")
        print("leaves nothing to draw a support from -- lower --n-folds or check the manifest:")
        for (task, obj_id), n in thin.items():
            print(f"  {task}/obj_id={obj_id}: present in {n} fold(s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="./data", help="directory holding the manifest CSVs")
    parser.add_argument("--n-folds", type=int, default=5, help="number of grouped folds to deal the training manifests into")
    parser.add_argument("--seed", type=int, default=0, help="seed for the shuffle; must match the -split_seed a run passes")
    parser.add_argument("--out", default="", help="output CSV path (default: <data-root>/splits/folds_seed<seed>_k<n_folds>.csv)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing assignment")
    args = parser.parse_args()

    if args.n_folds < 2:
        parser.error(f"--n-folds must be at least 2, got {args.n_folds}")

    out_path = args.out or fold_csv_path("", args.seed, args.n_folds, data_root=args.data_root)
    if os.path.exists(out_path) and not args.force:
        # Silently rewriting this file would move patients between folds under runs that
        # already cited it, so the overwrite has to be asked for.
        sys.exit(
            f"{out_path} already exists. Runs citing it would change meaning if it moved -- "
            f"pass --force to overwrite, or --out to write a new file."
        )

    train_df = read_train_manifests(args.data_root)
    groups = collapse_to_groups(train_df)

    print(f"{len(train_df)} manifest rows over {len(groups)} groups, "
          f"{groups['task'].nunique()} tasks")

    groups["fold"] = assign_folds(groups, args.n_folds, args.seed)
    groups["n_folds"] = args.n_folds
    groups["seed"] = args.seed

    assignment = groups.sort_values(["task", "group"]).reset_index(drop=True)
    report(assignment, train_df, args.n_folds)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    assignment[["group", "task", "n_rows", "fold", "n_folds", "seed"]].to_csv(out_path, index=False)
    print(f"\nwrote {len(assignment)} group assignments to {out_path}")
    print("commit this file -- it is what makes two runs' folds comparable")


if __name__ == "__main__":
    main()
