"""Grouped train/val/test splitting for the manifest CSVs under ./data.

Three things make a naive split wrong here:

1. A manifest row is a ``(volume, obj_id)`` pair, so one patient owns several
   rows -- 13 for a BTCV volume, and for Sarcoma two *files* (``STS_001_Mass``
   and ``STS_001_Edema``). Splitting on rows or filenames leaks the same
   anatomy into both sides, so everything here splits on ``group_key``.

2. Every query conditions on an in-context support volume. If the val split
   drew its supports from itself, the reported number would depend on labels
   the model is being scored against. ``resolve_split`` therefore always hands
   back the *train* fold as the support pool, whatever the query split is.

3. Which group lands in which fold is *read* from a committed CSV, not redrawn
   from the seed here. A seeded reshuffle at load time is only reproducible
   while the manifests are frozen -- adding one volume silently rewrites every
   fold and quietly breaks comparability with earlier runs. Generate the CSV
   once with ``python split_fold.py``; ``-fold_csv`` points at a different one.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

# Sarcoma stores one file per structure, so the patient id is the group, not the
# file. Every other dataset is one file per volume and the path is already unique.
_SARCOMA_PATIENT = re.compile(r"^(?P<pid>.*STS_\d+)_")


def group_key(gt_path):
    """Map a Series of gt_paths to the identifier that must not straddle splits."""

    def key(path):
        match = _SARCOMA_PATIENT.match(path)
        return match.group("pid") if match else path

    return pd.Series(gt_path, dtype=object).map(key)


def fold_csv_path(fold_csv, split_seed, n_folds, data_root="./data"):
    """Where ``split_fold.py`` puts the assignment for one (seed, n_folds) pair.

    An explicit ``-fold_csv`` wins, so a run can cite an assignment kept anywhere.
    """
    if fold_csv:
        return fold_csv
    return os.path.join(data_root, "splits", f"folds_seed{split_seed}_k{n_folds}.csv")


def read_fold_assignment(path, n_folds):
    """Load ``split_fold.py``'s output as a group -> fold Series, checking it fits ``n_folds``."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no fold assignment at {path}. Generate it once with "
            f"`python split_fold.py --n-folds {n_folds}` (add --seed to match -split_seed), "
            f"then commit it so every run splits the same way."
        )

    df = pd.read_csv(path)

    duplicated = df["group"][df["group"].duplicated()].unique()
    if duplicated.size:
        raise ValueError(f"{path} assigns these groups more than one fold: {list(duplicated)}")

    stored = sorted(df["n_folds"].unique())
    if stored != [n_folds]:
        raise ValueError(
            f"{path} was built for n_folds={stored}, but this run passed -n_folds {n_folds}. "
            f"Regenerate it with `python split_fold.py --n-folds {n_folds}`."
        )

    out_of_range = df["fold"][(df["fold"] < 0) | (df["fold"] >= n_folds)].unique()
    if out_of_range.size:
        raise ValueError(f"{path} contains fold ids outside [0, {n_folds}): {list(out_of_range)}")

    return df.set_index("group")["fold"]


def resolve_split(train_df, test_df, mode, args):
    """Return ``(query_df, support_df)`` for ``mode`` in {'train', 'val', 'test'}.

    ``args.fold`` selects which of the ``args.n_folds`` grouped folds recorded in the
    fold-assignment CSV is held out for validation; ``args.fold < 0`` disables the split
    entirely (train on all of ``train_df``, no val split -- there is then nothing to
    select a best checkpoint on) and does not need the CSV at all.

    ``test_df`` may be a zero-argument callable, and is then only called for
    ``mode == 'test'``. Training builds only the 'train' and 'val' datasets, so with a
    callable the ``*Ts.csv`` manifests are not so much as opened during a training run.
    """
    assert mode in ("train", "val", "test"), f"unknown mode {mode!r}"

    if args.fold < 0:
        train_query, val_query = train_df, None
    else:
        assert 0 <= args.fold < args.n_folds, f"fold {args.fold} out of range for n_folds {args.n_folds}"
        path = fold_csv_path(args.fold_csv, args.split_seed, args.n_folds)
        folds = read_fold_assignment(path, args.n_folds)

        groups = group_key(train_df["gt_path"])
        # A group the CSV has never heard of means the manifests grew since the split was
        # generated. Dropping it into train by default would make the fold silently depend
        # on when the run happened, which is the whole thing this file exists to prevent.
        missing = sorted(set(groups) - set(folds.index))
        if missing:
            raise ValueError(
                f"{len(missing)} group(s) in the training manifest are absent from {path}, "
                f"e.g. {missing[:3]}. The manifests changed after the split was generated -- "
                f"rerun `python split_fold.py --force` and treat the folds as new."
            )

        is_val = (train_df["fold"] == args.fold)
        train_query, val_query = train_df[~is_val], train_df[is_val]

    # print(train_query, val_query)
    # The support pool is the labelled data the model was trained on -- never the
    # val or test split.
    support_df = train_query

    if mode == "train":
        query_df = train_query
    elif mode == "val":
        if val_query is None:
            raise ValueError("mode='val' requires -fold >= 0; got -fold < 0 (no val split)")
        query_df = val_query
    else:
        query_df = test_df() if callable(test_df) else test_df

    return query_df.reset_index(drop=True), support_df.reset_index(drop=True)


def check_support_coverage(query_df, support_df, num_support, mode, label=""):
    """Fail loudly at construction if some query class has no support to draw from.

    ``__getitem__`` samples a support volume of the same ``(task, obj_id)`` and at
    least ``num_support`` positive slices, excluding the query's own group. A split
    that starves one class would otherwise blow up mid-epoch inside a worker.
    """
    eligible = support_df[support_df["n_pos"] >= num_support].copy()
    eligible["_group"] = group_key(eligible["gt_path"]).values
    available = eligible.groupby(["task", "obj_id"])["_group"].nunique()

    # In train mode query and support are the same frame, so a class needs a second
    # group to fall back on once the query's own group is excluded.
    needed = 2 if mode == "train" else 1

    missing = []
    for task, obj_id in query_df[["task", "obj_id"]].drop_duplicates().itertuples(index=False):
        if available.get((task, obj_id), 0) < needed:
            missing.append(f"{task}/obj_id={obj_id} ({available.get((task, obj_id), 0)} of {needed})")

    if missing:
        raise ValueError(
            f"{label or mode} split has no usable support volumes for: {', '.join(missing)}. "
            f"Lower -num_support (currently {num_support}), lower -n_folds, or pick another -fold."
        )


def sample_support_index(ds, index, task, obj_id):
    """Pick a row in ``ds``'s support pool to use as the in-context example for query ``index``.

    Expects the ``sup_*``/``group``/``mode``/``num_support``/``split_seed`` attributes
    that the dataset ``__init__``s set up. The query's own group is excluded so a
    volume never supports itself, and for Sarcoma neither does the same patient's
    other structure file.
    """
    eligible = (
        (ds.sup_task == task)
        & (ds.sup_obj_id == obj_id)
        & (ds.sup_n_pos >= ds.num_support)
    )
    candidates = np.flatnonzero(eligible)
    if candidates.size == 0:
        raise RuntimeError(
            f"no support volume for query {ds.gt_path[index]} (task={task}, obj_id={obj_id}, "
            f"num_support={ds.num_support}) in the {ds.mode} split's support pool"
        )

    if ds.mode == "train":
        return int(np.random.choice(candidates))

    # Fixed per query: a fresh draw each epoch would make val/test Dice move with the
    # support choice rather than the model, and best-checkpoint selection would latch
    # onto whichever epoch happened to get an easy support.
    rng = np.random.default_rng(ds.split_seed * 100003 + index)
    return int(candidates[rng.integers(candidates.size)])


def read_manifest(csv_root, subset, task_prefix=None, recursive=False):
    """Load and concat the manifest CSVs for one subset ('Tr' or 'Ts')."""
    pattern = os.path.join(csv_root, "**", f"*{subset}.csv") if recursive else os.path.join(csv_root, f"*{subset}.csv")
    paths = sorted(glob.glob(pattern, recursive=recursive))
    if task_prefix:
        paths = [p for p in paths if os.path.basename(p).startswith(task_prefix)]

    if not paths:
        raise FileNotFoundError(
            f"no *{subset}.csv manifests in {csv_root}" + (f" for task {task_prefix!r}" if task_prefix else "")
        )

    return pd.concat([pd.read_csv(p, index_col=0) for p in paths], ignore_index=True)
