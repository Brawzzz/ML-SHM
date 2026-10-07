#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import json
import hashlib
import warnings

from dataclasses import dataclass

import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer

from .base import SHMDataset


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
@dataclass
class Splits:

    """
    Indices (into SHMDataset.X) of the train / validation / test sets.

    Built ONCE per (dataset, split config, seed) and saved to disk, so that every
    model compared on a dataset sees exactly the same data.

    train : healthy signals only (unsupervised training)
    val   : healthy (other blocks) + a fraction of the damaged signals
            -> early stopping, hyperparameter tuning, threshold selection
    test  : healthy + damaged signals never seen before -> final evaluation, used once
    """

    train : np.ndarray
    val   : np.ndarray
    test  : np.ndarray

    #================================================================================#
    def check(self, ds: SHMDataset) -> None:

        """Sanity checks : disjoint sets, no damaged signal in train, no healthy block shared between sets"""

        #---------------------------------------------
        (tr, va, te) = (set(self.train), set(self.val), set(self.test))

        if tr & va or tr & te or va & te:
            raise AssertionError("train / val / test indices overlap")

        if ds.y[self.train].any():
            raise AssertionError("damaged signals found in the training set")

        #------------------------------
        def healthy_blocks(idx):
            idx = np.asarray(idx, dtype=int)
            return set(ds.groups[idx][ds.y[idx] == 0])

        (b_tr, b_va, b_te) = (healthy_blocks(self.train), healthy_blocks(self.val), healthy_blocks(self.test))

        if (b_tr & b_va or b_tr & b_te or b_va & b_te) and not getattr(self, "_sample_level", False):
            raise AssertionError("an acquisition block is shared between two sets")

    #================================================================================#
    def describe(self, ds: SHMDataset) -> str:

        def count(idx):
            return f"{int((ds.y[idx] == 0).sum())} healthy / {int((ds.y[idx] == 1).sum())} damaged"

        return (f"split -> train : {count(self.train)} | val : {count(self.val)} | test : {count(self.test)}")

#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def make_splits(ds: SHMDataset,
                strategy: str = "group",
                val_size: float = 0.15,
                test_size: float = 0.15,
                damaged_val_size: float = 0.3,
                seed: int = 0) -> Splits:

    """
    Split a dataset into train / val / test.

    Healthy and damaged signals are split separately :
      - healthy : train (1 - val_size - test_size) / val (val_size) / test (test_size)
      - damaged : val (damaged_val_size) / test (1 - damaged_val_size), never in train

    Parameters
    ----------
    ds               : raw dataset
    strategy         : "random" -> sample level, random (optimistic : correlated neighbours on both sides)
                       "group"  -> whole acquisition blocks, randomly assigned
                       "chrono" -> whole acquisition blocks, oldest -> train, then val, then test
    val_size         : fraction of the healthy signals in the validation set
    test_size        : fraction of the healthy signals in the test set
    damaged_val_size : fraction of the damaged signals in the validation set (0 -> all in test)
    seed             : random seed

    Returns
    ----------
    Splits
    """

    #---------------------------------------------
    if strategy not in ("random", "group", "chrono"):
        raise ValueError(f"unknown split strategy : {strategy} (expected 'random', 'group' or 'chrono')")

    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError(f"invalid healthy fractions : val_size={val_size}, test_size={test_size}")

    if not 0 <= damaged_val_size < 1:
        raise ValueError(f"damaged_val_size must be in [0, 1) : {damaged_val_size}")

    rng = np.random.default_rng(seed)

    #---------------------------------------------
    idx_h = np.flatnonzero(ds.y == 0)
    idx_d = np.flatnonzero(ds.y == 1)

    (h_parts, h_sample) = _assign(idx_h, ds.groups, [1 - val_size - test_size, val_size, test_size], strategy, rng, "healthy")

    if len(idx_d):
        (d_parts, _) = _assign(idx_d, ds.groups, [0.0, damaged_val_size, 1 - damaged_val_size], strategy, rng, "damaged")
    else:
        warnings.warn("no damaged signal in the dataset : val / test contain healthy signals only")
        d_parts = [np.array([], dtype=int)] * 3

    #---------------------------------------------
    splits = Splits(train = np.sort(np.concatenate([h_parts[0], d_parts[0]])).astype(int),
                    val   = np.sort(np.concatenate([h_parts[1], d_parts[1]])).astype(int),
                    test  = np.sort(np.concatenate([h_parts[2], d_parts[2]])).astype(int))

    splits._sample_level = h_sample
    splits.check(ds)

    return splits

#================================================================================#
def _assign(idx: np.ndarray, groups: np.ndarray, fractions: list, strategy: str,
            rng: np.random.Generator, what: str) -> tuple[list, bool]:

    """
    Assign the indices `idx` to len(fractions) parts (train, val, test) according to `strategy`.

    Returns (parts, sample_level) where sample_level is True if a block-level split was
    impossible (not enough blocks) and a sample-level split was used instead.
    """

    #---------------------------------------------
    fractions = np.asarray(fractions, dtype=float)
    wanted    = fractions > 0
    n_parts   = len(fractions)

    blocks = np.unique(groups[idx])                      # sorted -> chronological order

    #------------------------------
    if strategy != "random" and len(blocks) < wanted.sum():

        warnings.warn(f"[{what}] only {len(blocks)} acquisition block(s) for {int(wanted.sum())} sets : "
                      f"falling back to a sample-level random split (results may be optimistic)")
        strategy = "random"

    #---------------------------------------------
    if strategy == "random":

        perm   = rng.permutation(idx)
        bounds = np.round(np.cumsum(fractions) / fractions.sum() * len(perm)).astype(int)
        parts  = np.split(perm, bounds[:-1])

        return ([np.sort(p) for p in parts], True)

    #---------------------------------------------
    if strategy == "group":
        blocks = rng.permutation(blocks)

    sizes = np.array([np.sum(groups[idx] == b) for b in blocks])

    #------------------------------ cut the ordered blocks to match the fractions in number of signals
    target   = np.cumsum(fractions) / fractions.sum() * sizes.sum()
    mid      = np.cumsum(sizes) - sizes / 2
    block_of = np.searchsorted(target, mid, side="right").clip(0, n_parts - 1)

    #------------------------------ every wanted set must receive at least one block
    if any(wanted[k] and not np.any(block_of == k) for k in range(n_parts)):

        counts   = _block_counts(len(blocks), fractions)
        block_of = np.repeat(np.arange(n_parts), counts)

    #---------------------------------------------
    parts = []
    for k in range(n_parts):
        chosen = blocks[block_of == k]
        parts.append(np.sort(idx[np.isin(groups[idx], chosen)]))

    return (parts, False)

#================================================================================#
def _block_counts(n_blocks: int, fractions: np.ndarray) -> np.ndarray:

    """Number of blocks per set : proportional to the fractions, at least 1 for every wanted set"""

    wanted = fractions > 0
    counts = np.where(wanted, 1, 0)
    rest   = n_blocks - counts.sum()

    if rest > 0:
        extra   = np.floor(fractions / fractions.sum() * rest).astype(int)
        counts += extra
        left    = rest - extra.sum()
        order   = np.argsort(-fractions)
        for k in order[:left]:
            counts[k] += 1

    return counts

#================================================================================#
def load_or_make_splits(ds: SHMDataset, split_cfg: dict, data_cfg: dict, seed: int, splits_dir: str) -> Splits:

    """
    Load the splits of this (dataset, data params, split config, seed) from `splits_dir`,
    or build and save them. The file name contains a fingerprint of everything the split
    depends on, so every model run on the same dataset reuses the very same split.
    """

    #---------------------------------------------
    key = json.dumps({"data"   : data_cfg,
                      "split"  : split_cfg,
                      "seed"   : seed,
                      "n"      : int(len(ds.y)),
                      "n_dam"  : int(ds.y.sum()),
                      "length" : int(ds.signal_length)}, sort_keys=True, default=str)

    fingerprint = hashlib.sha1(key.encode()).hexdigest()[:10]
    path        = os.path.join(splits_dir, f"{ds.name}_{split_cfg.get('strategy', 'group')}_{fingerprint}.npz")

    #---------------------------------------------
    if os.path.exists(path):

        f      = np.load(path)
        splits = Splits(train=f["train"], val=f["val"], test=f["test"])
        splits._sample_level = bool(f["sample_level"])
        splits.check(ds)

        print(f" -> split loaded : {path}")
        return splits

    #---------------------------------------------
    splits = make_splits(ds, seed=seed, **split_cfg)

    os.makedirs(splits_dir, exist_ok=True)
    np.savez(path, train=splits.train, val=splits.val, test=splits.test,
             sample_level=splits._sample_level, key=key)

    print(f" -> split saved  : {path}")
    return splits

#================================================================================#
def make_scaler(name: str):

    """
    Scaler fitted on the TRAINING signals only.

    minmax   : per time-sample min/max (previous behaviour)
    standard : per time-sample mean/std
    none     : identity
    """

    #---------------------------------------------
    if name == "minmax":
        return MinMaxScaler()

    if name == "standard":
        return StandardScaler()

    if name in ("none", None):
        return FunctionTransformer()

    raise ValueError(f"unknown scaler : {name} (expected 'minmax', 'standard' or 'none')")

#================================================================================#
def prepare(ds: SHMDataset, splits: Splits, scaler_name: str = "minmax") -> tuple[dict, object]:

    """
    Fit the scaler on the training set and apply it to the three sets.

    Returns
    ----------
    data   : {"train": (X, y), "val": (X, y), "test": (X, y)}  (float32 / int64)
    scaler : the fitted scaler (to save with the model)
    """

    #---------------------------------------------
    scaler = make_scaler(scaler_name)
    scaler.fit(ds.X[splits.train])

    data = {}
    for name in ("train", "val", "test"):
        idx        = getattr(splits, name)
        data[name] = (scaler.transform(ds.X[idx]).astype(np.float32), ds.y[idx])

    return (data, scaler)
