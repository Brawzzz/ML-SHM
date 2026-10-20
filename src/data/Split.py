#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import itertools
import numpy as np

from dataclasses import dataclass
from sklearn.model_selection import GroupShuffleSplit

from .GWDataset import GWDataset


#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
SPLIT_METHODS = ("group", "random")


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
@dataclass
class Splits:

    """
    This class splits the loaded data in : Train | Validation | Test set

    It does not contain the raw signals - only the row indices of the data file to minimize the used memory 

    Attributes
    ----------

    train        : indices of healthy signals only
    validation   : indices of healthy + damaged signals
    test         : indices of healthy + damaged signals never seen before
    method       : "group"  -> the signals of a block (ds.groups) are always in the same set
                   "random" -> each signal is assigned independently (blocks can be shared)
    """

    #---------------------------------------------
    train        : np.ndarray
    validation   : np.ndarray
    test         : np.ndarray
    method       : str = "group"

    #================================================================================#
    def check(self, ds : GWDataset) -> None:

        """
        Check the sanity of a splits : 

            1 - only healthy signals in train
            2 - no overlap between train | validation | test sets
            3 - no block shared between sets (only for method = "group")
            
        Args
        ----------
        ds : loaded guided wave dataset
        """

        #---------------------------------------------
        if ds.y[self.train].any():
            raise AssertionError("train set contains damaged signals")

        train_set       = set(self.train)
        validation_set  = set(self.validation)
        test_set        = set(self.test)

        if (train_set & validation_set) or (train_set & test_set) or (validation_set & test_set):
            raise AssertionError("train | val | test indices overlap")

        if self.method == "random":
            return

        #---------------------------------------------
        sets = {"train" : self.train, "validation" : self.validation, "test" : self.test}

        for (label, class_name) in ((0, "healthy"), (1, "damaged")):

            blocks = {name : set(ds.groups[idx[ds.y[idx] == label]]) for (name, idx) in sets.items()}

            for (a, b) in itertools.combinations(blocks, 2):
                shared = blocks[a] & blocks[b]
                if shared:
                    raise AssertionError(f"{class_name} blocks shared between {a} and {b} : {sorted(shared)}")

    #================================================================================#
    def describe(self, ds: GWDataset) -> str:

        """
        This method print and return a string that describe a Split e.g.: 
        
        Example:
        ----------
        //-------------------- Split of UTAH dataset --------------------//
        train      : healthy  16030 ( 80.1 %) | damaged      0 (  0.0 %) | blocks   324
        validation : healthy   1976 (  9.9 %) | damaged  27451 ( 69.9 %) | blocks   594
        test       : healthy   2016 ( 10.1 %) | damaged  11810 ( 30.1 %) | blocks   279

        Args
        ----------
        ds : loaded guided wave dataset

        Return:
        ----------
        desc : (str) description of a Split
        """

        #---------------------------------------------
        lines = []
        lines.append(f"//------------------- Split of {ds.name} dataset ({self.method}) -------------------//")

        for name, idx in (("train", self.train), ("validation", self.validation), ("test", self.test)):

            n_healthy   = _count(ds, idx, 0)
            n_damaged   = _count(ds, idx, 1)
            pct_healthy = 100 * n_healthy / max((ds.y == 0).sum(), 1)
            pct_damaged = 100 * n_damaged / max((ds.y == 1).sum(), 1)
            n_blocks    = len(np.unique(ds.groups[idx]))

            lines.append(f"{name:<10} : healthy {n_healthy:>6} ({pct_healthy:5.1f} %) | "
                         f"damaged {n_damaged:>6} ({pct_damaged:5.1f} %) | "
                         f"blocks {n_blocks:>5}")

        #-------------------------
        blocks_per_set = [set(ds.groups[idx]) for idx in (self.train, self.validation, self.test)]
        all_blocks     = set.union(*blocks_per_set)
        n_shared       = sum(1 for b in all_blocks if sum(b in s for s in blocks_per_set) > 1)

        lines.append(f"\nshared blocks : {n_shared} / {len(all_blocks)}")

        #-------------------------
        desc = "\n".join(lines) + "\n"

        return desc

    #================================================================================#
    def save(self, 
             path : str, 
             ds   : GWDataset) -> None:

        """
        Save a split in `path` 

        Args
        ----------
        path : (str) file path where to save the Split 
        ds   : (GWDataset) data stroed in a GWDataset object

        Returns
        ----------
        None
        """

        #---------------------------------------------
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        np.savez(path,
                 train      = self.train,
                 validation = self.validation,
                 test       = self.test,
                 method     = self.method,
                 y          = ds.y,         
                 groups     = ds.groups)

#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def _count(ds : GWDataset, idx, value):
    return int((ds.y[idx] == value).sum())

#================================================================================#
def load_splits(path : str, ds : GWDataset) -> Splits:

    """
    Load a split from a file 

    Args
    ----------
    path : (str) path to the saved split
    ds   : (GWDataset) splitted dataset

    Return
    ----------
    splits : (Splits) loaded split
    
    """

    #---------------------------------------------
    split_file = np.load(path)

    same_dataset = ("y" in split_file and "groups" in split_file
                    and np.array_equal(split_file["y"],      ds.y)
                    and np.array_equal(split_file["groups"], ds.groups))
    
    if not same_dataset:
        raise ValueError(f"split {path} was made on another version of the {ds.name} dataset")

    #-------------------------
    method = str(split_file["method"]) if "method" in split_file else "group"

    splits = Splits(train      = split_file["train"], 
                    validation = split_file["validation"], 
                    test       = split_file["test"],
                    method     = method)
    
    splits.check(ds)

    return splits
    
#================================================================================#
def split(ds          : GWDataset,
          train_ratio : float = 0.8,
          val_ratio   : float = 0.5,
          seed        : int   = 0,
          method      : str   = "random") -> Splits:

    """
    Split a guided-wave dataset into Train | Validation | Test sets, by acquisition groups.

    Healthy and damaged signals are split separately, each class with its own ratios :

    |    sets    |            healthy signals             |  damaged signals   |  
    |==========================================================================|
    |   train    |            train_ratio                 |         0          |
    | validation | (1 - train_ratio) * HEALTHY_VAL_RATIO  |     val_ratio      |
    |   test     | (1 - train_ratio) * HEALTHY_VAL_RATIO  |   1 - val_ratio    |

    method = "random" : each signal is assigned independently, as in a classic random split
                            (e.g. to reproduce papers). Consecutive, almost identical measurements can end up
                            in train AND test : the test scores are then optimistic.

    method = "group"  : all the signals of a group (same acquisition period -- ds.groups) go to the same
                        set, so that identical measurements are never on both sides of the split.
    
    The validation set is used for early stopping, threshold selection and hyperparameter tuning. 
    The test set is only used for the final evaluation.

    Args
    ----------
    ds : (GWDataset) raw loaded guided waves dataset
    
    train_ratio : (float) proportion of healthy signals used for training
    val_ratio   : (float) proportion of damaged signals used for validation
    seed        : (int) random seed
    method      : (str) "group" (by acquisition block) | "random" (signal by signal) -- defaults "group"

    Returns
    ----------
    split : Split object made with the given parameters 
    """

    #---------------------------------------------
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1) : {train_ratio}")

    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1) : {val_ratio}")

    if method not in SPLIT_METHODS:
        raise ValueError(f"unknown split method : {method} (expected one of {SPLIT_METHODS})")

    #---------------------------------------------
    HEALTHY_VAL_RATIO = 0.5

    healthy_idx = np.flatnonzero(ds.y == 0)
    damaged_idx = np.flatnonzero(ds.y == 1)

    #-------------------------
    if len(damaged_idx) == 0:
        raise ValueError(f"[{ds.name}] no damaged signal : impossible to evaluate damage detection "
                         f"(check the loader parameters, e.g. damage_states for OGW)")

    healthy_ratios = [train_ratio, 
                      (1 - train_ratio) * HEALTHY_VAL_RATIO, 
                      (1 - train_ratio) * HEALTHY_VAL_RATIO]
    
    damaged_ratios = [0.0,
                      val_ratio,
                      1 - val_ratio]

    #-------------------------
    if method == "group" :
        groups = ds.groups  
    else :
        np.arange(len(ds.y))

    healthy_parts = assign(healthy_idx, groups, healthy_ratios, rng_seed=seed)
    damaged_parts = assign(damaged_idx, groups, damaged_ratios, rng_seed=seed+1)
    
    #---------------------------------------------
    split = Splits(train      = np.sort(np.concatenate([healthy_parts[0], damaged_parts[0]])).astype(int),
                   validation = np.sort(np.concatenate([healthy_parts[1], damaged_parts[1]])).astype(int),
                   test       = np.sort(np.concatenate([healthy_parts[2], damaged_parts[2]])).astype(int),
                   method     = method)

    split.check(ds)

    return split

#================================================================================#
def assign(indices  : list[int], 
           groups   : np.ndarray, 
           ratios   : list[float] = [80., 10., 10.], 
           rng_seed : int = 42):

    """
    Splits data indices into Train | Validation | Test sets
    without separating elements that belong to the same group.

    This function uses a successive approach to ensure there is no data leakage between the sets.
    It is using two successive GroupShuffleSplits

    Args
    ----------
    indices   : (list[int]) index of a specific class (healthy or damaged)
    groups    : (np.ndarray) Tabs of groups labels corresponding to data. 
                The length must at least be equal to the max value in 'indices`.
    ratios    : (lis[int]) wanted amount of the data in each sets -- 
                defaults [80., 10., 10.] --> 

    rng_seed  : (int) random generator seed

    Returns:
    ----------
    list : [train, validation, test] list indicies for the Train, Validation and Test sets  
    """

    #---------------------------------------------
    indices = np.asarray(indices)
    total   = sum(ratios)

    (r_train, r_validation, r_test) = (r / total for r in ratios)

    #-------------------------
    if r_train == 0:
        (train_index, tmp_index) = (np.array([], dtype=int), indices)

    else:
        gss1 = GroupShuffleSplit(n_splits     = 1, 
                                 test_size    = (r_validation + r_test), 
                                 random_state = rng_seed)
        
        (train_idx,   tmp_idx)   = next(gss1.split(indices, groups=groups[indices]))
        (train_index, tmp_index) = (indices[train_idx], indices[tmp_idx])

    #-------------------------
    if r_validation == 0:
        (validation_index, test_index) = (np.array([], dtype=int), tmp_index)

    elif r_test == 0:
        (validation_index, test_index) = (tmp_index, np.array([], dtype=int))

    else:
        gss2 = GroupShuffleSplit(n_splits     = 1, 
                                 test_size    = r_test / (r_validation + r_test), 
                                 random_state = rng_seed)

        (validation_idx,   test_idx)   = next(gss2.split(tmp_index, groups=groups[tmp_index]))
        (validation_index, test_index) = (tmp_index[validation_idx], tmp_index[test_idx])

    train       = np.sort(train_index)
    validation  = np.sort(validation_index)
    test        = np.sort(test_index)

    return [train, validation, test]