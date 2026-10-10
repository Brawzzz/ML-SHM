#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer

from .GWDataset import GWDataset
from .Split import Splits


#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def prepare(ds: GWDataset, splits: Splits, scaler_name: str = "minmax") -> tuple[dict, object]:

    """
    Prepar the differents Splits sets for runs 
    
        1 - Normalization : Fit the scaler on the training set and apply it to the three sets.

    Parameters:
    ----------
    ds          : (GWDataset.GWDataset) loaded data as a GWDataset
    splits      : (Split.Splits) split performed on 'ds'
    scaler_name : (str) name of the scaler bject to use -- defaults : scaler_name = "minmax"

    Returns
    ----------
    data : dictionnary containing the values of three scaled sets 
            
            {
            "train"     : (X, y), 
            "validation" : (X, y), 
            "test"       : (X, y)
            }  (float32 / int64)

    scaler : the fitted scaler (to save with the model)
    """

    #---------------------------------------------
    scaler = _scaler(scaler_name)
    scaler.fit(ds.X[splits.train])

    data = {}
    for name in ("train", "validation", "test"):
        idx        = getattr(splits, name)
        data[name] = (scaler.transform(ds.X[idx]).astype(np.float32), ds.y[idx])

    return (data, scaler)

#================================================================================#
def _scaler(name: str) -> MinMaxScaler | StandardScaler | FunctionTransformer:

    """
    Return the wanted scaler corresponding to "name" 

    Parameters:
    ----------
    name : (str) name of the wanted scaler method

        :minmax   : sklearn.MinMaxScaler (min/max)
        :standard : sklearn.StandardScaler (mean/std)
        :none     : sklearn.FunctionTransformer (identity)

    Returns:
    ----------
    scaler : (MinMaxScaler | StandardScaler | FunctionTransformer) empty scaler corresponding to name
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