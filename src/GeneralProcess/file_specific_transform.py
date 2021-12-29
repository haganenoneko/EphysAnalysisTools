# Copyright (c) 2021 Delbert Yip
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

""" File-specific transforms 
Use this script to implement file-specific transformations to the data. 

A *justified* use case is when a recording(s) whose protocol did not 
follow the specified command for one or more sweeps. In this case, add 
a subclass of `AbstractTransform` that either corrects or removes
the 'offending' sweeps.

A justified use case should not modify the current record. An *unjustified* use 
case manipulates the data in such a way that the current record (timepoints, 
current amplitudes, sweep order, etc.) are substantially modified. 

If desired, use the `with_validation` decorator, which applies simple constraints to the transformation.
These constraints are:
1. The current record is untouched, or removed with the corresponding voltage command
2. The order of sweeps is preserved
3. The maximum and minimum voltage in each command is preserved. 
"""

import numpy as np 
import pandas as pd

from typing import List, Tuple, Dict
from typing import Callable, Union

from abc import ABC, abstractmethod

from decorator import decorator 

from functools import singledispatch

from GeneralProcess.base import NDArrayFloat, NDArrayInt, NDArrayBool
from GeneralProcess.base import extendListAsArray

# ------------------------------- Custom types ------------------------------- #

TransformInput = Tuple[str, List[int], pd.DataFrame]
TransformOutput = Tuple[List[int], pd.DataFrame]
TransformType = Callable[[str, List[int], pd.DataFrame], TransformOutput]

# -------------------------------- Interfaces -------------------------------- #

class AbstractTransform(ABC):
    
    @abstractmethod
    def __init__(self, df: pd.DataFrame) -> None:
        pass 
    
    @staticmethod
    @abstractmethod
    def static_transform(self, func: TransformType): 
        pass 

# --------------------------- Transform validation --------------------------- #
class TransformError(ValueError):
    def __init__(self, validation: str, data: pd.DataFrame) -> None:
        super().__init__(
            f"This transform does not satisfy the constraint:\
                < {validation} >\n{data}"
        )

def validateVoltageRange(transformed: pd.DataFrame, vlims: Tuple[int, int]=(-200, 100)) -> bool:
    """Ensure that all voltages are in `self._voltage_range`"""

    N = int(transformed.shape[1]/2)
    df = transformed.iloc[:, N:]
    v_min = df.min(axis=0) 
    v_max = df.max(axis=0)
    
    try:
        assert (v_min.copy() > vlims[0]).all()
    except AssertionError:
        raise TransformError(
            f"There are voltages below minimum {vlims[0]}",
            v_min
        )
    
    try:
        assert (v_max.copy() < vlims[1]).all()
    except AssertionError:
        raise TransformError(
            f"There are voltages above the maximum {vlims[1]}",
            v_max
        )
    
    return True 
        
def validateColumns(original: pd.DataFrame, transformed: pd.DataFrame) -> bool:
    """Ensure that columns match the original or are even in number"""
    
    n_col = original.shape[1]
    assert n_col % 2 == 0
    
    if transformed.shape[1] == n_col: 
        return True 
    
    if transformed.shape[1] % 2 != 0: 
        data = f"{original.head}\n{transformed.head}"
        raise TransformError("An even number of columns.", data)
    
    return True 
    
def validateRows(original: pd.DataFrame, transformed: pd.DataFrame) -> bool:
    """Ensure that rows match the original or are equal in number"""
    
    if transformed.shape[0] == original.shape[0]:
        return True 
        
    def get_nrows(df: pd.DataFrame, col: int) -> int:
        return df.iloc[:, col].dropna().shape[0]
    
    n_0 = get_nrows(transformed, 0)
    for i in range(transformed.shape[1]):
        n_i = get_nrows(transformed, i)
        if n_i == n_0: continue 

        data = f"First column: {n_0}\t{i}-th column: {n_i}"
    
        raise TransformError(
            "Equal number of non-NaN elements in each column.",
            data
        )
            
    return True 

@decorator 
def with_validation(
    func: TransformType, *args: TransformInput, vlims: Tuple[int, int]=(-200, 200)
) -> TransformOutput:
    
    epoch_inds, transformed = func(*args)
    validateColumns(args[2], transformed)
    validateVoltageRange(transformed, vlims=vlims)
    validateRows(args[2], transformed)
    return epoch_inds, transformed 

# -------------------------- Implemented transforms -------------------------- #

@singledispatch
def linTransEpochTimes(
    times: List[float], 
    shift: Union[float, List[float]]=None, 
    scale: Union[float, List[float]]=None, 
    condition_on_sweep_index: Callable[[NDArrayInt], NDArrayBool]=None,
    condition_on_epoch_index: Callable[[NDArrayInt], NDArrayBool]=None,
    condition_on_value: Callable[[NDArrayFloat], bool]=None
) -> List[float]:

    arr = np.array(times)
    dims = np.array([np.arange(x, dtype=int) for x in arr.shape])
    
    if condition_on_sweep_index is not None:
        dims[0] = condition_on_sweep_index(dims[0])
    else:
        dims[0] = 1
        
    if condition_on_epoch_index is not None:
        dims[1] = condition_on_epoch_index(dims[1])
    else:
        dims[1] = 1 
    
    if isinstance(shift, list):
        shift = extendListAsArray(shift, arr.shape)
    
    if isinstance(scale, list):
        scale = extendListAsArray(scale, arr.shape)
    
    if condition_on_value is None:
        return (arr + shift).tolist()
    
    for (i, j), value in np.ndenumerate(arr):
        
        if not (condition_on_value(value) and dims[i,j]): 
            continue 
        
        arr[i, j] = value*scale[i, j] + shift[i, j] 
        
    return arr.tolist()

D = Dict[str, List[List[float]]]
@linTransEpochTimes.register
def _(times: D, fname: str, **kwargs) -> D:
    
    if fname not in times:
        return times 
    
    ts = times[fname]
    for i, t in enumerate(ts):
        ts[i] = linTransEpochTimes(t, **kwargs)
        
    return times 