# Copyright (c) 2021 Delbert Yip
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np 
import pandas as pd 

from abc import ABC, abstractmethod

from pydantic import BaseModel, ValidationError, validator

from typing import Dict, Any, List, Union, TypeVar, Tuple
import numpy.typing as npt 

# ------------------------------- Custom types ------------------------------- #

TNumber = TypeVar('TNumber', int, float)

NDArrayBool = npt.NDArray[np.bool_]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float64]

FloatOrArray = Union[float, NDArrayFloat]

KwDict = Dict[str, Any]

# ------------------------------- Base classes ------------------------------- #

class Recording(BaseModel):
    """Base class to hold data for individual recordings"""
    raw_data: pd.DataFrame
    name: str 
    params: pd.Series
    epoch_intervals: Dict[int, List[int]]
    attrs: Dict[str, Any]

class RecordingWithLeak(Recording):
    """Recordings that contain leak ramp steps"""
    ramp_startend: List[int]
    
class RecordingWithMemTest(RecordingWithLeak):
    """Recordings that contain leak ramp and membrane test steps"""
    mt_startend: List[int]

# ---------------------------------------------------------------------------- #

class AbstractAnalyzer(ABC):
    @abstractmethod
    def __init__(self, data: Recording, show: bool
    ) -> None:
        pass 
    
    @abstractmethod 
    def run(self):
        pass 
    
    @abstractmethod
    def plot_results(self) -> None:
        pass 
    
    @abstractmethod
    def extract_data(self, key: str) -> None:
        pass 

    @classmethod 
    def save_pdf(cls: object, fig: plt.Figure) -> None:
    
        if '_data' not in cls.__dict__: 
            return 
        elif cls._data.attrs['pdf'] is None: 
            return 
        
        pdf = cls._data.attrs['pdf']
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig=fig)
        
        return 
        
# ----------------------------- Simple functions ----------------------------- #

def scalarTimesList(scalar: TNumber, lst: List[TNumber]) -> List[TNumber]:
    return [scalar*x for x in lst]

def extendListAsArray(lst: List[TNumber], dims: Tuple[int, int]) -> Union[NDArrayFloat, NDArrayInt]:
    
    elemType = type(lst[0])
    lst = np.full((dims[0], len(lst)), lst, dtype=elemType)
    
    if lst.shape == dims: 
        return lst 
    
    extension = (0, dims[1] - lst.shape[1])
    lst = np.pad(lst, ((0, 0), extension), 
                    mode='constant', constant_values=lst[-1])
    
    return lst     

# define a single exponential
def exp1(
    t: FloatOrArray, dI: FloatOrArray, tau: FloatOrArray, I_ss: FloatOrArray
): 
    """Single-exponential function

    :param t: time
    :type t: FloatOrArray
    :param dI: current
    :type dI: FloatOrArray
    :param tau: time constant
    :type tau: float
    :param I_ss: steady-state current
    :type I_ss: FloatOrArray
    :return: simulated exponential
    :rtype: NDArrayFloat
    """
    return dI*np.exp(-t/tau) + I_ss