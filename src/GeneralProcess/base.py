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

from dataclasses import dataclass

from typing import Dict, Any, List, Union
import numpy.typing as npt 

# ------------------------------- Custom types ------------------------------- #
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float64]

FloatOrArray = Union[float, NDArrayFloat]

# ------------------------------- Base classes ------------------------------- #

@dataclass
class AbstractRecording:
    raw_data: pd.DataFrame
    name: str 
    params: pd.Series
    ramp_startend: List[int]
    epoch_intervals: Dict[int, List[int]]
    attrs: Dict[str, Any]
    
    """Base class to hold data for individual recordings"""

class AbstractAnalyzer(ABC):
    @abstractmethod
    def __init__(self, data: AbstractRecording, show: bool
    ) -> None:
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