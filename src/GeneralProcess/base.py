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

from typing import Dict, Any

# ---------------------------------------------------------------------------- #

@dataclass
class AbstractRecording:
    raw_data: pd.DataFrame
    name: str 
    params: pd.Series
    attrs: Dict[str, Any]

class AbstractAnalyzer(ABC):
    @abstractmethod
    def __init__(
        self, data: AbstractRecording, show: bool, 
        aggregate_pdf: PdfPages
    ) -> None:
        pass 
    
    @abstractmethod
    def plot_results(self, pdf: PdfPages) -> None:
        pass 
    
    @abstractmethod
    def extract_data(self, key: str) -> None:
        pass 
