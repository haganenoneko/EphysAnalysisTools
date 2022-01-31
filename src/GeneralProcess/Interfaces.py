# Copyright (c) 2022 Delbert Yip
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pandas as pd
from typing import Dict, List, Any
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------- #
#                          Interfaces for data objects                         #
# ---------------------------------------------------------------------------- #


class Recording:
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
#                        Interfaces for analysis classes                       #
# ---------------------------------------------------------------------------- #


class AbstractAnalyzer(ABC):
    @abstractmethod
    def __init__(self, data: Recording) -> None:
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def plot_results(self) -> None:
        """
        Plot analysis results. Plotting methods are usually implemented 
        in a separate class, as an instance of 'AbstractPlotter,' but may 
        also be implemented here.
        """
        pass

    @abstractmethod
    def extract_data(self, key: str) -> None:
        """Extract a class attribute with name 'key'"""
        pass


# ---------------------------------------------------------------------------- #
#                     Interfaces for visualization classes                     #
# ---------------------------------------------------------------------------- #

def save_pdf(data: Recording, fig: plt.Figure) -> None:

    if not hasattr(data, 'pdf'):
        return

    pdf = data.attrs['pdf']
    pdf.savefig(fig, bbox_inches='tight')


class AbstractPlotter(ABC):
    """Abstract interface for plotting leak subtraction results"""

    @abstractmethod
    def __init__(self, data: Recording, show=False) -> None:
        pass

    @abstractmethod
    def create_figure(self) -> None:
        return

    @abstractmethod
    def add_labels(self) -> None:
        return

    @abstractmethod
    def add_legend(self) -> None:
        return

    @abstractmethod
    def format_axes(self) -> None:
        return

    @abstractmethod
    def plot(self) -> None:
        return

    def save(
        self, fig: plt.Figure, save_path: str,
        data: Recording, show=False
    ) -> None:
        save_pdf(data, fig)
        self.fig.savefig(save_path, dpi=300)

        if show:
            plt.show()
        plt.close(self.fig)
