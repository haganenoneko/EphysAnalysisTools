# Copyright (c) 2022 Delbert Yip
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import glob
import math
import os
from multiprocessing import Value

from abc import ABC, abstractmethod
from typing import Callable, Tuple, List, Dict, Any, Union
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages

from scipy.interpolate import UnivariateSpline
from scipy.stats import pearsonr, sem

import lmfit

from dataclasses import dataclass

from GeneralProcess.ActivationCurves import multi_sort
from GeneralProcess.Base import NDArrayFloat, CleanlyDropNaNs, Recording, AbstractPlotter, AbstractAnalyzer, multi_sort

# --------------------------------- Constants -------------------------------- #

RTF_CONSTANT = (8.31446261815324*298)/(96.485)

# ---------------------------------------------------------------------------- #


def getIonSets(
    ion_set: Union[str, Dict[str, List[int]]],
    permeant=['Na', 'K', 'Cl']
) -> Dict[str, List[int]]:
    """
    Get internal and external concentrations of ions
    """
    # dictionary of ion concentrations, {ion : [internal, external]}
    if ion_set is None:
        return {"Na": [10, 110], "K": [130, 30], "Cl": [141, 144.6]}

    if ion_set == "HighK":
        return {"Na": [10, 135], "K": [130, 5.4], "Cl": [141, 144.6]}

    if not isinstance(ion_set, dict):
        raise ValueError(
            f"`ion_set` can only be None, 'HighK', or a Dictionary of ion concentrations, not\n{ion_set}")

    if not all(x in ion_set for x in permeant):

        logging.warning(
            f"The following ion set\n{ion_set}\n does not have one or more of permeant ions: {permeant}. Resorting to default: Na, K, Cl.")

        permeant = ['Na', 'K', 'Cl']

    for ion in permeant:

        if ion not in ion_set:
            raise KeyError(f"Ion set does not contain concentrations of {ion}")

        if len(ion_set[ion]) != 2:
            raise ValueError(
                f"Ion set does not contain two values (internal, external concentrations) for ion {ion}:\n{ion_set}")

    return ion_set

# ---------------------------------------------------------------------------- #
#                            Models of leak current                            #
# ---------------------------------------------------------------------------- #


class AbstractCurrentModel(ABC):
    """Abstract interface for models of leak current"""

    @abstractmethod
    def set_params(self) -> lmfit.Parameters:
        """Create lmfit Parameters"""
        return

    @abstractmethod
    def simulate(
        self, params: lmfit.Parameters,
        voltages: Union[float, NDArrayFloat]
    ) -> Union[float, NDArrayFloat]:
        """Simulate leak current for given voltages"""
        return

    def residual(
        self, params: lmfit.Parameters,
        voltages: NDArrayFloat, current: NDArrayFloat
    ) -> NDArrayFloat:
        model = self.simulate(params, voltages)
        return model - current

    def fit(self, voltages: NDArrayFloat, current: NDArrayFloat,
            method='leastsq', **kwargs) -> lmfit.minimizer.MinimizerResult:

        params = self.set_params()

        if len(voltages.shape) > 1 or len(current.shape) > 1:
            raise ValueError(
                f"Voltages and current should be one-dimensional, but have shapes {voltages.shape} and {current.shape}, respectively.")

        res = lmfit.minimize(self.residual, params, args=(current,),
                             method=method, **kwargs)
        self.res = res
        return res

    def __repr__(self) -> str:
        if self.res is None:
            return ""
        return lmfit.fit_report(self.res)

# ---------------------------- GHK current models ---------------------------- #


class AbstractGHKCurrent(AbstractCurrentModel):
    """
    Abstract model of GHK current

    References for GHK equations:
    1. Johnston and Wu, p. 58
    2. Hille 2001, p. 473
    3. https://en.wikipedia.org/wiki/Goldman%E2%80%93Hodgkin%E2%80%93Katz_flux_equation
    """

    def __init__(
        self, ion_set: Dict[str, List[List[float]]], RMP: float
    ) -> None:

        self.ion_set = ion_set
        self.ion_deltas()
        self.E_RMP = math.exp(RMP / RTF_CONSTANT)

    def ion_deltas(self) -> None:
        """Compute internal - external for each ion"""
        self.ion_deltas = {
            name: conc[0] - conc[1] for name, conc in self.ion_set.items()
        }

    @staticmethod
    def ghk(voltages: NDArrayFloat, d_ion: float) -> NDArrayFloat:
        """GHK current equation"""

        E = voltages / RTF_CONSTANT
        E2 = np.exp(-E)

        return 96.485 * E * (d_ion*E2) / (1 - E2)

    def ghk_zero(self, parvals: Dict[str, float]) -> float:
        """GHK evaluated at 0 mV"""
        current = 0.

        for ion, perm in parvals.items():
            if ion not in self.ion_deltas:
                continue
            current += perm * self.ion_deltas[ion]

        return 96.485 * current

    def ghk_nonzero(
        self, voltages: Union[float, NDArrayFloat], parvals: Dict[str, float]
    ) -> Union[float, NDArrayFloat]:

        return sum(
            (perm * self.ghk(voltages, self.ion_deltas[ion]) for
             ion, perm in parvals.items())
        )

    @abstractmethod
    def get_all_permeabilities(
        self, params: lmfit.Parameters
    ) -> Dict[str, float]:
        """Compute permeabilities for all ions"""
        return

    def simulate(
        self, params: lmfit.Parameters, voltages: Union[float, NDArrayFloat]
    ) -> Union[float, NDArrayFloat]:
        """
        Compute GHK leak equation, assuming nothing about the number
        or identity of permeant ions.
        """
        # parvals = params.valuesdict()
        parvals = self.get_all_permeabilities()

        # ghk current evaluated at zero mV
        i_zero = self.ghk_zero(parvals)

        if hasattr(voltages, 'shape'):

            out = np.zeros_like(voltages)
            nonzero = np.nonzero(voltages)

            out[~nonzero] = i_zero
            out[nonzero] = sum(self.ghk_nonzero(voltages[nonzero], parvals))

            return out

        if voltages == 0:
            return i_zero
        else:
            return sum(self.ghk_nonzero(voltages, parvals))

    def residual(
        self, params: lmfit.Parameters, voltages: NDArrayFloat,
        current: NDArrayFloat
    ) -> NDArrayFloat:

        model = self.simulate(params, voltages)
        return model - current

    def _create_repr_header(self) -> str:
        header = f"GHK model\nIon set\n{'Ion':<8}{'Internal':^8}{'External':^8}\n"
        for name, conc in self.ion_set.items():
            header += f"{name:<8}{conc[0]:^8}{conc[1]:^8}"
        return header

    def __repr__(self) -> str:

        header = self._create_repr_header()
        try:
            return header + '\n' + self.res
        except KeyError:
            logging.info("No fit results to show.")
            return header


class BionicGHKCurrent(AbstractGHKCurrent):

    def set_params(self) -> lmfit.Parameters:
        params = lmfit.Parameters()
        params.add('K', value=1e-5, min=1e-8, max=1.)
        return params

    def _get_perm(
        self, P_K: Union[float, NDArrayFloat]
    ) -> Union[float, NDArrayFloat]:
        """Infer absolute P_Na"""
        K = self.ion_set['K']
        Na = self.ion_set['Na']
        E = self.E_RMP
        return P_K*((E*K[0] - K[1]) / (Na[1] - E*Na[0]))

    def get_all_permeabilities(self, params: lmfit.Parameters) -> Dict[str, float]:

        K = self.ion_set['K']
        Na = self.ion_set['Na']
        E = self.E_RMP

        parvals = params.valuesdict()
        parvals['Na'] = parvals['K']*((E*K[0] - K[1]) / (Na[1] - E*Na[0]))
        return parvals


class TrionicGHKCurrent(AbstractGHKCurrent):

    def set_params(self) -> lmfit.Parameters:
        params = lmfit.Parameters()
        params.add('K', value=1e-5, min=1e-8, max=1.)
        params.add('Cl', value=0.1, min=1e-8, max=1.)
        return params

    def get_all_permeabilities(self, params: lmfit.Parameters) -> Dict[str, float]:

        K = self.ion_set["K"]
        Na = self.ion_set["Na"]
        Cl = self.ion_set["Cl"]

        parvals = params.valuesdict()

        parvals['Na'] = (
            parvals['Cl']*(E*Cl[1] - Cl[0]) + (E*K[0] - K[1])
        ) / (Na[1] - E*Na[0])

        return parvals

# -------------------------------- Ohmic leak -------------------------------- #


class OhmicCurrent(AbstractCurrentModel):
    """Simple Ohmic model of leak current. No initialization needed."""

    def set_params(self) -> lmfit.Parameters:
        params = lmfit.Parameters()
        params.add('g_leak', value=-1., min=-15., max=15.)
        params.add('E_leak', value=-10., min=-80., max=20.)

    def simulate(self, params: lmfit.Parameters, voltages: Union[float, NDArrayFloat]) -> Union[float, NDArrayFloat]:
        return params['g_leak'] * (voltages - params['E_leak'])

    def __repr__(self) -> str:
        return "Ohmic model\n" + super().__repr__()


# ---------------------------------------------------------------------------- #
#                              Locate leak region                              #
# ---------------------------------------------------------------------------- #

class LocateLeak(ABC):
    """
    Abstract interface for implementing methods that find
    subset of data to perform leak subtraction on
    """
    @abstractmethod
    def __init__(self, df: pd.DataFrame) -> None:
        return

    def _preprocess_data(self, df: pd.DataFrame):

        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        elif not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Data must be a numpy array or pd.DataFrame, not {type(df)}"
            )

        self.N = int(df.shape[1]/2)

    @staticmethod
    def _downsample(data: pd.DataFrame, factor: int) -> pd.DataFrame
        if factor < 1:
            return data
        return data.iloc[::factor, :]

    @abstractmethod
    def find(self) -> pd.DataFrame:
        return

    def clone_target(
        self, target: Union[pd.Series, pd.DataFrame],
        agg: Callable[[NDArrayFloat], NDArrayFloat] = np.mean,
        **agg_kwargs
    ) -> pd.DataFrame:
        """
        Clone target region used for fitting leak current
        Aggregate dataframe columns using `agg`
        """
        if len(target.shape) < 2:
            return pd.DataFrame(
                np.tile(target.values, (1, self.N))
            )

        if agg is not None:
            new_target = agg(target, **agg_kwargs)
            return pd.DataFrame(
                np.tile(new_target.values, (1, self.N))
            )
        else:
            return target

# ------------------------- Classic leak ramp locator ------------------------ #


class LocateLeakRamp(LocateLeak):
    def __init__(
        self, df: pd.DataFrame, ramp_startend: List[List[int]]
    ) -> None:
        self.df = self._preprocess_data(df)

        if len(ramp_startend)/2 != self.N:
            raise ValueError("Number of leak ramp epochs != number of traces.")

        self.ramp_startend = ramp_startend

    def find(self, downsample=0) -> pd.DataFrame:
        """Extract voltage and current of voltage ramps"""

        df = self.df

        if len(self.ramp_startend) < 2:
            self.transformed = False

            t0, t1 = self.ramp_startend

            return self._downsample(df.iloc[t0:t1+1, :], downsample)

        # file-specific transform for leak ramp epochs
        # returns a list of start and end times
        self.transformed = True
        ramp_epochs = self.ramp_startend

        ramp_df = []
        ramp_pro_df = []

        for i, t0 in enumerate(ramp_epochs[::2]):
            t1 = ramp_epochs[2*i + 1]

            ramp_df.append(
                df.iloc[t0:t1+1, i]
            )

            ramp_pro_df.append(
                df.iloc[t0:t1+1, self.N + i]
            )

        ramp_df = CleanlyDropNaNs(pd.concat(ramp_df, axis=1))
        ramp_pro_df = CleanlyDropNaNs(pd.concat(ramp_pro_df, axis=1))
        return self._downsample(
            pd.concat([ramp_df, ramp_pro_df], axis=1),
            downsample
        )


# ------------------------- Alternative leak locators ------------------------ #

class GUILeakLocator(LocateLeak):
    """
    GUI interface to select target area of protocol
    to use for fitting leak current parameters
    """

    def __init__(self, df: pd.DataFrame, epoch_times: List[List[int]]) -> None:
        self.df = df
        self.epoch_times = epoch_times

        raise NotImplementedError(f"GUILeakLocator is not implemented yet")

    def find(self) -> pd.DataFrame:
        return


class VaryingStepLeakLocator(LocateLeak):
    """
    Finds target area of protocols where voltage differs
    between traces, accoridng to some threshold, and
    if multiple such targets are identified, returns the
    most varying region.
    """

    def __init__(
        self, df: pd.DataFrame, epoch_times: List[List[int]],
        epoch_levels: List[List[float]]
    ) -> None:
        self.df = df
        self.epoch_times = epoch_times
        self.epoch_levels = epoch_levels

        raise NotImplementedError(
            "VaryingStepLeakLocator is not implemented yet")

    def find(
        self, min_dV: float = 50.,
        rank_metric: Callable[[NDArrayFloat], NDArrayFloat] = None
    ) -> pd.DataFrame:
        return

# ---------------------------------------------------------------------------- #
#                              Subtraction methods                             #
# ---------------------------------------------------------------------------- #


@dataclass
class LeakSubtractResults:
    """Results of leak subtraction"""
    fitted: Union[NDArrayFloat, List[NDArrayFloat]]
    params: List[lmfit.Parameters]
    rvals_fitVolts: NDArrayFloat
    rvals_dataVolts: NDArrayFloat
    subtracted: pd.DataFrame


def checkSubtractionQuality(
    original: pd.DataFrame, subtracted: pd.DataFrame,
    results: LeakSubtractResults, threshold=0.02, down: int = 5
) -> None:
    if threshold < 0 or threshold > 1:
        logging.warning(
            f"Threshold must be within (0, 1], not {threshold}. Setting to default, 0.02")
        threshold = 0.02

    params = results.params
    par_names = params[0].valuesdict().keys()
    params = {name: [p[name] for p in params] for name in par_names}

    header = f"{'Mean':<8}{'Std':^8}"

    mu = np.mean(results.rvals_fitVolts)
    if mu < 0.9:
        logging.warning(
            f"Fit has poor correlation to the calibrated voltage record: {mu}. 0.9 is considered good.")

    logging.info(f"""
        {'Pearson correlation (Recorded current vs. Voltage)':<8}
        {header}
        {np.mean(results.rvals_dataVolts):<8}{np.std(results.rvals_dataVolts):^8}
        \nPearson correlation (Fit vs. Voltage)
        {header}
        {mu:<8}{np.std(results.rvals_fitVolts):^8}""")

    N = int(original.shape[1]/2)
    sgn = (subtracted.iloc[::down, :N] / original.iloc[::down, :N]).values
    diff = (subtracted.iloc[::down, :N] - original.iloc[::down, :N]).values

    n_sgn = np.nonzero(sgn < 0)
    n_diff = np.nonzero(diff > np.max(diff) * threshold)
    n_both = np.intersect1d(n_sgn, n_diff)

    n_sgn = 100*n_sgn.size / sgn.size
    n_diff = 100*n_diff.size / diff.size
    n_both = 100*n_both.size / sgn.size

    logging.info(
        f"""
        \nPercentage of leak-subtracted record compared to original record
        {'Opposite polarity':<8}{f'Deviation by more than {threshold}\%':^8}{'Both':>8}
        {f'{n_sgn}\%':<8}{f'{n_diff}\%':^8}{f'{n_both}\%':>8}
        """
    )

    if n_both > 5:
        logging.warning(
            f"More than 5\% of the leak-subtracted current record exhibits opposite polarity and deviation greater than {threshold}\% compared to the original current record.")


class LeakSubtractor(AbstractAnalyzer):

    def __init__(self, data: Recording) -> None:
        self.data = data
        self.df = data.raw_data
        self.N = int(self.df.shape[1]/2)

        # uninitialized
        self.target: pd.DataFrame = None
        self.res: LeakSubtractResults = None

    def set_method(self, method: str, RMP: float = None,
        ion_set: Dict[str, List[float]] = None,
                   permeant_ions=['Na', 'K', 'Cl']
                   ) -> None:

        if method not in ['ohmic', 'ghk2', 'ghk3', 'ghk']:
            raise ValueError(
                f"Leak current model must be one of ['ohmic', 'ghk2', 'ghk3', 'ghk'], not {method}.")

        if method == 'ohmic':
            self.model = OhmicCurrent
            self.plotter = PlotOhmic
            return

        if RMP is None or not isinstance(RMP, float):
            raise ValueError(
                f"Resting membrane potential must be specified to model leak current with GHK equations. Provided {RMP}")

        ion_set = getIonSets(ion_set, permeant=permeant_ions)

        if method in ['ghk', 'ghk2']:
            self.model = BionicGHKCurrent(ion_set, RMP)
        elif method == 'ghk3':
            self.model = TrionicGHKCurrent(ion_set, RMP)

        self.plotter = PlotGHK

        return

    def set_locator(self, locator: str, **kwargs):

        if locator not in ['ramp', 'GUI', 'step']:
            raise ValueError(
                f"Locator method must be one of ['ramp', 'GUI', 'step'], not {locator}.")

        if locator == 'ramp'
           if not hasattr(self.data, 'ramp_startend'):
                raise AttributeError(
                    f"Data does not have attribute 'ramp_startend'. Perhaps try using a different locator method (using {locator}).")

            self.locator = LocateLeakRamp(
                self.data.raw_data, self.data.ramp_startend)

        elif locator == 'GUI':
            self.locator = GUILeakLocator(
                self.data.raw_data, self.data.epoch_intervals)

        elif locator == 'step':
            self.locator = VaryingStepLeakLocator(
                self.data.raw_data, self.data.epoch_intervals, **kwargs)

    def find(self, **kwargs):
        """Find and store the target region for fitting leak current models"""
        self.target = self.locator.find(**kwargs)

    def subtract(self, **fit_kwargs) -> LeakSubtractResults:
        """Fit leak current models and apply subtraction over entire recording"""
        N = self.N
        subtracted = self.df
        leak_params: List[lmfit.Parameters] = []

        fitted: List[NDArrayFloat] = []
        rvals_fitVolts = np.zeros(self.N)
        rvals_dataVolts = np.zeros(self.N)

        for i in range(N):
            sweep = self.target.iloc[:, [i, N+i]].dropna().values

            res = self.model.fit(sweep[:, 1], sweep[:,0], **fit_kwargs)
            logging.info(repr(self.model))
            logging.info(f"Fit status for {i}-th trace: {res.message}")

            leak_params.append(res.params)

            leak_i = self.model.simulate(res.params, self.df.iloc[:, N+i])
            fitted.append(leak_i)

            rvals_fitVolts[i] = pearsonr(sweep[:, 1], leak_i)[0]
            rvals_dataVolts[i] = pearsonr(sweep[:, 1], sweep[:,0])[0]

            subtracted.iloc[:, i] -= leak_i 

        self.res = LeakSubtractResults(
            fitted, leak_params, rvals_fitVolts,
            rvals_dataVolts, subtracted
        )
        return self.res

    def plot_results(self, **kwargs) -> None:
        if self.res is None:
            raise ValueError(f"Leak subtraction results are not available.")

        self.plotter(self.data, **kwargs).plot(self.res)

    def check(self, **kwargs) -> None:
        checkSubtractionQuality(
            self.data.raw_data, self.res.subtracted, self.res, **kwargs
        )

    def run(self, model: str, locator: str, show=False,
        model_kwargs: dict={}, locator_kwargs: dict={},
        find_kwargs: dict={}, fit_kwargs: dict={},
            plot_kwargs: dict={}, check_kwargs: dict={}
            ):
        self.set_method(model, **model_kwargs)
        self.set_locator(locator, **locator_kwargs)
        self.find(**find_kwargs)
        self.subtract(**fit_kwargs)
        self.check(**check_kwargs)

        if show:
            self.plot_results(**plot_kwargs)

    def extract_data(self, key: str) -> Any:
        if key is None:
            return self.res

        if not hasattr(self, key) and not hasattr(self.res, key):
            raise AttributeError(f"{key} is not a valid attribute")
        elif hasattr(self, key):
            return self.__getattribute__(key)
        elif hasattr(self.res, key):
            return self.res.__getattribute__(key)

# ---------------------------------------------------------------------------- #
#                               Plotting methods                               #
# ---------------------------------------------------------------------------- #


class PlotOhmic(AbstractPlotter):

    def __init__(
        self, data: Recording, results: LeakSubtractResults,
        save_path: str, show=False, downsample: int =5
    ) -> None:
        self.fig: plt.figure = None
        self.axs: List[plt.Axes] = None
        self.downsample = downsample

        self.create_figure()
        self.add_labels()
        self.plot(results)
        self.save(show, save_path, data)

    def create_figure(self) -> None:
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 3)

        axs = [gs[0, 0], gs[0,1], gs[0,2], gs[1,:]]

        self.fig = fig
        self.axs = [fig.add_subplot(g) for g in axs]

    def add_labels(self) -> None:
        self.axs[0].set_title(r"$\gamma_{\mathregular{leak}}$ (pS)")
        self.axs[1].set_title(r"$E_{\mathregular{leak}}$ (mV)")
        self.axs[2].set_title(r"$r^2$")
        self.axs[3].set_ylabel(r"Current (pA)")
        self.axs[3].set_xlabel("Time (ms)")

    def plot_leak_params(self, res: LeakSubtractResults):
        """Plot fit parameters and correlation coefficients"""

        params = res.params
        xpos = range(len(params))
        kwargs = dict(marker='o', mec='k', ls='none', markersize=6)

        # plot parameter values
        self.axs[0].plot(xpos, [p['g_leak'] for p in params],
                         c='k', **kwargs)

        self.axs[1].plot(xpos, [p['E_leak'] for p in params],
                         c='k', **kwargs)

        # plot pearson R correlations
        self.axs[2].plot(xpos, res.rvals_dataVolts,
                         c='orange', label="I-V", **kwargs)

        self.axs[2].plot(xpos, res.rvals_fitVolts,
                         c='k', label="Fit-I", **kwargs)

    def format_axes(self, N: int) -> None:

        self.axs[3].locator_params(axis='x', nbins=5)
        self.axs[3].locator_params(axis='y', nbins=5)

        # if more than 6 traces, make every 2nd label empty
        xlabs = np.arange(1, N+1)
        if N > 6:
            xlabs[1::2] = ""

        # tick appearance
        for a in self.axs[:3]:
            a.set_xticks(range(N))
            a.set_xticklabels(xlabs)
            a.set_xlabel("Sweep #")

        for a in self.axs:
            ybot, ytop = a.get_ylim()
            if ytop == 1:
                a.set_ylim(ybot - 0.05, ytop + 0.05)
            else:
                dy = (ytop - ybot)*0.05
                a.set_ylim(ybot - dy, ytop + dy)

        self.fig.suptitle("Fitting of Linear Voltage Ramps")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def plot(self, results: LeakSubtractResults) -> None:

        ax3 = self.axs[3]
        down = self.downsample
        N = int(results.subtracted.shape[1]/2)

        subtracted = results.subtracted
        times = subtracted.index[::down]
        times -= times[0]

        # plot pre-subtracted current
        ax3.plot(times[::down], results.subtracted.iloc[::down, :N], 
                marker='o', markersize=3, markevery=5, ls='none',
                 c='gray', label="Original")

        # plot subtracted current
        ax3.plot(
            times[::down], self.ramp_df.iloc[::down, :N], 
            lw=2, c='r', label="Subtracted")

        # plot leak parameters
        self.plot_leak_params(results)

        fitted = results.fitted
        line_border = [pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]
        kwargs = dict(lw=2, c='yellow', path_effects=line_border)

        # plot fitted leak current
        if isinstance(fitted, np.ndarray):
            ax3.plot(times, np.transpose(fitted),
                     lw=1.5, c='blue', ls=':', label="Fit")
        elif isinstance(fitted, list):
            for i, fit in enumerate(fitted):
                times = subtracted.iloc[::down, i].dropna().index                
                label = "Fit" if i == 0 else None
                ax3.plot(times, fit[::down], label=label, **kwargs)
        else:
            raise TypeError(f"Fitted data should be a Numpy array or list of numpy arrays, not {type(fitted)}")

        self.add_legend(N)
        self.format_axes()

    def add_legend(self, N: int) -> None:

        ax2, ax3 = self.axs[3:]

        h, l = ax2.get_legend_handles_labels()
        ax2.legend(
            h[:2], list(set(l)),
            loc=1, bbox_to_anchor=[1.65, 1])

        h, l = ax3.get_legend_handles_labels()
        ax3.legend([h[0], h[N], h[-1]], [l[0], l[N], l[-1]],
                   loc=1, bbox_to_anchor=[1.19, 1])


class PlotGHK(AbstractPlotter):
    def __init__(
        self, data: Recording, results: LeakSubtractResults,
        model: AbstractGHKCurrent, save_path: str,
        show=False, downsample: int =5
    ) -> None:
        self.fig: plt.figure = None
        self.axs: List[plt.Axes] = None
        self.downsample = downsample

        self.create_figure()
        self.add_labels()
        self.plot_leak_params(results, model)
        self.plot(results)
        self.save(show, save_path, data)

    def create_figure(self) -> None:
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, 7)

        # ion permeability values for K
        ax0 = fig.add_subplot(gs[0, :3])
        ax1 = fig.add_subplot(gs[1, :3])
        ax2 = fig.add_subplot(gs[:, 3:])

        self.fig = fig
        self.axs = [ax0, ax1, ax2]

    def add_labels(self) -> None:
        ax0, ax1, ax2 = self.axs
        ax0.set_ylabel(r"$P_K$", rotation=0, labelpad=14)

        # correlation coefficient
        ax1.set_ylabel("I(V)\n %s" % r"$r^2$", rotation=0, labelpad=14)
        ax1.set_xlabel("Sweep Number")

        # leak subtracted ramp
        ax2.set_ylabel(r"Current (pA)")
        ax2.set_xlabel("Time (ms)")

    def plot_leak_params(
        self, res: LeakSubtractResults, model: AbstractGHKCurrent
    ):
        """Plot fit parameters and correlations"""
        xpos = range(len(res.fitted))

        kwargs = dict(marker='o', ls='-', alpha=0.6, markersize=6)
        cmap = plt.cm.get_cmap('jet')
        def get_clr(i): return cmap((i+1) / len(ion_names))

        # pearson correlations
        self.axs[1].plot(xpos, res.rvals_dataVolts,
                         marker='o', ls='none', label="I-V")

        self.axs[1].plot(xpos, res.rvals_fitVolts, marker='o',
                         fillstyle='none', ls='none', label="Fit-I")

        # get inferred permeabilities
        allPerm = [model.get_all_permeabilities(pars) for pars in res.params]

        # rearrange permeabilities for each ion into separate key-value pairs
        ion_names = allPerm[0].keys()
        allPerm = {name: [P[name] for P in allPerm] for name in ion_names}

        for i, name in enumerate(ion_names):
            perm = [P[name] for P in allPerm]

            self.axs[0].plot(xpos, perm, label=f"$P_{{name}}$",
                             c=get_clr(i), **kwargs)

    def plot(self, res: LeakSubtractResults) -> None:

        down = self.downsample
        subtracted = res.subtracted
        N = int(subtracted.shape[1]/2)

        times = subtracted.index[::down]
        times -= times[0]

        ax2 = self.axs[2]
        ax2.plot(times, subtracted.iloc[::down, :N], lw=2, alpha=0.5, 
                 c='r', label="Original")

        fitted = res.fitted
        line_border = [pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]
        kwargs = dict(lw=2, c='yellow', alpha=0.8,
                      path_effects=line_border, label="Fit")

        if isinstance(fitted, np.ndarray):
            ax2.plot(times, np.transpose(fitted), **kwargs)
        elif isinstance(fitted, list):
            for i, fit in enumerate(fitted):
                times = subtracted.iloc[:, i].dropna().values()
                ax2.plot(times[::down], fit[::down], **kwargs)                                    
        else:
            raise TypeError(f"Fitted data should be a Numpy array or list of numpy arrays, not {type(fitted)}")

        # plot subtracted current
        ax2.plot(times[::5], subtracted.iloc[::5, :N], lw=2, alpha=0.5, 
                 c='lightblue', label="Subtracted")

        self.add_legend(N)
        self.format_axes(N)

    def format_axes(self, N: int) -> None:

        ax0, ax1, ax2 = self.axs

        # scientific notation for y axis ticks
        ax0.ticklabel_format(axis='y', scilimits=(-2, 2))

        # add legend for permeabilities plot
        ax0.legend(loc='center right')

        # xticks for N traces, plus 3 empty ticks for in-plot legends
        xpos = range(N+int((3/8)*N))
        xlabs = list(range(1, N+1))
        xlabs.extend([""]*int((3/8)*N))

        # apply xtick positions and labels
        ax0.set_xticks(xpos)
        ax1.set_xticks(xpos)

        ax0.set_xticklabels([])
        ax1.set_xticklabels(xlabs)

        ax2.locator_params(axis='x', nbins=5)
        ax2.locator_params(axis='y', nbins=5)

        for a in self.axs:
            ybot, ytop = a.get_ylim()
            if ytop == 1:
                a.set_ylim(ybot - 0.1, ytop + 0.1)
            else:
                dy = (ytop - ybot)*0.2
                a.set_ylim(ybot - dy, ytop + dy)

        self.fig.suptitle("Fitting of Linear Voltage Ramps with GHK")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def add_legend(self, N: int) -> None:

        ax1, ax2 = self.axs[1:]

        ax1.legend(loc='center right', framealpha=0.5)

        h, l = ax2.get_legend_handles_labels()
        ax2.legend([h[0], h[N], h[-1]], [l[0], l[N], l[-1]],
                   loc='lower center', ncol=3)

# ---------------------------------------------------------------------------- #
#                     Current-Voltage Analysis and Plotting                    #
# ---------------------------------------------------------------------------- #


class AnalyzeIV(AbstractAnalyzer):
    """
    Find reversal potential, ion permeabilities, and linear fit from instantaneous current.  

    `df_i` = extracted epoch of leak-subtracted current       
    `df_v` = corresponding voltage command  
    `khz` = sampling frequency, in khz  
    `w` = starting from the start of each trace, time window to determine instantaneous current, in ms  
    `Cm` = membrane capacitance; if given a plot of current density will also be produced  
    """

    def __init__(self, data: Recording) -> None:
        self.data = data

    def get_Iinst(self, df_i, df_v, Cm, w: int =5):
        khz = self.data.attrs['khz']

        # select current and voltage in desired window given by `w`
        i_inst = df_i.iloc[:w*khz, :].mean(axis=0).values
        voltages = df_v.iloc[1, :].values

        # current density
        i_inst_cm = i_inst / Cm

        # linear fits of i_inst and i_inst_cm against voltage using Chebyshev
        LinFit_i = np.polyfit(voltages, i_inst, deg=1)
        LinFit_i_cm = np.polyfit(voltages, i_inst_cm, deg=1)

    def run(self):
        return super().run()

    def plot_results(self) -> None:
        return super().plot_results()

    def extract_data(self, key: str) -> None:
        return super().extract_data(key)


class PlotIV(AbstractPlotter):
    def __init__(self, data: Recording, show=False) -> None:
        super().__init__(data, show=show)

    def create_figure(self) -> None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5),
                                constrained_layout=True)
        fig.suptitle("I-V")

        self.fig = fig
        self.axs = axs

    def add_labels(self) -> None:

        # align ylabels to the top edge of the axis
        self.axs[0].set_title("Current (pA)", fontweight='bold', pad=12)
        self.axs[1].set_title("Current\nDensity (pA/pF)",
                              fontweight='bold', pad=12)

        for ax in self.axs:
            # align xlabel to the right edge of the axis
            ax.set_xlabel("Voltage (mV)", ha="right", x=0.98)

    def format_axes(self) -> None:

        for ax in self.axs:
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_position('zero')
                ax.spines[spine].set_alpha(0.5)

            # turn off right and top spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Show ticks in the left and lower axes only
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

    @staticmethod
    def find_Erev(
        params: Dict[str, float], spl: UnivariateSpline,
        v_min=-40, v_max=10
    ) -> float: 
        
        try:
            Erev = [x for x in spl.roots() if v_min < x < v_max][0]
        except (ValueError, RuntimeError):
            # reversal potential from linear fits
            logging.error("Failed to estimate x-intercept of I-V from the root of polynomial splines. Defaulting to inference from parameters of linear Ohmic fit to leak current.")
            
            if 'E_leak' in params and 'g_leak' in params:
                I_zero = -linear_fit['E_leak'] / linear_fit['g_leak']
            else:
                raise KeyError(f"'E_leak' and 'g_leak' were not found in parameters:\n{params}")
        
        return Erev
                
    def add_IV(
        self, ax: plt.Axes, linear_fit: lmfit.Parameters,
        voltages: NDArrayFloat, current: NDArrayFloat,
        Cm: float, GHK_model: AbstractGHKCurrent
    ) -> lmfit.Parameters:
        # unpack linear fit parameters 
        params = linear_fit.valuesdict()
        
        # fit spline to current/voltage
        voltages, current = multi_sort(zip(voltages, current))
        spl = UnivariateSpline(voltages, current)

        Erev = self.find_Erev(linear_fit, spl)
                
        # fit Iinst-V with GHK current equation
        self.E = math.exp(Erev/RTF_CONSTANT)
        ghk_params = GHK_model.fit(voltages, current)
        i_ghk = GHK_model.simulate(ghk_params, voltages)
        
        # absolute permeabilities 
        perms = GHK_model.get_all_permeabilities(ghk_params)
        P_K, P_Na = perms['K'], perms['Na']
        P_K_Na = P_K / P_Na 

        ax.plot(voltages, current, marker='o', ls='none', 
                markersize=6, c='k', label=None)
        ax.plot(voltages, np.polyval(L, V), lw=2, ls='--', 
                c='blue', label="Linear")
        ax.plot(voltages, np.polyval(list(params.values()), voltages), 
               lw=2, ls='--', c='blue', label="Linear")
        ax.plot(voltages, i_ghk, c='r', lw=2, label='GHK')
        ax.plot(voltages, spl(voltages), ls=':', c='g',
                lw=2.5, label="Spline")

        # expand xlimits if necessary
        xlims = list(ax.get_xlim())
        if xlims[1] > 0 or xlims[0] > -50:
            if 0 < xlims[1] < 50: xlims[1] = 50
            elif xlims[0] > -50: xlims[0] = -50
            ax.set_xlim(xlims[0], xlims[1])
        elif xlims[1] < 0:
            xlims[1] = 0
            ax.set_xlim(xlims[0], xlims[1])

        # nbins for yticks
        ax.locator_params(axis='y', nbins=4)

        # information to label in each plot
        s = "\n".join([
            f"$P_{{K}}/P_{{Na}}$ = {P_K_Na:.2f}",
            f"$P_{{K}}$ = {P_K:.1e}",
            f"$P_{{Na}}$ = {P_Na:.1e}",
            f"$E_{{rev}}$ = {Erev:.1f} mV",
            f"$C_m$ = {Cm:d} pF"
        ])
        
        # add text box with the labels
        if xlims[0] < -120:
            ax.text(0.65, 0.2, s, transform=ax.transAxes, fontsize=11, va='top')
        else:
            ax.text(0.75, 0.2, s, transform=ax.transAxes, fontsize=11, va='top')

        logging.info(f"""IV Parameters
            P_K/P_Na = {P_K_Na},
            P_K = {P_K}, P_Na = {P_Na},
            Erev = {Erev}, Cm = {Cm}
            """)
            
        # legend
        ax.legend(loc='upper left', fontsize=11, 
                  bbox_to_anchor=(0, 0.7, 0.5, 0.5))

        return Erev, P_K, P_Na


def IV_analysis(self, , plot_results=False, output=False, pdfs=None):
    """


    ## Output  
    `plot_results` = if True, visualizes I-V and (I/Cm)-V with linear fits, labelled with Cm and permeability values.  
    `output` = if True, returns array of computed values - (reversal, P_K, P_Na, P_Na/P_K, Iinst, Iinst/Cm)  
    `pdfs` = if `plot_results = True` and `pdfs` is not None, then the visualization is saved into the specified pdf multipages object  

    Note - this method is categorically distinct from the rest of the class, yet is part of the class to make use of GHK-related methods. However, the same set of internal/external ion compositions is assumed.    
    """


    if plot_results:

        for i in range(2):


        plot_IV(ax[0], i_inst, LinFit_i)
        Erev, P_K, P_Na = plot_IV(ax[1], i_inst_cm, LinFit_i_cm, return_params=True)

        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if pdfs is not None:
            pdfs.savefig(bbox_inches='tight')

        plt.show()
        plt.close()

    else:
        raise Exception("`plot_results` must be enabled for IV analysis, otherwise parameters won't be returned.")

    if output:
        # create dataframe with columns for instantaneous current and current density
        Iinst_df = pd.DataFrame(data={"pA": i_inst, "pA/pF" : i_inst_cm}, index=voltages)
        IV_params = pd.DataFrame(data={
            "Erev": Erev, "P_K" : P_K, "P_Na" : P_Na, "C_m" : Cm
        })

        return IV_params, Iinst_df

# ---------------------------------------------------------------------------- #
#                            Pseudo-leak subtraction                           #
# ---------------------------------------------------------------------------- #

def PseudoLeakSubtraction(self, original, epochs=None,
                          use_tails=True, tail_threshold=20, method='ohmic', plot_results=False):
    """
    Leak subtraction based on instantaneous current and holding current. Only applicable for activation protocols. 
    `original` = original dataframe 
    `epochs` = dictionary of epochs, {sweep # : [epoch1, epoch2, ...]} 
    `use_tails` = whether to use tail currents (follow immediately after activation)
    `method` = 'ohmic' or 'ghk' 

    Returns leak-subtracted dataframe 
    """
    if epochs is None:
        epochs = self.epochs

    if self.N is None:
        self.N = int(original.shape[1]/2)

    khz = self.khz

    # find index of first pulse with varying voltage
    u = None
    for k, v in epochs.items():

        if u is not None:
            break

        for i, t in enumerate(v):

            # skip if epoch is not in first sweep
            if t not in epochs[1]:
                break

            # voltage at epoch `t` for all sweeps
            volts = original.iloc[(t + 10*khz):(t + 110*khz), self.N:].mean(axis=0)

            # check if difference between consecutive sweeps is at least 5 mV
            if np.all(np.abs(volts.iloc[1:].values - volts.iloc[:-1].values) >= 5):
                u = i
                break

    # voltage and current at onset of first varying-voltage pulse (activation)
    t0 = epochs[1][u]
    # select first 50ms, with 20ms offset
    i_vary = original.iloc[(t0 + 20*khz):(t0 + 70*khz), :]
    v_vary = [int(x/5)*5 for x in i_vary.iloc[10, self.N:]]
    i_vary = i_vary.iloc[:, :self.N].mean(axis=0)

    # voltage and current at holding potential
    t0 = epochs[1][0]
    # select last 50ms, with 100ms offset
    i_hold = original.iloc[(t0 - 150*khz):(t0 - 100*khz), :]
    v_hold = i_hold.iloc[10, self.N:].values
    i_hold = i_hold.iloc[:, :self.N].mean(axis=0) 

    if use_tails:
        # select epoch that acts as upper bound of tail
        t0 = epochs[1][u+2]

        # check change in current in last 250ms of tail, w/ 20ms offset and 5ms avg
        # if change is > tail_threshold, skip tails
        dI_ss = original.iloc[(t0 - 270*khz):(t0 - 20*khz), :self.N].rolling(5*khz).mean().dropna()
        dI_ss = (dI_ss.max(axis=0) - dI_ss.min(axis=0)).abs()

        if (dI_ss <= tail_threshold).all():
            # select last 50ms, with 50ms offset
            i_tails = original.iloc[(t0 - 100*khz):(t0 - 50*khz), :]
            v_tails = i_tails.iloc[10, self.N:].values
            i_tails = i_tails.iloc[:, :self.N].mean(axis=0)

            current = [i_hold, i_vary, i_tails]
            volts = [v_hold, v_vary, v_tails]
        else:
            print("`use_tails=True`, but will not proceed, because change in current over last 500 (+20) ms is greater than 10pA")
            print(dI_ss)

            current = [i_hold, i_vary]
            volts = [v_hold, v_vary]
    else:
        current = [i_hold, i_vary]
        volts = [v_hold, v_vary]

    df_sub = original.copy()
    leak_params = []
    for i in range(self.N):

        i_ = [x.iloc[i] for x in current]
        v_ = [x[i] for x in volts]

        # fit with ohmic leak equation
        if method == 'ohmic':
            self.fit_ohmic_leak(i_, v_)
            leak_i = self.get_leak_current(original.iloc[:, self.N+i], self.popt, mode="ohmic")
        # fit with ghk equation
        elif method in ['double', 'triple']:
            self.fit_ghk_leak(i_, v_, mode=method)
            leak_i = self.get_leak_current(original.iloc[:, self.N+i], self.popt, mode=method)

        # subtract from the entire trace
        df_sub.iloc[:, i] -= leak_i 

        leak_params.append(self.popt)

    self.leak_params = leak_params
    if plot_results:

        f = plt.figure(figsize=(10, 6), constrained_layout=True)
        gs = f.add_gridspec(nrows=2, ncols=3)
        ax1 = f.add_subplot(gs[0, :])
        ax2 = f.add_subplot(gs[1, 0])
        ax3 = f.add_subplot(gs[1, 1])
        ax4 = f.add_subplot(gs[1, 2])
        # axs = [ax1, ax2, ax3, ax4]

        # plot original and usbtracted current time courses
        ax1.set_ylabel("Current (pA)")
        ax1.set_xlabel("Time (ms)")
        ax1.plot(original.iloc[:, :self.N], alpha=0.5, c='gray', lw=1)
        ax1.plot(df_sub.iloc[:, :self.N], c='r', lw=2)

        ax2.set_title(r"$\gamma$ (pS)")
        ax3.set_title(r"$E_{\mathregular{leak}}$ (mV)")
        for a in [ax2, ax3, ax4]:
            a.set_xlabel("Sweep #")

        # plot leak parameters
        for j, ax in enumerate([ax2, ax3]):
            # extract jth parameter for each sweep
            p_j = [x[j] for x in leak_params]
            ax.plot(range(1, self.N + 1), p_j, marker='o', ls='none', c='lightblue')

        # plot current and voltage values used for fitting, along with fitted equations
        for i in range(self.N):
            clr = cmap(i/(self.N - 1))

            i_ = [x.iloc[i] for x in current]
            v_ = [x[i] for x in volts]

            ax4.plot(v_, i_, marker='o', ls='none', c=clr)

            # voltages for interpolation of fitted equations
            v_ = range(min((min(v_), -150)), 50)

            # plot interpolated fits on ax4
            if method == 'ohmic':
                leak_i = self.get_leak_current(v_, leak_params[i], mode="ohmic")
            elif method in ['double', 'triple']:
                leak_i = self.get_leak_current(v_, leak_params[i], mode=method)

            ax4.plot(v_, leak_i, c=clr, lw=1, alpha=0.5, ls='--')

        plt.show()
        plt.close()

    return df_sub
