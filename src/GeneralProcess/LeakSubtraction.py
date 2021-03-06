# Copyright (c) 2022 Delbert Yip
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import math
import re
import lmfit
import logging

from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any, Union, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import pearsonr

from dataclasses import dataclass

from GeneralProcess.Base import KwDict, NDArrayFloat, CleanlyDropNaNs, UnsupportedError, multi_sort
from GeneralProcess.Interfaces import Recording, AbstractPlotter, AbstractAnalyzer

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
        self, ion_set: Dict[str, List[List[float]]], RMP: float = 0.
    ) -> None:

        self.ion_set = ion_set
        self.ion_deltas()

        self._RMP = RMP
        self._eRMP = math.exp(RMP / RTF_CONSTANT)

    # --------------- Reversal potential and thermodynamic exponent -------------- #
    @property
    def rmp(self):
        return self._RMP

    @rmp.setter
    def rmp(self, new_rmp: float):
        self._RMP = new_rmp
        try:
            self._eRMP = math.exp(new_rmp / RTF_CONSTANT)
        except ValueError:
            raise ValueError(
                f"Failed to compute thermodynamic exponent with given value of resting membrane potential: < {new_rmp} >")

    @rmp.getter
    def rmp(self) -> float: return self._RMP

    def ion_deltas(self) -> None:
        """Compute difference (internal - external) concentrations for each ion"""
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
        parvals = params.valuesdict()
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
        E = self._eRMP

        return P_K*((E*K[0] - K[1]) / (Na[1] - E*Na[0]))

    def get_all_permeabilities(
        self, params: lmfit.Parameters
    ) -> Dict[str, float]:

        K = self.ion_set['K']
        Na = self.ion_set['Na']
        E = self._eRMP

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
        E = self._eRMP

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
    def _downsample(data: pd.DataFrame, factor: int) -> pd.DataFrame:
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
        raise NotImplementedError(
            "VaryingStepLeakLocator is not implemented yet")


class VaryingStepLeakLocator(LocateLeak):
    """
    Finds target area of protocols where voltage differs
    between traces, accoridng to some threshold, and
    if multiple such targets are identified, returns the
    most varying region.
    """

    def __init__(
        self, df: pd.DataFrame, epoch_times: List[List[int]], khz: int
    ) -> None:
        self._df = df
        self._khz = khz
        self._epochs = epoch_times

        self._N = int(df.shape[1]/2)

    @staticmethod
    def _validate_method(mode: str, first_length: int) -> str:
        """Check that `mode` specifies a supported method of selecting epoch for leak subtraction

        :param mode: method of selecting epoch for leak subtraction
        :type mode: str
        :param first_length: number of epochs in first sweep
        :type first_length: int
        :raises UnsupportedError: if `mode` is unsupported
        :raises ValueError: if a numeric `mode` is given, but exceeds `first_length`
        :return: supported `mode`
        :rtype: str
        """

        if mode in ['first', 'last', 'min', 'max']:
            return mode

        # more general pattern
        # mtch = re.search(r"([a-z]+)([1-9]+)", mode)

        mtch = re.search(r"(n|top)([1-9]+)", mode)

        if not mtch:
            raise UnsupportedError(
                mode, ['first', 'last', 'min', 'max', 'num#', 'top#'])

        # select the n-th epoch
        if mtch.group(0) == 'n':
            if int(mtch.group(1)) >= first_length:
                raise ValueError(
                    f"An integer argument to `mode` indicates that the `mode`-th epoch should be selected for leak subtraction, but only {first_length} epochs are present in the first sweep.")

            return mode

        # select the top n epochs with largest voltage variation
        if mtch.group(0) == 'top':
            n = int(mtch.group(1))
            if n >= first_length:
                logging.warning(
                    f"`{mode}` selects `{n}` with the greatest voltage variation between sweeps, but the first sweep only contains {first_length} epochs. `{mode}` will thus be truncated to {first_length}")

                return f"top{first_length}"

            return mode

    @staticmethod
    def _validate_found_epochs(inds: List[int], dvs: List[float]) -> bool:

        try:
            assert len(inds) == len(dvs)
        except AssertionError:
            raise ValueError(
                f"Unequal number of identified epochs and corresponding maximum voltage differences:\nEpoch indices: {inds:>8}\nMaximum voltage differences: {dvs:>8}")

        for i, ind in enumerate(inds):
            try:
                assert isinstance(ind, int)
            except AssertionError:
                raise TypeError(f"{i}-th epoch is not an integer: {ind}")

            try:
                assert isinstance(dvs[i], float)
            except AssertionError:
                raise TypeError(
                    f"{i}-th maximum voltage difference is not a float: {dvs[i]}")

        return True

    @staticmethod
    def _select_step(
        inds: List[int], dvs: List[float], mode: str
    ) -> Tuple[int, float]:

        if len(inds) == 1 or mode == 'first':
            return inds[0], dvs[0]
        elif mode == 'last':
            return inds[-1], dvs[-1]
        elif mode == 'min':
            return inds[np.argmin(dvs)], min(dvs)
        elif mode == 'max':
            return inds[np.argmax(dvs)], max(dvs)

        n = int(mode[1:])
        if 'top' in mode:
            dvs, inds = multi_sort([dvs, inds])
            return inds[:n], dvs[:n]
        else:
            return inds[n], dvs[n]

    def _find_step(
        self, w: int = 10, dt: int = 100, dv_min: float = 5.,
        n_min: int = 2, mode: str = 'first'
    ) -> Union[int, List[int]]:
        """Find index of epoch that varies in voltage between sweeps.

        :param w: initial offset (in units of `1/self._khz`) added to each epoch onset times when selecting intervals for sweep-to-sweep comparison, defaults to 10
        :type w: int, optional
        :param dt: duration (in units of `1/self._khz`) of intervals selected for sweep-to-sweep comparison, defaults to 100
        :type dt: int, optional
        :param dv_min: minimum difference in voltage between `(i+1)`-th and `i`-th sweeps at the selected epoch, defaults to 5.
        :type dv_min: float, optional
        :param n_min: minimum number of sweeps in a recording for which the difference in voltage at the selected epoch exceeds `dv_min`, defaults to None
        :type n_min: int, optional
        :param mode: how epochs will be selected. See `_select_step()` for details, defaults to 'first'
        :type mode: str, optional
        :raises UnsupportedError: if `mode` is not supported. See `_select_step()` for details
        :raises RuntimeError: if epochs that are found via the `dv_min` condition do not pass type-based validation. See `_validate_found_epochs()` for details.
        :return: index of selected epoch
        :rtype: int
        """

        df = self._df
        epochs = self._epochs
        n_min = min(n_min, self._N)

        # index of first sweep
        first = min(list(epochs.keys()))

        mode = self._validate_method(mode, len(epochs[first]))

        # endpoints for selections
        dt = (dt + w)*self._khz
        w *= self._khz

        # index of first pulse with varying voltage
        target = []
        target_dv = []

        for sweep in epochs.values():
            if target and mode == 'first':
                break

            for i, t in enumerate(sweep):
                if t not in epochs[first]:
                    break

                volts = df.iloc[(t + w): (t + dt), self._N:]\
                    .mean(axis=0).dropna()

                if volts.shape[0] < n_min:
                    continue

                dvolts = (volts.iloc[1:] - volts.iloc[:-1]).abs()

                if (dvolts >= dv_min).all():
                    target.append(i)
                    target_dv.append(dvolts.max())

                    if mode == 'first':
                        break

        try:
            self._validate_found_epochs(target, target_dv)
        except (ValueError, TypeError) as e:
            logging.error(e)
            raise RuntimeError(
                f"Failed to find suitable epochs for pseudoleak subtraction.")

        # select epoch according to 'mode'
        target, target_dv = self._select_step(target, target_dv, mode)
        logging.info(
            f"The {target}-th epoch was selected for pseudo-leak subtraction. Between sweeps, the maximum difference in voltage at the {target}-th epoch is {target_dv}")

        return target

    def _extract_current_voltage(
        self, ind: int, w: int = 10, dt: int = 50
    ) -> NDArrayFloat:
        """Extract voltages and (time average of) current at the `ind`-th epoch

        :param ind: index of epoch
        :type ind: int
        :param w: offset to add to the epoch onset time when selecting interval of data 
        :type w: int
        :param dt: duration of interval to select
        :type dt: int
        :raises IndexError: voltages are taken as the 10-th row of the selection.
        :return: current and voltages at the `ind`-th epoch as `numpy` arrays
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """

        # endpoints for selections
        dt = (w + dt) * self._khz
        w *= self._khz

        t0 = self._epochs[1][ind]
        df_sel = self._df.iloc[(t0 + w):(t0 + dt), :]

        # round voltages to nearest 5
        # current = df_sel.iloc[:, :self._N]
        # voltage = (df_sel.iloc[:, self._N:] / 5).round().astype(int) * 5
        # return voltage, current

        return df_sel.values

    def find(
        self, find_kw: KwDict = {}, extract_kw: KwDict = {},
        agg_func: Callable[[NDArrayFloat], float] = np.mean
    ) -> pd.DataFrame:
        """Find current/voltage to use for leak subtraction from conditions on epochs in the recording protocol

        :param find_kw: keyword arguments to `_find_step`. The `mode` keyword argument determines how epochs are chosen. See `_find_step` and `validate_found_epochs` for details, defaults to {}
        :type find_kw: KwDict, optional
        :param extract_kw: keyword arguments to `_extract_current_voltage`, which extracts a selection of the original data. The beginning and width of the selection can be controlled by specifying keyword arguments `w` and `dt`, defaults to {}
        :type extract_kw: KwDict, optional
        :param agg_func: function that converts `pd.DataFrame`s from `_extract_current_voltage` to scalars, defaults to np.mean
        :type agg_func: Callable[[NDArrayFloat], float], optional
        :raises e: if aggregation fails with `agg_func` and additionally fails upon using the fallback `np.mean`
        :return: `pd.DataFrame` with voltage and current in the first and second columns, respectively
        :rtype: pd.DataFrame
        """
        target = self._find_step(**find_kw)

        if isinstance(target, list):
            inds = [0]
            inds.extend(target)
        else:
            inds = [0, target]

        leak_data = np.zeros((len(inds), 2*self._N))
        for i in inds:
            X = self._extract_current_voltage(i, **extract_kw)

            try:
                leak_data[i, :] = agg_func(X, axis=0)
            except (RuntimeError, ValueError, TypeError):
                logging.warning(
                    f"Failed to apply provided function `{agg_func}` to aggregate the data. Falling back to `np.mean`")

            try:
                leak_data[i, :] = np.mean(X, axis=0)
            except (RuntimeError, ValueError, TypeError) as e:
                raise e

        # swap voltage and current columns
        out = pd.DataFrame(leak_data)

        new_order = np.arange(0, out.shape[1], dtype=int)
        new_order[:self._N] += self._N
        new_order[self._N:] -= self._N
        return out.iloc[:, new_order]

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

    def set_method(
        self, method: str, RMP: float = None,
        ion_set: Dict[str, List[float]] = None,
        permeant_ions=['Na', 'K', 'Cl']
    ) -> None:
        """Set method (aka 'model') to use for fitting leak current/voltage.

        :param method: one of 'ohmic' or 'ghk#', where # is an integer between 1 and 3, inclusive. 
        :type method: str
        :param RMP: resting membrane potential. Must be specified to use a GHK model. Optional otherwise. Defaults to `None`
        :type RMP: float, optional
        :param ion_set: dictionary of `[internal, external]` ion concentrations with ion names as keys, defaults to None
        :type ion_set: Dict[str, List[float]], optional
        :param permeant_ions: list of permeant ions. That is, the ions that will be used for fitting GHK models, defaults to ['Na', 'K', 'Cl']
        :type permeant_ions: list, optional
        :raises ValueError: if `method` is unsupported
        :raises ValueError: if `RMP` is not specified for a GHK model
        """

        if method not in ['ohmic', 'ghk2', 'ghk3', 'ghk']:
            raise UnsupportedError(method, ['ohmic', 'ghk2', 'ghk3', 'ghk'])

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
        """Set method used to find leak current/voltage

        :param locator: one of 'ramp', 'GUI', or 'step'
        :type locator: str
        :raises UnsupportedError: unsupported `locator` argument
        :raises AttributeError: if `ramp` is specified, but the data (`Recording` object) does not have the `ramp_startend` attribute
        """

        if locator not in ['ramp', 'GUI', 'step']:
            raise UnsupportedError(locator, ['ramp', 'GUI', 'step'])

        if locator == 'ramp':
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

            res = self.model.fit(sweep[:, 1], sweep[:, 0], **fit_kwargs)
            logging.info(repr(self.model))
            logging.info(f"Fit status for {i}-th trace: {res.message}")

            leak_params.append(res.params)

            leak_i = self.model.simulate(res.params, self.df.iloc[:, N+i])
            fitted.append(leak_i)

            rvals_fitVolts[i] = pearsonr(sweep[:, 1], leak_i)[0]
            rvals_dataVolts[i] = pearsonr(sweep[:, 1], sweep[:, 0])[0]

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
            model_kwargs: dict = {}, locator_kwargs: dict = {},
            find_kwargs: dict = {}, fit_kwargs: dict = {},
            plot_kwargs: dict = {}, check_kwargs: dict = {}
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
        save_path: str, show=False, downsample: int = 5
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

        axs = [gs[0, 0], gs[0, 1], gs[0, 2], gs[1, :]]

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
            raise TypeError(
                f"Fitted data should be a Numpy array or list of numpy arrays, not {type(fitted)}")

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
        show=False, downsample: int = 5
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
            raise TypeError(
                f"Fitted data should be a Numpy array or list of numpy arrays, not {type(fitted)}")

        # plot subtracted current
        ax2.plot(times[::5], subtracted.iloc[::5, :N], lw=2, alpha=0.5,
                 c='lightblue', label="Subtracted")

        self.add_legend(N)
        self.format_axes(N)
