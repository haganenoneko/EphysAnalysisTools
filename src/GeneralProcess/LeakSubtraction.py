# Copyright (c) 2022 Delbert Yip
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import glob, math, os 

from abc import ABC, abstractmethod 
from typing import Callable, Tuple, List, Dict, Any, Union
import logging 

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages

from scipy.interpolate import UnivariateSpline
from scipy.stats import pearsonr, sem
import lmfit 

from dataclasses import dataclass 

from GeneralProcess.ActivationCurves import multi_sort
from GeneralProcess.Base import NDArrayFloat, CleanlyDropNaNs, Recording, AbstractPlotter

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
        return {"Na" : [10, 110], "K" : [130, 30], "Cl" : [141, 144.6]} 
    
    if ion_set == "HighK":
        return {"Na" : [10, 135], "K" : [130, 5.4], "Cl" : [141, 144.6]} 
    
    if not isinstance(ion_set, dict):
        raise ValueError(f"`ion_set` can only be None, 'HighK', or a Dictionary of ion concentrations, not\n{ion_set}")
    
    if not all(x in ion_set for x in permeant):
        
        logging.warning(f"The following ion set\n{ion_set}\n does not have one or more of permeant ions: {permeant}. Resorting to default: Na, K, Cl.")
        
        permeant = ['Na', 'K', 'Cl']
    
    for ion in permeant:
        
        if ion not in ion_set: 
            raise KeyError(f"Ion set does not contain concentrations of {ion}")

        if len(ion_set[ion]) != 2:
            raise ValueError(f"Ion set does not contain two values (internal, external concentrations) for ion {ion}:\n{ion_set}")
        
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
            method='leastsq', **kwargs):
        
        params = self.set_params()
        
        if len(voltages.shape) > 1 or len(current.shape) > 1: 
            raise ValueError(f"Voltages and current should be one-dimensional, but have shapes {voltages.shape} and {current.shape}, respectively.")
        
        res = lmfit.minimize(self.residual, params, 
                            method=method, **kwargs)
        self.res = res 
        return res

    def __repr__(self) -> str:
        if self.res is None: return "" 
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
            name : conc[0] - conc[1] for name, conc in self.ion_set.items()
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
            if ion not in self.ion_deltas: continue 
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
        return P_K*( (E*K[0] - K[1]) / (Na[1] - E*Na[0]) )
    
    def get_all_permeabilities(self, params: lmfit.Parameters) -> Dict[str, float]:
        
        K = self.ion_set['K']
        Na = self.ion_set['Na']
        E = self.E_RMP
        
        parvals = params.valuesdict()
        parvals['Na'] =  parvals['K']*( (E*K[0] - K[1]) / (Na[1] - E*Na[0]) )
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
        if factor < 1: return data 
        return data.iloc[::factor, :]
        
    @abstractmethod
    def find(self) -> pd.DataFrame:
        return 
    
    def clone_target(
        self, target: Union[pd.Series, pd.DataFrame],
        agg: Callable[[NDArrayFloat], NDArrayFloat]=np.mean, 
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
    
    def find(
        self, min_dV: float=50., 
        rank_metric: Callable[[NDArrayFloat], NDArrayFloat]=None
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
    subtracted: Union[NDArrayFloat, pd.DataFrame]
    
class LeakSubtractor:
    
    def __init__(self, df: pd.DataFrame, model: AbstractCurrentModel) -> None:
        self.df = df 
        self.model = model 
        self.N = int(df.shape[1]/2)

    def subtract(
        self, plotter: Callable[[NDArrayFloat], None]=None,
        fit_kwargs: Dict[str, Any]={}, plotter_kwargs: Dict[str, Any]={}
    ) -> LeakSubtractResults:

        N = self.N 
        subtracted = self.df.values 
        leak_params: List[lmfit.Parameters] = [] 
        
        fitted: List[NDArrayFloat] = [] 
        rvals_fitVolts = np.zeros(self.N)
        rvals_dataVolts = np.zeros(self.N)
        
        for i in range(N):
            sweep = self.df.iloc[:, [i, N+i]].dropna().values 
            
            res = self.model.fit(sweep[:,1], sweep[:,0], **fit_kwargs)
            logging.info(repr(self.model))
            logging.debug(f"Fit status for {i}-th trace: {res.message}")
            
            params = res.params 
            leak_params.append(params)
            
            leak_i = self.model.simulate(params, self.df.iloc[:, N+i])
            fitted.append(leak_i)

            rvals_fitVolts[i] = pearsonr(sweep[:,1], leak_i)[0]
            rvals_dataVolts[i] = pearsonr(sweep[:,1], sweep[:,0])[0]
            
            subtracted.iloc[:,i] -= leak_i 
            
        if plotter is not None:
            plotter(subtracted, **plotter_kwargs)
        
        return LeakSubtractResults(
            fitted, leak_params, rvals_fitVolts, 
            rvals_dataVolts, subtracted
        )
    
    
# ---------------------------------------------------------------------------- #
#                               Plotting methods                               #
# ---------------------------------------------------------------------------- #

class PlotOhmic(AbstractPlotter):
    
    def __init__(
        self, data: Recording, results: LeakSubtractResults,
        save_path: str, show=False, downsample: int=5
    ) -> None:
        self.fig: plt.figure = None 
        self.axs: List[plt.Axes] = None 
        self.downsample = downsample 
        
        self.create_figure()
        self.add_labels()
        self.plot(results)
        self.save(show, save_path, data)
        
    def create_figure(self) -> None:
        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(2, 3)
        
        axs = [gs[0,0], gs[0,1], gs[0,2], gs[1,:]]
        
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
        ax3.plot(times[::down], results.subtracted.iloc[::down,:N], 
                marker='o', markersize=3, markevery=5, ls='none', 
                c='gray', label="Original")
        
        # plot subtracted current 
        ax3.plot(
            times[::down], self.ramp_df.iloc[::down,:N], 
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
                times = subtracted.iloc[::down,i].dropna().index                
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
        show=False, downsample: int=5
    ) -> None:
        self.fig: plt.figure = None 
        self.axs: List[plt.Axes] = None 
        self.downsample = downsample 
        
        self.create_figure()
        self.add_labels()
        self.plot(results)
        self.save(show, save_path, data)
        
    def create_figure(self) -> None:
        fig = plt.figure(figsize=(14, 6))
        gs = f.add_gridspec(2, 7)
        
        # ion permeability values for K 
        ax0 = f.add_subplot(gs[0,:3])
        ax1 = f.add_subplot(gs[1,:3])
        ax2 = f.add_subplot(gs[:,3:])
        
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
        self, res: LeakSubtractResults, model: AbstractGHKCurrent,
        inferred_ion: str='Na'
    ):
        xpos = range(len(res.fitted))
        
        kwargs = dict(marker='o', ls='-', alpha=0.6, markersize=6)
        cmap = plt.cm.get_cmap('jet')
        get_clr = lambda i: cmap((i+1) / len(ion_names))
        
        # pearson correlations         
        self.axs[1].plot(xpos, res.rvals_dataVolts, 
                        marker='o', ls='none', label="I-V")
        
        self.axs[1].plot(xpos, res.rvals_fitVolts, marker='o', 
                fillstyle='none', ls='none', label="Fit-I")
        
        # get inferred permeabilities 
        allPerm = [model.get_all_permeabilities(pars) for pars in res.params]
        
        # rearrange permeabilities for each ion into separate key-value pairs 
        ion_names = allPerm[0].keys()
        allPerm = {name : [P[name] for P in allPerm] for name in ion_names}
                
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
        ax2.plot(times, subtracted.iloc[::down,:N], lw=2, alpha=0.5, 
                c='r', label="Original")
        
        fitted = res.fitted 
        line_border = [pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]
        kwargs = dict(lw=2, c='yellow', alpha=0.8, 
                      path_effects=line_border, label="Fit")
        
        if isinstance(fitted, np.ndarray):
            ax2.plot(times, np.transpose(fitted), **kwargs)
        elif isinstance(fitted, list):
            for i, fit in enumerate(fitted):
                times = subtracted.iloc[:,i].dropna().values()
                ax2.plot(times[::down], fit[::down],**kwargs)                                    
        else:
            raise TypeError(f"Fitted data should be a Numpy array or list of numpy arrays, not {type(fitted)}")
        
        # plot subtracted current 
        ax2.plot(times[::5], subtracted.iloc[::5,:N], lw=2, alpha=0.5, 
                c='lightblue', label="Subtracted")
        
    def format_axes(self, N: int) -> None:
        
        ax0, ax1 = self.axs[:2]
        
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
    
    
    h, l = ax1.get_legend_handles_labels()
    ax1.legend(h[:2], list(set(l)), loc='center right', framealpha=0.5)
                
    h, l = ax2.get_legend_handles_labels()
    ax2.legend([h[0], h[N], h[-1]], [l[0], l[N], l[-1]], loc='lower center', ncol=3)
    ax2.locator_params(axis='x', nbins=5)
    ax2.locator_params(axis='y', nbins=5)
                    
    for a in [ax0, ax1, ax2]:
        ybot, ytop = a.get_ylim()
        if ytop == 1:
            a.set_ylim(ybot - 0.1, ytop + 0.1) 
        else:
            dy = (ytop - ybot)*0.2
            a.set_ylim(ybot - dy, ytop + dy) 
        
    f.suptitle("Fitting of Linear Voltage Ramps with GHK")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    
class leak_subtract():
    def __init__(self, ramp_times, khz=2, epochs=None, residual=False, ion_set=None):
        """
        `ramp_times` = [start, end], index values for voltage ramp  
        `epochs` = times of epochs in the recording  
        `pname` = name of protocol  
        `residual` = whether to allow a residual ohmic leak current with different parameters 
        'ion_set' = dictionary of solution compositions; {ion : [in, out]} for each ion 
        """
        self.res = residual 
        self.ramp_startend = ramp_times 
        self.khz = khz 
        self.epochs = epochs 
        
        self.transformed = False  
        
        # data 
        self.ramp_df = None 
        
        # number of sweeps 
        self.N = None 
        
        # black border to plots of fitted leak current
        self.line_border = [pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]
        
    


    
    def check_quality(self, original, df, show=False):
        """
        Assess quality of leak subtraction.  

        Compute linearity for fit by: 1 - sum(x - mu)^2 / sum(x^2), where `x` is the leak-subtracted data and the horizontal line `y = mu` is the mean of the data, our expected outcome of a perfect subtraction. Current threshold for 'good' subtraction is R^2 > 0.1.    
        
        `original` = original data, pre-subtraction  
        `df` = output of `self.do_leak_subtraction`  
        
        Returns True if quality is acceptable, False otherwise.  
        show = if True, returns an I-V plot for leak ramps + the zero-current region chosen.
        """
        # the first epoch should have zero current
        N = self.N 
            
        # extract leak ramp from un-subtracted data 
        # ramp_df_raw = self.find_ramp(original, return_ramp=True)
        
        E = 0 
        for i in range(N):
            # y_raw = ramp_df_raw.iloc[:,i].dropna().values 
            y_fit = self.ramp_df.iloc[:,i].dropna().values ** 2
            
            E += 1 - np.mean(y_fit / (max(y_fit) - min(y_fit)))
        
        E *= 1/N
        # print(E)
        # exit()
        
        if E > 0.9:
            print("     Leak subtraction is acceptable.\n    Mean R^2 is %.3f. The threshold is > 0.9." % E)
            return True 
        else:
            print("     Leak subtraction is unacceptable.\n      Mean R^2 is %.3f. The threshold is > 0.9." % E)
            return False 
    
    def do_leak_subtraction(self, df, method="ohmic", return_params=False, 
                            plot_results=False, pdfs=None):
        """
        Perform leak subtraction.  
        
        `method` = if `"ohmic"`, fits linear ohmic equation to leakage ramps. If `"ghk"`, fits a GHK equation instead. `ghk` assumes only K+ and Na+ permeate. Support for K+, Na+, and Cl- is available by passing `method=ghk_triple`.  
        `return_params` = whether to return leak-subtraction parameters. For `method=linear`, these are parameters to `ohmic_leak`. If a linear fit fails, or if `method=spline`, returns a UnivariateSpline object.  
        `plot_results` = whether to show parameters and/or leak-subtracted output  
        `pdfs` = multipage PDFs object to append figures to. Only if `plot_results = True`. 
        """
        # finds leakage ramp and saves isolated current/voltage as a class variable
        self.find_ramp(df)
                
        if method == "ohmic":
            out = self.linear_subtraction(df, plot_results=plot_results, pdfs=pdfs)
            self.check_quality(df, out, show=plot_results)
            
        elif "ghk" in method:
            print("     `ghk` selected for linear subtraction method. \n")
            
            # first fit a linear equation to get reversal potential 
            self.linear_subtraction(df, plot_results=False, pdfs=None)
            # set reversal potential to be the mean from linear fit to each trace 
            self.E = math.exp(np.mean([x[1] for x in self.leak_params])/self.RT_F) 
            
            # fit P_K in GHK current equation 
            # permeable to Na, K, and Cl 
            if "triple" in method:
                out = self.ghk_subtraction(df, mode='triple', 
                                        plot_results=plot_results, pdfs=pdfs)
            # permeable to Na and K only 
            else:
                out = self.ghk_subtraction(df, mode='double', 
                                        plot_results=plot_results, pdfs=pdfs)    
            
            self.check_quality(df, out, show=plot_results)
        
        return out 

    def get_leak_params(self):
        return self.leak_params 
    

# ---------------------------------------------------------------------------- #
#                     Current-Voltage Analysis and Plotting                    #
# ---------------------------------------------------------------------------- #
    
def IV_analysis(self, df_i, df_v, Cm, khz=2, w=5, plot_results=False, output=False, pdfs=None):
    """
    Find reversal potential, ion permeabilities, and linear fit from instantaneous current.  
    
    `df_i` = extracted epoch of leak-subtracted current       
    `df_v` = corresponding voltage command  
    `khz` = sampling frequency, in khz  
    `w` = starting from the start of each trace, time window to determine instantaneous current, in ms  
    `Cm` = membrane capacitance; if given a plot of current density will also be produced  
    
    ## Output  
    `plot_results` = if True, visualizes I-V and (I/Cm)-V with linear fits, labelled with Cm and permeability values.  
    `output` = if True, returns array of computed values - (reversal, P_K, P_Na, P_Na/P_K, Iinst, Iinst/Cm)  
    `pdfs` = if `plot_results = True` and `pdfs` is not None, then the visualization is saved into the specified pdf multipages object  
    
    Note - this method is categorically distinct from the rest of the class, yet is part of the class to make use of GHK-related methods. However, the same set of internal/external ion compositions is assumed.    
    """
    
    # select current and voltage in desired window given by `w`
    i_inst = df_i.iloc[:w*khz, :].mean(axis=0).values 
    voltages = df_v.iloc[1, :].values
    
    # current density 
    i_inst_cm = i_inst / Cm 
    
    # linear fits of i_inst and i_inst_cm against voltage using Chebyshev 
    LinFit_i = np.polyfit(voltages, i_inst, deg=1)
    LinFit_i_cm = np.polyfit(voltages, i_inst_cm, deg=1)
            
    if plot_results:
        f, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        f.suptitle("I-V")
        
        for i in range(2):
            for spine in ['left', 'bottom']:
                # move left and bottom spines to the center
                ax[i].spines[spine].set_position('zero')
                # transparency 
                ax[i].spines[spine].set_alpha(0.5)
                
            # turn off right and top spines 
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)

            # Show ticks in the left and lower axes only
            ax[i].xaxis.set_ticks_position('bottom')
            ax[i].yaxis.set_ticks_position('left')
            
            # align xlabel to the right edge of the axis 
            ax[i].set_xlabel("Voltage (mV)", ha="right", x=0.98)
            
        # align ylabels to the top edge of the axis
        ax[0].set_title("Current (pA)", fontweight='bold', pad=12)
        ax[1].set_title("Current\nDensity (pA/pF)", fontweight='bold', pad=12)
        
        def plot_IV(A, I, L, V=voltages, return_params=False):
            """
            A = axes 
            I = current 
            L = linear fit parameters 
            G = GHK current 
            S = spline 
            """
            
            # fit spline to current/voltage
            V, I = multi_sort(zip(V, I))
            spl = UnivariateSpline(V, I)
            
            try:
                Erev = [x for x in spl.roots() if -40 < x < 10][0]
            except:
                # reversal potential from linear fits 
                print(" Failed to get Erev as root from polynomial spline.")
                Erev = -L[0]/L[1] 
            
            # fit Iinst-V with GHK current equation
            self.E = math.exp(Erev/self.RT_F)
            self.fit_ghk_leak(I, V)
            # simulate GHK current 
            G = self.get_leak_current(V, self.popt, mode='double')
                    
            # absolute and relative permeabilities
            P_K = self.popt[0]
            P_Na = self.ghk_get_permeability([self.popt])[0]
            P_K_Na = P_K/P_Na
            
            # information to label in each plot
            s = (r"$P_{K}/P_{Na}$ = %.2f" % P_K_Na) + "\n" + \
                (r"$P_{K}$ = %.1e" % P_K) + "\n" + (r"$P_{Na}$ = %.1e" % P_Na) + "\n" + \
                (r"$E_{rev}$ = %.1f mV" % Erev) + "\n" + (r"$C_m$ = %d pF" % Cm)     
            
            A.plot(V, I, marker='o', ls='none', markersize=6, c='k', label=None)
            A.plot(V, np.polyval(L, V), lw=2, ls='--', c='blue', label="Linear")
            A.plot(V, G, c='r', lw=2, label='GHK')
            A.plot(V, spl(V), ls=':', c='g', lw=2.5, label="Spline")
            
            # expand xlimits if necessary 
            xlims = list(A.get_xlim())
            if xlims[1] > 0 or xlims[0] > -50:
                if 0 < xlims[1] < 50: 
                    xlims[1] = 50
                elif xlims[0] > -50:
                    xlims[0] = -50
                A.set_xlim(xlims[0], xlims[1])
            elif xlims[1] < 0:
                xlims[1] = 0
                A.set_xlim(xlims[0], xlims[1])
            
            # nbins for yticks 
            A.locator_params(axis='y', nbins=4)
            
            # add text box with the labels 
            if xlims[0] < -120:
                A.text(0.65, 0.2, s, transform=A.transAxes, fontsize=11, va='top')
            else:
                A.text(0.75, 0.2, s, transform=A.transAxes, fontsize=11, va='top')
            
            print("IV parameters:",
                "\n PK/PNa$ = ", P_K_Na, "\n PK = ", P_K, 
                "\n PNa = ", P_Na, "\n Erev = ", Erev, 
                "\n Cm = ", Cm)
            
            # legend 
            A.legend(loc='upper left', fontsize=11, bbox_to_anchor=(0, 0.7, 0.5, 0.5))
            
            if return_params:
                return Erev, P_K, P_Na
            
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
        Iinst_df = pd.DataFrame(data={"pA" : i_inst, "pA/pF" : i_inst_cm}, index=voltages)
        IV_params = pd.DataFrame(data={
            "Erev" : Erev, "P_K" : P_K, "P_Na" : P_Na, "C_m" : Cm
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
    i_hold = i_hold.iloc[:,:self.N].mean(axis=0) 
            
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
        
        #fit with ohmic leak equation 
        if method == 'ohmic':
            self.fit_ohmic_leak(i_, v_)  
            leak_i = self.get_leak_current(original.iloc[:,self.N+i], self.popt, mode="ohmic")
        #fit with ghk equation 
        elif method in ['double', 'triple']:
            self.fit_ghk_leak(i_, v_, mode=method)
            leak_i = self.get_leak_current(original.iloc[:,self.N+i], self.popt, mode=method)
        
        # subtract from the entire trace 
        df_sub.iloc[:,i] -= leak_i 
        
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
    