# Copyright (c) 2022 Delbert Yip
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import glob, math, os 

from abc import ABC, abstractmethod 
from typing import Generator, List, Dict, NamedTuple, Union
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

from GeneralProcess.ActivationCurves import multi_sort
from GeneralProcess.Base import NDArrayFloat

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
    
class DoubleGHKCurrent(AbstractGHKCurrent):
    
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
        
class TripleGHKCurrent(AbstractGHKCurrent):
    
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
        

    
    
    
            
    

    def apply_subtraction(self, leak_params):
        """
        Apply linear or GHK subtraction 
        
        Returns:
        - subtracted = leak-subtracted current 
        - fitted_ramps = fit of leak equation to leak ramps 
        - r_values = pearson coefficients for fit to leak ramps 
        """               
        if self.ramp_df is None:
            raise Exception("Tried calling `apply_subtraction` before `self.ramp_df` defined.")
        
        # subtract from just the ramp, and plot the fitted leak current; we do this separately from subtracting the entire recording for efficiency when we don't require visualization 
        # compute current from fitted parameters for the voltage ramp
        fitted_ramps = [] 
        # compute pearson coefficient for each fitted ramp 
        r_values = [] 
                
        # subtracted ramps 
        subtracted = self.ramp_df.copy() 
        # number of sweeps 
        N = self.N 
        
        for i in range(N):
            y = self.ramp_df.iloc[:,N+i].dropna().values 
            
            fitted_ramps.append(
                self.get_leak_current(y, params=leak_params[i])
            )
            r_values.append(pearsonr(y, fitted_ramps[i])[0]) 
            
            # subtract from leak ramps 
            if subtracted.shape[0] != len(y):
                subtracted.iloc[:len(y),i] = self.ramp_df.iloc[:len(y), i] - fitted_ramps[i]                 
            else:
                subtracted.iloc[:,i] = self.ramp_df.iloc[:,i] - fitted_ramps[i]      
        
        return subtracted, fitted_ramps, r_values 
    
    def linear_subtraction(self, df, plot_results=False, pdfs=None):
        """
        Fit linear ohmic equation to leak ramps.  
        `df` = original input dataframe  
        `plot_results` = if True, shows leak parameters and subtracted output. 
        """
        if self.ramp_df is None:
            # print("     Cannot perform leak subtraction without `ramp_df`. Check arguments to `do_leak_subtraction.")
            raise Exception(" Cannot perform leak subtraction without `ramp_df`. Check arguments to `do_leak_subtraction.`") 
        
        ramp_arr = self.ramp_df.values    # convert dataframe to np array 
        N = self.N
        tmp = df.copy() 
                
        leak_params = []    
        for i in range(N):            
            #fit leak ramp with ohmic leak equation 
            if self.transformed:
                y = self.ramp_df.iloc[:,[i, N+i]].dropna().values 
                self.fit_ohmic_leak(y[:,0], y[:,1])
            else:
                self.fit_ohmic_leak(ramp_arr[:,i], ramp_arr[:,N+i])  
            
            # subtract from the entire trace 
            leak_i = self.get_leak_current(df.iloc[:,N+i].values, self.popt)
            tmp.iloc[:,i] -= leak_i 
                        
            leak_params.append(self.popt) 
        
        self.leak_params = leak_params
        if plot_results:
            self.PlotOhmic(ramp_arr, pdfs=pdfs)
        
        return tmp 
    
    def ghk_subtraction(self, df, mode="double", plot_results=False, pdfs=None):
        """
        Fit nonlinear GHK equation to leak ramps.  
        `df` = original input dataframe  
        `mode` = "double" for permeable Na and K, "triple" for Na/K/Cl  
        `plot_results` = if True, shows leak parameters and subtracted output. 
        """
        if self.ramp_df is None:
            raise Exception(" Cannot perform leak subtraction without `ramp_df`. Check arguments to `do_leak_subtraction.`") 
        
        if mode not in ["double", "triple"]:
            raise Exception("   In call to `ghk_subtraction`, `mode = %s` was passed, but only `double` and `triple` are accepted." % mode)
        
        # reset leak parameters, if any 
        self.leak_params = None 
        # convert dataframe to np array 
        ramp_arr = self.ramp_df.values    
        
        N = self.N 
        tmp = df.copy() 
        
        leak_params = []         
        for i in range(N):            
            #fit leak ramp with ohmic leak equation 
            if self.transformed:
                y = self.ramp_df.iloc[:,[i, N+i]].dropna().values 
                self.fit_ghk_leak(y[:,0], y[:,1], mode=mode)
            else:
                self.fit_ghk_leak(ramp_arr[:,i], ramp_arr[:,N+i], mode=mode)  
    
            leak_i = self.get_leak_current(df.iloc[:,N+i].values.tolist(), 
                                            self.popt, mode=mode)
            tmp.iloc[:,i] -= leak_i 

            leak_params.append(self.popt) 
        
        self.leak_params = leak_params
        if plot_results:
            self.PlotGHK(ramp_arr, pdfs=pdfs)
        
        return tmp
    
    def PlotOhmic(self, ramp_arr, save_path=None, pdfs=None):
        """
        Plot parameters, fit, and subtracted current from using Ohmic equation
        `ramp_arr` = self.ramp_df in array type 
        `save_path` = path to save figures to
        `pdfs` = PDF object
        """
        if self.ramp_df is None:
            raise Exception("`PlotGHK` called before `self.ramp_df` set.")
        
        if self.leak_params is None:
            raise Exception("`PlotGHK` called before leak parameters set.")
        
        if self.N is None:
            N = int(self.df.shape[1]/2)
        else:
            N = self.N 
            
        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(2, 3)
        
        ax0 = fig.add_subplot(gs[0,0])
        ax0.set_title(r"$\gamma_{\mathregular{leak}}$ (pS)")
        # ax0.set_ylabel("pS", rotation=0, labelpad=18)
        
        ax1 = fig.add_subplot(gs[0,1])
        ax1.set_title(r"$E_{\mathregular{leak}}$ (mV)")
        # ax1.set_ylabel("mV", rotation=0, labelpad=10)
        
        ax2 = fig.add_subplot(gs[0,2])
        ax2.set_title(r"$r^2$")
        
        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_ylabel(r"Current (pA)")
                    
        times = self.ramp_df.index
        times -= times[0] 
        
        # plot pre-subtracted current 
        ax3.plot(times, self.ramp_df.iloc[:,:N], marker='o', 
                markersize=3, markevery=5, ls='none', 
                c='gray', label="Original")
        
        ax3.set_xlabel("Time (ms)")
        
        # compute Pearson coefficient for recorded current vs. voltage ramp (NOT the fit!) 
        # print(transformed)
        if self.transformed:
            r_values = [] 
            # select ith leak ramp and remove NaNs
            for i in range(N):
                y = self.ramp_df.iloc[:,[i, N+i]].dropna().values 
                r_values.append( pearsonr(y[:,1], y[:,0])[0] )
        else:
            r_values = [pearsonr(ramp_arr[:,N+i], ramp_arr[:,i])[0] for i in range(N)] 
        
        # appearance of plotting for fit parameters 
        FitKwargs = dict(marker='o', mec='k', ls='none', markersize=6)
        
        ax0.plot(range(N), [p[0] for p in self.leak_params], c='k', **FitKwargs)
        ax1.plot(range(N), [p[1] for p in self.leak_params], c='k', **FitKwargs)
        ax2.plot(range(N), r_values, c='orange', label="I-V", **FitKwargs)
                    
        # apply subtraction
        # subtracted = leak-subtracted current 
        # fitted_ramps = fit of leak equation to leak ramps 
        # r_values = pearson coefficients for fit to leak ramps 
        subtracted, fitted_ramps, r_values = self.apply_subtraction(self.leak_params)  
        
        # replace ramp current with leak-subtracted values 
        self.ramp_df = subtracted 
        
        # plot Ohmic fit on top of leak ramp currents 
        try:
            ax3.plot(
                times, np.transpose(fitted_ramps), 
                lw=1.5, c='blue', ls=':',
                # path_effects=self.line_border,
                label="Fit")
        except:
            for i in range(len(fitted_ramps)):
                y = fitted_ramps[i] 
                times =self.ramp_df.iloc[:,i].dropna().index 
                
                if i == 0:
                    ax3.plot(times, y, lw=2, c='yellow', 
                        path_effects=self.line_border,
                        label="Fit")
                else:
                    ax3.plot(times, y, lw=2, c='yellow', 
                        path_effects=self.line_border,
                        label=None)
        
        # r_values = [pearsonr(ramp_arr[:,N+i], fitted_ramps[i])[0] for i in range(N)]
        ax2.plot(range(N), r_values, c='k', label="Fit-I", **FitKwargs)
        
        h, l = ax2.get_legend_handles_labels()
        ax2.legend(h[:2], list(set(l)), loc='upper right', bbox_to_anchor=[1.65, 1])
        
        ax3.plot(times[::5], self.ramp_df.iloc[::5,:N], 
                lw=2, c='r', label="Subtracted")
        ax3.set_xlabel("Time (ms)")
                    
        h, l = ax3.get_legend_handles_labels()
        ax3.legend([h[0], h[N], h[-1]], [l[0], l[N], l[-1]], 
                loc='upper right', bbox_to_anchor=[1.19, 1])
        ax3.locator_params(axis='x', nbins=5)
        ax3.locator_params(axis='y', nbins=5)
        
        # if more than 6 traces, make every 2nd label empty
        if N > 6:
            xlabs = [""]*N 
            for j in range(0, N, 2):
                xlabs[j] = j+1 
        else:
            xlabs = range(1, N+1)    
        
        # tick appearance
        for a in [ax0, ax1, ax2]:
            a.set_xticks(range(N))
            a.set_xticklabels(xlabs)
            a.set_xlabel("Sweep #")            
                        
        for a in [ax0, ax1, ax2, ax3]:
            ybot, ytop = a.get_ylim()
            if ytop == 1:
                a.set_ylim(ybot - 0.05, ytop + 0.05) 
            else:
                dy = (ytop - ybot)*0.05                 
                a.set_ylim(ybot - dy, ytop + dy) 
            
        fig.suptitle("Fitting of Linear Voltage Ramps")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if pdfs is not None:
            pdfs.savefig(bbox_inches='tight')
            
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        plt.show()
        plt.close() 
    
    def PlotGHK(self, ramp_arr, mode='double', pdfs=None):
        """
        Plot parameters, fit, and subtracted current from using GHK equation
        `ramp_arr` = self.ramp_df in array type 
        `pdfs` = PDF object
        """
        if self.ramp_df is None:
            raise Exception("`PlotGHK` called before `self.ramp_df` set.")
        
        if self.leak_params is None:
            raise Exception("`PlotGHK` called before leak parameters set.")
        
        if self.N is None:
            N = int(self.df.shape[1]/2)
        else:
            N = self.N 
        
        f = plt.figure(figsize=(14, 6))
        gs = f.add_gridspec(2, 7)
        
        # ion permeability values for K 
        ax0 = f.add_subplot(gs[0,:3])
        ax0.set_ylabel(r"$P_K$", rotation=0, labelpad=14)
                    
        # correlation coefficient 
        ax1 = f.add_subplot(gs[1,:3])
        ax1.set_ylabel("I(V)\n %s" % r"$r^2$", rotation=0, labelpad=14)
        ax1.set_xlabel("Sweep Number")
        
        # leak subtracted ramp 
        ax2 = f.add_subplot(gs[:,3:])
        ax2.set_ylabel(r"Current (pA)")
        ax2.set_xlabel("Time (ms)")
                    
        times = self.ramp_df.index
        ax2.plot(times, self.ramp_df.iloc[:,:N], lw=2, alpha=0.5, c='r', label="Original")
        
        # compute Pearson coefficient for recorded current vs. voltage ramp (NOT the fit!) 
        # print(transformed)
        if self.transformed:
            r_values = [] 
            # select ith leak ramp and remove NaNs
            for i in range(N):
                y = self.ramp_df.iloc[:,[i, N+i]].dropna().values 
                r_values.append( pearsonr(y[:,1], y[:,0])[0] )
        else:
            r_values = [pearsonr(ramp_arr[:,N+i], ramp_arr[:,i])[0] for i in range(N)] 
        ax1.plot(range(N), r_values, marker='o', ls='none', label="I-V")
        
        # plot permeability for inferred permeabilities on separate y-axis
        ax0_inf = ax0.twinx()
        # get inferred permeabilities 
        P_inf = self.ghk_get_permeability(self.leak_params)
        
        if mode == "double":
            # plot permeability for K
            ax0.plot(range(N), self.leak_params, marker='o', ls='-', alpha=0.6, c='r',
                    markersize=6, label=r"$K^{+}$")  
            # ax0.axhline(np.mean(self.leak_params), c='r', lw=2, alpha=0.5, label=None)
        
            ax0_inf.set_ylabel(r"$P_{Na}$", rotation=0, labelpad=14)
        
            # plot permeability for Na on a separate y-axis 
            ax0_inf.plot(range(N), P_inf, marker='o', ls='-', alpha=0.6, 
                    c='lightblue', markersize=6, label=r"$Na^{+}$")
            # ax0.axhline(np.mean(P_inf), c='r', lw=2, alpha=0.5, label=None)

        else:
            P_K = [x[0] for x in self.leak_params]
            P_Na = [x[0] for x in P_inf]
            P_Cl = [x[1] for x in P_inf]
            
            # plot permeability for K
            ax0.plot(range(N), P_K, marker='o', ls='-', alpha=0.6, c='r', markersize=6, label=r"$K^{+}$")  
            # ax0.axhline(np.mean(P_K), c='r', lw=2, alpha=0.5, label=None)
            
            ax0_inf.set_ylabel(r"$P_{Na}$" + "\n" + r"$P_{Cl}$", rotation=0, labelpad=14)
            
            ax0_inf.plot(range(N), P_Na, marker='o', ls='-', alpha=0.6, c='lightblue', markersize=6, label=r"$Na^{+}$")
            # ax0_inf.axhline(np.mean(P_Na), c='lightblue', lw=2, alpha=0.5, label=None)
            
            ax0_inf.plot(range(N), P_Cl, marker='o', ls='-', alpha=0.6, c='yellow', markersize=6, label=r"$Cl^{-}$")
            # ax0_inf.axhline(np.mean(P_Cl), c='y', lw=2, alpha=0.5, label=None)
                                            
        # scientific notation for y axis ticks 
        ax0.ticklabel_format(axis='y', scilimits=(-2, 2))
        ax0_inf.ticklabel_format(axis='y', scilimits=(-2, 2))
        
        # add legend for permeabilities plot 
        h0, l0 = ax0.get_legend_handles_labels()
        h0_inf, l0_inf = ax0_inf.get_legend_handles_labels()
        ax0.legend(h0+h0_inf, l0+l0_inf, loc='center right')
            
        # subtracted = leak-subtracted current 
        # fitted_ramps = fit of leak equation to leak ramps 
        # r_values = pearson coefficients for fit to leak ramps 
        subtracted, fitted_ramps, r_values = self.apply_subtraction(self.leak_params)
            
        # pearson correlation between fit and data 
        ax1.plot(range(N), r_values, marker='o', fillstyle='none', ls='none', label="Fit-I")
        
        # xticks for N traces, plus 3 empty ticks for in-plot legends
        xpos = range(N+int((3/8)*N))
        xlabs = list(range(1, N+1))
        xlabs.extend([""]*int((3/8)*N))
        
        # apply xtick positions and labels
        ax0.set_xticks(xpos)
        ax0_inf.set_xticks(xpos)
        ax1.set_xticks(xpos)
        
        ax0.set_xticklabels([])
        ax0_inf.set_xticklabels([])            
        ax1.set_xticklabels(xlabs)
        
        # volts = self.ramp_df.iloc[:, N+i].dropna()                
        times = self.ramp_df.iloc[:,i].dropna().index 
        try:
            ax2.plot(times, np.transpose(fitted_ramps), 
                lw=2, c='yellow', alpha=0.8, path_effects=self.line_border, label="Fit")
        except:
            for i in range(len(fitted_ramps)):
                y = fitted_ramps[i] 
                ax2.plot(times, y, lw=2, c='yellow', alpha=0.8, 
                        path_effects=self.line_border, label="Fit")                                    
                
        # plot subtracted current 
        ax2.plot(times[::5], subtracted.iloc[::5,:N], lw=2, alpha=0.5, 
                c='lightblue', label="Subtracted")
                    
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
        
        if pdfs is not None:
            pdfs.savefig(bbox_inches='tight')
            
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        plt.show()
        plt.close() 
    
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
    