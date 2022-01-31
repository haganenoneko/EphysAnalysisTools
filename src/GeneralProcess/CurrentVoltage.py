# Copyright (c) 2022 Delbert Yip
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pandas as pd
import numpy as np

from dataclasses import dataclass

from scipy.interpolate import UnivariateSpline

from typing import Dict, List, Tuple

import lmfit
import logging

import matplotlib.pyplot as plt

from GeneralProcess.Base import NDArrayFloat, multi_sort
from GeneralProcess.Interfaces import Recording, AbstractAnalyzer, AbstractPlotter
from GeneralProcess.LeakSubtraction import AbstractGHKCurrent

# ---------------------------------------------------------------------------- #
#                     Current-Voltage Analysis and Plotting                    #
# ---------------------------------------------------------------------------- #


@dataclass
class IVResults:
    i_inst: pd.DataFrame
    params: pd.DataFrame
    params_cm: pd.DataFrame


class AnalyzeIV(AbstractAnalyzer):
    """
    Find reversal potential, ion permeabilities, and linear fit from instantaneous current.  

    `df_i` = extracted epoch of leak-subtracted current       
    `df_v` = corresponding voltage command  
    `khz` = sampling frequency, in khz  
    `w` = starting from the start of each trace, time window to determine instantaneous current, in ms  
    `Cm` = membrane capacitance; if given a plot of current density will also be produced  
    """

    def __init__(self, data: Recording, ghk_model: AbstractGHKCurrent) -> None:
        self.data = data
        self.GHK_model = ghk_model

    def get_Iinst(
        self, current: pd.DataFrame, voltages: pd.DataFrame, w: int = 5
    ):
        """Extract instantaneous current amplitude"""
        khz = self.data.attrs['khz']

        # select current and voltage in desired window given by `w`
        i_inst = current.iloc[:w*khz, :].mean(axis=0).values
        voltages = voltages.iloc[1, :].values

        return voltages, i_inst

    @staticmethod
    def find_Erev(
        params: NDArrayFloat, spl: UnivariateSpline, v_min=-40, v_max=10
    ) -> float:
        """
        Find reversal potential as the root of polynomial spline or parameters of linear fit
        """
        try:
            Erev = [x for x in spl.roots() if v_min < x < v_max][0]
        except (ValueError, RuntimeError):
            # reversal potential from linear fits
            logging.error(
                "Failed to estimate x-intercept of I-V from the root of polynomial splines. Defaulting to inference from parameters of linear Ohmic fit to leak current.")

            Erev = -params[0] / params[1]

        return Erev

    @staticmethod
    def fit_spl(
        voltages: NDArrayFloat, current: NDArrayFloat
    ) -> UnivariateSpline:
        """Fit polynomial spline"""
        spl = UnivariateSpline(voltages, current)
        return spl

    def fit_ghk(
        self, voltages: NDArrayFloat, current: NDArrayFloat, Erev: float
    ) -> lmfit.Parameters:
        """
        Fit instantaneous current-voltage relation 
        """
        # update reversal potential and thermodynamic exponent
        self.GHK_model.rmp = Erev

        # fit Iinst-V with GHK current equation
        ghk_params = self.GHK_model.fit(voltages, current)

        return ghk_params

    def run(self) -> Tuple[pd.DataFrame]:

        voltages, i_inst = self.get_Iinst()
        i_inst_cm = i_inst / self.data.params.loc['Cm']

        self.Iinst_df = pd.DataFrame(
            data={"pA": i_inst, "pA/pF": i_inst_cm}, index=voltages)

        res: Dict[str, NDArrayFloat] = dict()
        params: Dict[str, Dict[str, float]] = dict()

        for col in ['pA', 'pA/pF']:
            current = self.Iinst_df.loc[:, col].values

            # sort voltages (and corresponding current) in ascending order
            voltages, current = multi_sort(zip(voltages, current))

            linfit = np.polyfit(voltages, current, deg=1)
            i_ohmic = np.polyval(linfit, voltages)

            # simulate current with polynomial spline
            spl = self.fit_spl(voltages, current)
            Erev = self.find_Erev(linfit, spl)
            i_spl = spl(voltages)

            # simulate GHK current
            ghk_params = self.fit_ghk(voltages, current, Erev)
            i_ghk = self.GHK_model.simulate(ghk_params, voltages)

            # save simulated current
            res[f'ohmic_{col}'] = i_ohmic
            res[f'spl_{col}'] = i_spl
            res[f'ghk_{col}'] = i_ghk

            # save fit parameters
            pars = self.GHK_model.get_all_permeabilities(ghk_params)
            pars.update({'Erev': Erev, 'ohmic_y0': linfit[0],
                         'ohmic_gamma': linfit[1]})
            params[col] = pars

        # simulated current
        IV_sim = pd.DataFrame.from_dict(res)

        # fitted GHK permeabilities
        IV_params = pd.DataFrame.from_dict(params)

        return IV_sim, IV_params

    def plot_results(self, IV_sim: pd.DataFrame, IV_params: pd.DataFrame, ref_ion: str = 'K') -> None:
        """Create current-voltage plots

        :param IV_sim: Simulated I-V plots
        :type IV_sim: pd.DataFrame
        :param IV_params: Fit parameters
        :type IV_params: pd.DataFrame
        :param ref_ion: Ion used as reference to compute relative GHK permeabilities, defaults to 'K'
        :type ref_ion: str, optional
        """

        plotter = PlotIV(self.data, self.GHK_model)
        plotter.plot(self.Iinst_df, IV_sim, IV_params,
                     ref_ion=ref_ion)

        return None

    def extract_data(self, key: str) -> None:
        return super().extract_data(key)


class PlotIV(AbstractPlotter):
    def __init__(self, data: Recording, GHK_model: AbstractGHKCurrent) -> None:

        self.Cm = data.params['C_m']
        self.GHK_model = GHK_model

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

            ax.margins(0.15)

            ax.locator_params(axis='y', nbins=4)

    def add_legend(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def _replace_braces(label: str) -> str:
        return label.replace('lbr', "{{").replace("rbr", "}}")

    def create_IV_legend(
        self, params: Dict[str, float], ref_ion: str, Cm: float
    ) -> List[str]:
        """Create 'legend' for IV plots."""

        labels: List[str] = []

        if ref_ion in params:
            P_ref = params[ref_ion]
        else:
            ref_ion = None
            P_ref = None

        absolute_label = "$P_lbr{ion:<8}{val:>8.1e}"
        relative_label = "$P_lbr{ref}rbr/P_lbr{ion:<8}{val:>8.2f}"

        for ion, P_ion in params.items():

            if 'ohmic' in ion or 'Erev' in ion:
                continue

            if P_ref is None and ref_ion is None:
                ref_ion = ion
                P_ref = P_ion

            ion += 'rbr$'
            alab = self._replace_braces(
                absolute_label.format(ion=ion, val=P_ion))
            labels.append(alab)

            if ion != ref_ion:
                rlab = self._replace_braces(relative_label.format(
                    ref=ref_ion, ion=ion, val=(P_ion/P_ref)))
                labels.append(rlab)

        labels.extend([
            f"{'$E_{{rev}}$':<8}{params['Erev']:>8.1f} mV",
            f"{'$C_m$':<8}{Cm:>8d} pF"
        ])

        labels = "\n".join(labels)
        logging.info(labels)

        return labels

    def add_IV_legend(
        self, ax: plt.Axes, params: Dict[str, float], ref_ion: str
    ) -> Dict[str, float]:
        """Add labels for fit parameters, Erev, and Cm"""

        labels = self.create_IV_legend(
            params, ref_ion, self.data.params['C_m'])

        # add text box with the labels
        if ax.get_xlim()[0] < -120:
            ax.text(0.65, 0.2, labels, transform=ax.transAxes,
                    fontsize=11, va='top')
        else:
            ax.text(0.75, 0.2, labels, transform=ax.transAxes,
                    fontsize=11, va='top')

        # legend
        ax.legend(loc='upper left', fontsize=11,
                  bbox_to_anchor=(0, 0.7, 0.5, 0.5))

    def plot(
        self, IV_data: pd.DataFrame, IV_sim: pd.DataFrame,
        IV_params: pd.DataFrame, ref_ion: str = 'K',
        ohmic_kw=dict(lw=2, ls='--', c='blue'),
        spl_kw=dict(ls=':', c='g', lw=2.5),
        ghk_kw=dict(c='r', lw=2),
        data_kw=dict(marker='o', ls='none', ms=6, c='k'),
    ):
        """Plot current-voltage data and simulations

        :param IV_data: Instantaneous current for different conditions in columns and voltages for each condition in rows. When called from `AnalyzeIV.plot_results()`, the columns are raw current (`pA`) and current density (`pA/pF`)
        :type IV_data: pd.DataFrame
        :param IV_sim: Simulated current with simulations in columns and voltages in rows. The column names should be `[model name]_[condition]`, e.g. `ghk_pA`
        :type IV_sim: pd.DataFrame
        :param IV_params: Fit parameters used to generate `IV_sim`. When called from `AnalyzeIV`, these include GHK permeabilities (e.g. `Na`, `K`) in the rows and conditions in the columns
        :type IV_params: pd.DataFrame
        :param ref_ion: Reference ion for computing relative GHK permeabilities, defaults to 'K'
        :type ref_ion: str, optional
        :param ohmic_kw: Appearance of Ohmic fits, defaults to dict(lw=2, ls='--', c='blue')
        :type ohmic_kw: Dict[str, Any], optional
        :param spl_kw: Appearance of spline fits, defaults to dict(ls=':', c='g', lw=2.5)
        :type spl_kw: Dict[str, Any], optional
        :param ghk_kw: Appearance of GHK fits, defaults to dict(c='r', lw=2)
        :type ghk_kw: Dict[str, Any], optional
        :param data_kw: Appearance of data, defaults to dict(marker='o', ls='none', ms=6, c='k')
        :type data_kw: Dict[str, Any], optional
        """
        volts = IV_data.index.values
        self.format_axes()

        for i, col in enumerate(IV_data.columns):

            if i > len(self.axs):
                break

            self.axs[i].plot(volts, IV_data[col], label=None, **data_kw)

            # select columns containing 'col'
            header = IV_sim.filter(like=col, axis=1).columns

            if f'ghk_{col}' in header:
                self.axs[i].plot(volts, IV_sim[f'ghk_{col}'],
                                 label='GHK', **ghk_kw)
            elif f'ohmic_{col}' in header:
                self.axs[i].plot(volts, IV_sim[f'ohmic_{col}'],
                                 label='Ohmic', **ohmic_kw)
            elif f'spl_{col}' in header:
                self.axs[i].plot(volts, IV_sim[f'spl_{col}'],
                                 label='Spline', **spl_kw)

            self.add_IV_legend(self.axs[i], IV_params.loc[col], ref_ion)
