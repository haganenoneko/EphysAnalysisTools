# Copyright (c) 2021 Delbert Yip
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

""" Analysis of Voltage Clamp Seal Quality
This module analyzes the quality of recordings by computing and visualizing seal parameters membrane capacitance (Cm), series resistance (Rsr), and membrane (Rm), seal (Rsl), and pipette (Rp) resistances. 

Note that the methods used for Cm calculation assume that protocols include a symmetric voltage ramp. 

The functions in this module are inspired by the following references:
    1. http://www.billconnelly.net/?p=310
    2. https://www.electronics-tutorials.ws/filter/filter_2.html
    3. https://swharden.com/blog/2020-10-11-model-neuron-ltspice/
    4. http://www.billconnelly.net/?p=501
    5. https://www.electronics-tutorials.ws/amplifier/frequency-response.html
    6. https://support.moleculardevices.com/s/article/Membrane-Test-Algorithms
"""

from scipy.optimize import curve_fit
from scipy.integrate import simps

import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt

from typing import List, Dict, Any, Tuple, Union

from GeneralProcess.base import NDArrayFloat, KwDict, exp1
from GeneralProcess.base import AbstractAnalyzer, Recording_Leak_MemTest

from pydantic import BaseModel, ValidationError, validator

# ---------------------------------------------------------------------------- #


class VoltageClampQuality(AbstractAnalyzer):
    """
    Inspect the effectiveness of voltage clamp by computing the ratio 
    between the actual membrane potential ($V_{out}$) and the command 
    voltage ($V_{in}$). See Ref. 1 in the module header.

    The corner frequency is also called the "-3dB cutoff frequency," 
    which is the frequency at which the output gain is reduced to 
    79.71% of its maximum value. See Refs. 4 and 5 in the module header 
    for more information.
    """

    def __init__(
        self, data: Recording_Leak_MemTest, show: bool,
        test_freqs: List[float] = [1e-1, 1e4, 400], **kwargs
    ) -> None:

        self.__data = data

        self.__params = data.params.loc[
            ['C_m (pF)', 'R_m (M)', 'R_sr (M)']
        ].values.tolist()

        self._getFreqsAndVolageRatios(test_freqs)

        if show:
            self.plot_results(**kwargs)

    @staticmethod
    def _convertUnits(
        Cm: float, Rm: float, Rsr: float, to_SI=True
    ) -> List[float]:
        """Convert parameter units to SI if `to_SI=True`, else the reverse."""

        if to_SI:
            return [Cm * 1e-12, Rm * 1e6, Rsr * 1e6]
        else:
            return [Cm * 1e12, Rm * 1e-6, Rsr * 1e-6]

    @staticmethod
    def _transformFrequencies(
        freqs: NDArrayFloat, Cm: float, Rm: float, Rsr: float
    ) -> NDArrayFloat:
        """Compute the ratio of actual voltage and command voltage (aka `V_out` and `V_in`, respectively) for frequencies `freqs`

        :param freqs: frequencies in Hz
        :type freqs: NDArrayFloat
        :param Cm: membrane capacitance, in Farads
        :type Cm: float
        :param Rm: membrane resistance, in Ohms
        :type Rm: float
        :param Rsr: series resistance, in Ohms
        :type Rsr: float
        :return: ratios `V_out`/`V_in`, following the formula in Ref. 1 (see module heading)
        :rtype: NDArrayFloat
        """
        Rm2 = Rm**2
        Rm2i = 1/Rm2
        Cm2 = Cm**2
        w2 = (freqs*2*math.pi)**2

        A = Cm2*w2 + Rm2i
        B = ((1 / (Rm*A)) + Rsr)**2
        B += Cm2*w2 / (A**2)

        return np.real((A * B) ** -0.5)

    @staticmethod
    def _computeCornerFreq(Cm: float, Rm: float, Rsr: float) -> float:
        """
        Compute corner frequency of voltage clamp filter 
        (MHz if Rsr in MOhm and Cm in pF)
        """
        return np.real(
            (Rm**2 - 2*Rm*Rsr - Rsr**2)**0.5 / (2*math.pi*Cm*Rm*Rsr)
        )

    def _computeVoltageRatios(
        self, freqs: List[float]
    ) -> Tuple[NDArrayFloat, NDArrayFloat, float, float]:
        """compute Vout/Vin for ratios given frequencies

        :param freqs: Minimum, maximum, and number of frequencies to evaluate (in Hz)
        :type freqs: List[float]
        :return: `V_out`/`V_in` ratios for each frequency in a geometric sequence from `freqs[0]` to `freqs[1]`
        :rtype: NDArrayFloat
        """
        # parameters in SI units
        params_SI = self._convertUnits(*self.__params, to_SI=True)

        # arbitrary frequencies
        freqs = np.geomspace(*freqs)
        V_ratios = self._transformFrequencies(freqs, *params_SI)

        # corner frequency
        f_corner = self._computeVoltageRatios(*params_SI)
        corner_ratio = self._transformFrequencies(f_corner, *params_SI)

        return freqs, V_ratios, f_corner, corner_ratio

    def _getFreqsAndVolageRatios(self, freqs: List[float]) -> None:
        """Compute and store frequencies and voltage ratios

        :param freqs: minimum, maximum, and number of frequencies to evaluate
        :type freqs: List[float], optional
        """
        freqs, V_ratios, f_corner, corner_ratio = self._computeVoltageRatios(
            freqs)

        self.__results = dict(
            test_freqs=freqs,
            test_ratios=V_ratios,
            f_corner=f_corner,
            corner_ratio=corner_ratio
        )

    def plot_results(
        self, plot_kw: KwDict = {'lw': 2},
        fig_kw: KwDict = {'figsize': (9, 4)},
        corner_kw: KwDict = {'ls': '--', 'lw': 2},
        axes_kw: KwDict = {
            'xscale': 'log',
            'ylabel': dict(fontsize=20, rotation=0, labelpad=22),
            'grid': dict(b=True, which='both', axis='both', alpha=0.3),
        }
    ) -> None:
        """Plot a `V_out`/`V_in` vs. frequency (Hz) graph and indicate the corner frequency, given seal parameters in `self.__data`

        :param plot_kw: appearance of arbitrary frequencies, defaults to {'lw' : 2}
        :type plot_kw: KwDict, optional
        :param fig_kw: appearance of the Figure, defaults to {'figsize' : (9, 4)}
        :type fig_kw: KwDict, optional
        :param corner_kw: appearance of line indicating the corner frequency, defaults to {'ls' : '--', 'lw' : 2}
        :type corner_kw: KwDict, optional
        :param axes_kw: additional axes properties, defaults to { 'xscale' : 'log', 'ylabel' : dict(fontsize=20, rotation=0, labelpad=22), 'grid' : dict(b=True, which='both', axis='both', alpha=0.3), }
        :type axes_kw: KwDict, optional
        """

        freqs, V_ratios, f_corner, corner_ratio = self.__results.values()

        fig, ax = plt.subplots(**fig_kw)
        ax.plot(freqs, V_ratios, **plot_kw)

        # frequency and seal parameters in operational units
        f_corner = np.real(f_corner)/1e3
        Cm, Rm, Rsr = self._convertUnits(*self.__params, to_SI=False)

        label = [f"$f_c$ = {f_corner:.1f} kHz",
                 r"$V_{{out}}/V_{{in}}$ = {.3f}".format(corner_ratio),
                 f"$R_m$ = {Rm:d} M$\Omega$",
                 f"$C_m$ = {Cm:.1f} pF",
                 f"$R_s$ = {Rsr:.1f} M$\Omega$"
                 ]
        label = "\n".join(label)

        # plot corner ratio as a vertical line
        ax.axvline(f_corner, label="\n".join(label), **corner_kw)
        ax.legend(loc='lower left')

        ax.set_title(self.__data.name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"$\mathbf{\frac{V_{out}}{V_{in}}}$",)

        ax.update(axes_kw)

        fig.tight_layout()
        super().save_pdf(fig)

    def extract_data(self, key: str) -> Union[List[Any], Any]:

        if key is None:
            return list(self.__results.values())
        elif key in self.__results:
            return self.__results[key]
        else:
            print(f"{key} is an invalid key. \
                Available keys:\n{self.__results.keys()}")

# -------------------------- Membrane test analyses -------------------------- #


class EstimateRampCm(BaseModel):
    """
    Estimate membrane capacitance ($C_m$) from symmetric voltage ramps. 
    See Ref. 3 in the module header for more info.
    """
    khz: int
    ramp_startend: List[int]
    ramp_centerFrac: float

    @validator('ramp_centerFrac')
    def centerFrac_in_zero_one(cls: object, val: float) -> float:
        if 0 < val <= 1:
            return val
        raise ValueError(f"`centerFrac` must be in (0, 1], not {val}.")

    def _Cm_from_ramp(self, ramp: pd.DataFrame, thalf: int) -> float:

        khz = self.khz
        cf = self.ramp_centerFrac

        # split the ramp current into separate arms
        ramp1 = ramp.iloc[:thalf, 0].values[::-1]
        ramp2 = ramp.iloc[thalf:, 0].values

        # figure out the middle of the data we wish to sample from
        centerPoint = int(len(ramp1))/2
        centerLeft = int(centerPoint*(1 - cf))
        centerRight = int(centerPoint*(1 + cf))

        ramp_duration = ramp.shape[0]/(2*khz)

        # total voltage drop
        dV = ramp.iat[thalf+1, 1] - ramp.iat[0, 1]

        # ramp slope in mV/ms
        ramp_slope_ms = dV / (ramp_duration)

        # average slope deviation (distance from the mean)
        d_ramp = (ramp1 - ramp2)/2

        # deviation for the center
        d_ramp_center = d_ramp[centerLeft:centerRight]
        deviation = np.mean(d_ramp_center)

        return np.abs(deviation / ramp_slope_ms)

    def get_ramp_Cm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use `SW Harden`'s method to estimate membrane capcitance ($C_m$) 
        from symmetric voltage ramps (see Ref. 3 in the module header)

        :param startend: indices of `df` for the start and end of symmetric voltage rampps
        :type startend: List[int]
        :return: estimated membrane capacitance
        :rtype: List[float]
        """
        ta, tb = self.ramp_startend
        ramp = df.iloc[ta:tb, :]
        N = int(ramp.shape[1]/2)

        # ramp midpoint
        thalf = int(ramp.shape[0]/2)

        cm_vals = []
        for i in range(N):
            ramp_i = ramp.iloc[:, [i, N+i]]
            cm_i = self._Cm_from_ramp(ramp_i, thalf)
            cm_vals.append(cm_i)

        return pd.Series(cm_vals, name='Ramp_Cm')


def MemTest_SWH(dV_max: float, tau: float, I_d: float, I_dss: float) -> List[float]:
    """
    Compute seal parameters using SW Harden's formulae

    At time zero, access resistance limits our ability to deliver current `Id` to a known `dV` (`Cm` doesn't come into play yet). Thus, 
    $$ R_a = \Delta V_{max} / I_d $$

    the difference between this steady state current (Iss) and the last one (`Iprev`) is limited by the sum of `Rm` and `Ra`
    $$ R_m + R_a = \Delta V_{max} / (I_{ss} - I_0) $$
    $$ R_m = (\Delta V_{max} - R_a * (I_{ss} - I_0) ) / (I_{ss} - I_0) $$

    When we raise the cell's voltage (`Vm`) by delivering current through the pipette (Ra), some current escapes through `Rm`. From the cell's perspective when we charge it though, `Ra` and `Rm` are in parallel.

    $$ C_m = \tau / R_T $$
    $$ 1/R_T = 1/R_a + 1/R_m $$

    `Rm` leaks a small amount of the `Id` current that passes through Ra to charge `Cm`. We can calculate a correction factor as the ratio of `Ra` to `Rm` and multiply it by both of our resistances. `Cm` can be corrected by dividing it by the square of this ratio.

    :param dV_max: maximum voltage drop
    :type dV_max: float
    :param tau: single-exponential time constant fitted to the decay of the membrane test current from its peak
    :type tau: float
    :param I_d: difference between peak membrane test current and baseline current
    :type I_d: float
    :param I_dss: difference between steady-state membrane test current and baseline current 
    :type I_dss: float
    :return: [`tau`, `R_a`, `R_m`, `C_m`]
    :rtype: List[float]
    """
    R_a = abs(dV_max/I_d)*1e3
    R_m = abs((dV_max*1e-3 - R_a*I_dss*1e-6)/(I_dss*1e-12))*1e-6
    C_m = abs(tau / (1 / (1/R_a) + (1/R_m))) * 1e3

    if R_m >= 10*R_a:
        correction = 1 + (R_a / R_m)

        R_a *= correction
        R_m *= correction
        C_m *= 1/(correction**2)

    return [R_a, R_m, C_m]


def MemTest_MDC(
    I_t: NDArrayFloat, times: NDArrayFloat, dV_max: float,
    tau: float, I_ss: float, I_dss: float
) -> Tuple[List[float], float]:
    """
    Compute seal parameters using Molecular Devices' (MDC) formulae, and estimate the error by comparing the fitted time constant `tau` to that estimated by `Cm * R_t2`, where `R_t2` is the sum of access (`R_a`) and membrane resistances (`R_m`). See Ref. 6 in the module header for more information.

    :param I_t: membrane test current
    :type I_t: NDArrayFloat
    :param times: measurement times
    :type times: NDArrayFloat
    :param dV_max: maximum voltage drop
    :type dV_max: float
    :param tau: single-exponential time constant fitted to the decay of the membrane test current from its peak
    :type tau: float
    :param I_ss: steady state current at the end of the membrane test step
    :type I_ss: float
    :param I_dss: difference between `I_ss` and baseline current 
    :type I_dss: float
    :return: [`R_a`, `R_m`, `C_m`], deviation in `tau`
    :rtype: Tuple[List[float], float]
    """

    # C_m = Q /dV, where Q is obtained by integrating the capacitive transient
    Q = simps([x - I_ss for x in I_t], times) + I_dss*tau
    Cm = abs(Q/dV_max)

    # from Molecular Devices (MDC)
    # tau = R * Cm, 1/R = 1/Rm + 1/Ra = (Ra + Rm)/(Ra * Rm)
    # tau/Cm = (Ra*Rm) / (Ra + Rm)
    # Ra^2 - Ra*Rt + Rt*(tau/Cm) = 0, R_t = Ra + Rm
    R_t = abs((dV_max/I_dss)*1e3)
    quad_factors = [1, -R_t, R_t*(tau/Cm)*1e3]

    R_a = np.min(np.real(np.roots(quad_factors)))
    R_m = R_t - R_a

    # conductance
    g_t2 = (1/R_a) + (1/R_m)

    MDC_dtau = Cm/g_t2 - tau*1e3

    params_MDC = [R_a, R_m, Cm]

    return params_MDC, MDC_dtau


class VoltageClampEstim(AbstractAnalyzer):
    """
    Estimate $C_m$, $R_m$, and $R_{sr}$ from recording parameters 
    using two methods:

    1. `MemTest_SWH`
    `SWH` and `SW Harden` refer to Scott W. Harden. This method 
    estimates membrane capacitance $C_m$ from the current response to
    two symmetric symmetric voltage ramps. See Ref. 3 in the main 
    header for more information.

    2. `MemTest_MDC`
    `MDC` and `Molecular Devices` refer to algorithms implemented
    in pClamp/Axoscope software (Molecular Devices, LLC., San Jose, CA).
    The implementation here follows the description in Ref. 6.
    """

    def __init__(self, data: Recording_Leak_MemTest,
                 memtest_kw: KwDict = {}, centerFrac: float = 0.3,
                 show=False, memtest_plot_kw: KwDict = {},
                 ramp_cm_plot_kw: KwDict = {}) -> None:

        self.__data = data
        self.__khz = data.attrs['khz']
        
        self.params_SWH = []  # SW Harden methods = R_a, R_m, C_m 
        self.params_MDC = []  # MDC methods = R_a, R_m, C_m
        self.MDC_dtau =   []  # check correspondence between tau and Rm using tau ~ Rm*Cm

        self.run(memtest_kw, centerFrac)
        
        if show:
            self.plot_results(memtest_plot_kw=memtest_plot_kw,
                              ramp_cm_plot_kw=ramp_cm_plot_kw)

    @staticmethod
    def _fitMemTestExp1(
        times: NDArrayFloat, I_t: NDArrayFloat,
        p0: List[float], lowers: List[float], uppers: List[float]
    ) -> List[float]:
        """Fit single exponential to current in membrane test following peak current

        :param times: 
        :type times: NDArrayFloat
        :param I_t: current in membrane test following peak current
        :type I_t: NDArrayFloat
        :param p0: initial parameters (tau, C), defaults to None
        :type p0: List[float], optional
        :param bounds: defaults to None
        :type bounds: Tuple[List[float], List[float]], optional
        :return: fitted parameters 
        :rtype: List[float]
        """
        if lowers is None:
            lowers = [0, 1e-3, -1e3]
        if uppers is None:
            uppers = [1e3, 100, 1e3]

        popt, _ = curve_fit(exp1, times, I_t, p0=p0, bounds=(lowers, uppers))
        return popt

    def _truncateMemTestCapacitance(
        df: NDArrayFloat, time: NDArrayFloat
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """Truncate capacitance peak (current maximum) from membrane test step

        :param df: current
        :type df: NDArrayFloat
        :param time: array of measurement times
        :type time: NDArrayFloat
        :return: truncated times, current
        :rtype: Tuple[NDArrayFloat, NDArrayFloat]
        """

        # find index of capacitance peak
        peak_idx = int(np.argmax(df))

        # truncate time and current accordingly
        df = df[peak_idx:]
        times = time[peak_idx:] - time[peak_idx]

        return times, df

    @staticmethod
    def _getBaseCurrent(df: pd.DataFrame, t: int, fifth: int, khz: int) -> float:
        """current just before the membrane test step"""
        a = t - fifth - 10*khz
        b = t - 10*khz
        return df.iloc[a:b, 0].mean()

    @staticmethod
    def _getMaxVoltageDrop(
        I_t: NDArrayFloat, df: pd.DataFrame, t: int, fifth: int
    ) -> Tuple[float, float]:
        """Find maximal current and voltage drop"""
        dV = df.iat[t + 2*fifth, 1] - df.iat[t - 2*fifth, 1]
        dI = np.max(I_t) - np.min(I_t)
        return dI, dV

    def _oneSidedMemTest(
        self, df: pd.DataFrame, times: NDArrayFloat, I_t: NDArrayFloat,
        ind: int, fifth: int, **fit_kwargs
    ) -> None:
        """Estimate membrane test parameters for one side of the membrane test step"""
        # discard capacitance spike
        times, I_t = self._truncateMemTestCapacitance(times, I_t)

        I_base = self._getBaseCurrent(df, ind, fifth, self.__khz)
        dI_max, dV_max = self._getMaxVoltageDrop(df, ind, fifth)
        p0 = [dI_max, 10, np.mean(I_t[-fifth:])]

        dI_max, tau, I_ss = self._fitMemTestExp1(
            times, I_t, p0=p0, **fit_kwargs)

        I_peak = I_ss + dI_max
        I_d = I_peak - I_base
        I_dss = I_ss - I_base

        mdc, mdc_err = MemTest_MDC(I_t, times, dV_max, tau, I_ss, I_dss)
        self.params_MDC.append(mdc)
        self.MDC_dtau.append([tau, mdc_err])

        self.params_SWH.append(
            MemTest_SWH(dV_max, tau, I_d, I_dss)
        )

    def averageMemTestParams(self) -> Tuple[pd.DataFrame]:
        """Take average of start and end capacitive transients"""

        cols = ['tau', 'R_a', 'R_m', 'C_m']
        def _get_avg_df(params: List[List[float]]) -> pd.DataFrame:
            A = np.array(params)
            A = 0.5 * (A[::2] + A[1::2])
            return pd.DataFrame(A, columns=cols)
            
        mdc = _get_avg_df(self.params_MDC)
        swh = _get_avg_df(self.params_SWH)
        dtau = _get_avg_df(self.MDC_dtau)
                
        return mdc, swh, dtau 

    def estimate_memTest(
        self, startend: List[int], **fit_kwargs
    ) -> Tuple[pd.DataFrame]:
        """Using the method described by SW Harden and pClamp, estimate Ra, Rm, and Cm by fitting a single exponential to the capacitive transient of a membrane test step.

        :param startend: indices for the start and end of membrane test step in `df`
        :type startend: List[int]
        :param df: raw data, with 1:N current columns and N+1:2N voltage columns
        :type df: pd.DataFrame
        :param khz: sampling rate, in Khz
        :type khz: int, optional
        """
        df = self.__data.raw_data
        N = int(df.shape[1]/2)

        # estimate parameters for starting and ending capacitive transients
        for i in range(len(startend)-1):

            # isolate membrane test step
            ta, tb = startend[i:i+2]
            df_MT = df.iloc[ta:tb, :].copy()
            df_MT.index -= df_MT.index[0]

            # fifth of the interval used for I_ss calculations
            fifth = int(df_MT.shape[0]/5)
            times = df_MT.index.values.tolist()

            # invert transient if mean of first 20ms is less than last 20ms
            mu = df_MT.iloc[:, :N].mean(axis=0).values
            if mu[0] < mu[-1]:
                df_MT.iloc[:, :N] *= -1

            for j in range(N):
                I_t = df_MT.iloc[:, j].values
                self._oneSidedMemTest(df, times, I_t, ta, fifth, **fit_kwargs)
        
        # average parameters
        return self.averageMemTestParams()

    def estimate_RampCm(self, centerFrac: float) -> List[float]:
        """Estimate membrane capacitance $C_m$ from symmetric voltage ramps"""

        data = self.__data

        if 'ramp_startend' not in data.attrs:
            raise ValueError(f"{data.name} does not have start and\
                end indices for symmetric (leak) voltage ramps")

        try:
            cm_estim = EstimateRampCm(
                self.__khz, data.ramp_startend, centerFrac)
        except ValidationError as e:
            raise(e)

        return cm_estim.get_ramp_Cm(data.raw_data)

    def run(self, mt_kw: KwDict, centerFrac: float) -> None:

        data = self.__data
        self.params_MDC, self.params_SWH = self.estimate_memTest(
            data.mt_startend, **mt_kw)

        self.__ramp_cm = self.estimate_RampCm(centerFrac)
        return 

    def plot_MemTest(
        self, fig_kw: KwDict = {'figsize': (12, 6)},
        swh_kw: KwDict = dict(marker='o', c='r', alpha=0.7, label="SWH"),
        mdc_kw: KwDict = dict(marker='x', ms=8, c='k', alpha=0.7, label="MDC")
    ) -> None:

        # mt_mdc: pd.DataFrame, mt_swh: pd.DataFrame,
        mt_mdc = self.params_MDC
        mt_swh = self.params_SWH
        dtau = self.MDC_dtau
        
        fig, axs = plt.subplots(2, 3, **fig_kw)
        labels = [r"$R_a$ (M$\Omega$)", r"$R_m$ (M$\Omega$)", r"$C_m$ (pF)",
                  r"$\tau$ (ms)", r"$\tau - R_m C_m$"]

        for i, ax in np.nditer(axs):
            
            if i == mt_swh.shape[1]: 
                ax.plot(dtau.iloc[:,0], marker='o', label=labels[i])
                ax2 = ax.twinx()
                ax2.plot(dtau.iloc[:,1], marker='x', label=labels[i+1])
                ax.legend()
                break 
            
            ax.plot(mt_swh.iloc[:, i], **swh_kw)
            ax.plot(mt_mdc.iloc[:, i], **mdc_kw)

            ax.legend()
            ax.set_title(labels[i])

        for i in range(2):
            ax[1, i].set_xlabel("Sweep #")

        fig.suptitle("Estimation of Membrane Test Parameters")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        super().save_pdf(fig)

    def plot_RampCm(
        self, fig_kw: KwDict = {'figsize': (7, 4)},
        plot_kw: KwDict = {'marker': 'o', 'c': 'b', 'label': r'Ramp $C_m$'}
    ) -> None:
        """Plot $C_m$ values estimated from symmetric voltage ramps"""

        ramp_cm = self.__ramp_cm

        fig, ax = plt.subplots(**fig_kw)
        ax.set_ylabel("$C_m$ (pF)")
        ax.set_xlabel("Sweep #")

        ax.plot(ramp_cm, **plot_kw)

        fig.suptitle("Estimation of $C_m$ from Symmetric Voltage Ramps")
        fig.tight_layout()
        super().save_pdf(fig)

    def plot_results(
        self, memtest_plot_kw: Dict[str, dict],  ramp_cm_plot_kw: Dict[str, Union[float, dict]]
    ) -> None:

        self.plot_MemTest(**memtest_plot_kw)
        self.plot_RampCm(**ramp_cm_plot_kw)

    def __repr__(self) -> str:

        return f"""
        Analysis of voltage clamp quality for {self.__data.name}\n
        Membrane Test Analysis (S.W. Harden's method):\n{self.params_SWH}\n
        Membrane Test Analysis (Molecular Devices' method):\n{self.params_MDC}\n
        \t `tau - Rm*Cm`:\n{self.MDC_dtau}\n
        Membrane capacitance estimation from voltage ramps (S.W. Harden's method):\n
        {self.__ramp_cm}\n
        """

    def extract_data(self, key: str) -> Any:
        if key not in self.__dict__ and\
                key not in self.__data.attrs:
            return
        elif key in self.__dict__:
            return self.__dict__[key]
        elif key in self.__data.attrs:
            return self.__data.attrs[key]
