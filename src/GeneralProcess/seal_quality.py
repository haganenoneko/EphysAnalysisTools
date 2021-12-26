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
"""

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages

from typing import List, Any

from GeneralProcess.base import AbstractAnalyzer, AbstractRecording

# ---------------------------------------------------------------------------- #

class VoltageClampQuality(AbstractAnalyzer):
    """Compute and visualize seal parameters"""
    def __init__(self, data: AbstractRecording, 
                show: bool, aggregate_pdf: PdfPages) -> None:
        
        self._data = data
        self._show = show
        self._pdf = aggregate_pdf
        
    def 

# ---------------------------------------------------------------------------- #

def _plot_bc_vc_quality(
    freqs: np.ndarray, ratios: List[float], f_c: float, 
    seal_params: List[float], data_name: str, pdf: PdfPages,
    fig_kw: Dict[str, Any] = {'figsize' : (9, 4)},
    
) -> None:
    
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, ratios, lw=2)

    # vout/vin at the corner frequency
    corner_ratio = trans_func(f_c, Rsr, Rm, Cm)

    fc_khz = np.real(f_c)/1000
    Cm *= 1e12
    Rm *= 1e-6
    Rsr *= 1e-6

    # plot corner ratio as a vertical line
    ax.axvline(f_c, ls='--', c='yellow',
               label="$f_c$ = %.1f kHz \n$V_{out}/V_{in}$ = %.3f \n$R_m$ = %d M$\Omega$ \n$C_m$ = %.1f pF \n$R_s$ = %.1f M$\Omega$" % (fc_khz, corner_ratio, Rm, Cm, Rsr))
    ax.legend(loc='lower left')

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$\mathbf{\frac{V_{out}}{V_{in}}}$",
                  fontsize=20, rotation=0, labelpad=22)
    ax.set_title(fname)

    ax.set_xscale('log')
    ax.grid(b=True, which='both', axis='both', alpha=0.3)

    fig.tight_layout()

    if pdf is not None:
        pdf.savefig(bbox_inches='tight')

def BC_VoltageClampQuality(
    fname: str, params: pd.Series, 
    pdfs=None, show=False
):
    """
    fname = filenam   
    params = pd Series containing experimental parameters for file  
    pdfs = multipage pdf object to save plots  
    """
    
    # print(params)
    # colnames = ['R_pp (M)', 'R_sl (G)','C_m (pF)', 'R_m (M)', 'R_sr (M)']
    Cm, Rm, Rsr = params.loc[['C_m (pF)', 'R_m (M)', 'R_sr (M)']]

    Cm *= 1e-12
    Rm *= 1e6
    Rsr *= 1e6

    def trans_func(f, Rsr, Rm, Cm):
        Rm2 = Rm**2
        Rm2i = (1/Rm2)
        Cm2 = Cm**2
        w2 = (f*2*math.pi)**2

        t = (
            ((Cm2 * w2 + Rm2i)**0.5) *
            ((
                (((1/(Rm * (Cm2 * w2 + Rm2i))) + Rsr)**2) +
                (Cm2 * w2) / ((Cm2 * w2 + Rm2i)**2)
            )**0.5)
        )

        return np.real(1/t)

    # corner frequency of voltage clamp filter (MHz if Rsr in MOhm and Cm in pF)
    f_c = np.real((Rm**2 - 2*Rm*Rsr - Rsr**2)**0.5 / (2*math.pi*Cm*Rm*Rsr))
    # f_c *= 1e6     # convert to Hz

    # tau for voltage clamp
    # tau = Cm*((Rsr*Rm)/(Rsr + Rm))

    # frequencies for plotting
    freqs = np.geomspace(1e-1, 1e4, 400)
    # compute Vout/Vin for ratios given frequencies
    ratios = [trans_func(f, Rsr, Rm, Cm) for f in freqs]

    

    plt.close()


def file_specific_transform(fname, times=None, df=None):
    """
    Apply file-specific transformation to times, as necessary.  
    `fname` = filename or 'FA_env_+20'
    `df` = dataframe 
    `times` = bounding intervals for leak ramp  
    `khz` = sampling rate 
    """
    FA_env = [
        '2091004', '20911007', '20o08003', '20o16001',  # +20
        '20o16002', '21521006'                          # -55
    ]
    to_transform = ["20903005"]
    to_transform.extend(FA_env)

    if fname in to_transform:
        if times is not None:

            # ramp start (t0) and end (t1) times
            if isinstance(times, list):

                if fname == '20903005':
                    # after the first trace, multiply all epochs by 2
                    t0, t1 = times
                    return [t0, t1, 2*t0, 2*t1, 2*t0, 2*t1]

            elif isinstance(times, dict):

                if fname in times.keys():
                    if fname == '20903005':
                        d = times[fname]

                        for i in range(1, len(d)):
                            d[i] = [2*x for x in d[i]]

                        times[fname] = d

                    elif fname in FA_env:
                        if isinstance(times[fname][0], list):
                            if times[fname][0][-1] > 47000:
                                val = times[fname][0]
                                times[fname][0][-3:] = [x -
                                                        7233 for x in val[-3:]]

                        elif isinstance(times[fname][0], int):
                            if times[fname][0] > 47000:
                                times[fname][0] -= 7233

            elif isinstance(times, tuple):

                # tuple = abf_sum, csv_sum -> force abf_sum = csv_sum
                if fname == '20903005':
                    return times[0]
                else:
                    return times[1]

            return times

        if df is not None:
            if fname in FA_env:
                N = int(df.shape[1]/2)

                dt = df.shape[0] - 47937
                new = df.iloc[40704:, N].copy()

                # plt.plot(df.iloc[:,N], lw=2, c='y')
                df.iloc[40704:(40704+dt), N] = new.iloc[7233:].values
                df.iloc[52766:, N] = df.iloc[-1, N]

                # plt.plot(new, ls='--', lw=2, c='w')
                # plt.plot(df.iloc[:, N], c='r', alpha=0.5, lw=2)
                # plt.show()
                # exit()

                return df

    else:
        if times is not None:
            if isinstance(times, tuple):
                # abf_sum, csv_sum -> keep csv_sum
                return times[1]
            else:
                # don't change anything
                # print(" `file_specific_transform` was called, but `fname` is None. Please provide a filename.")
                return times

        elif df is not None:
            return df


def estimate_Cm(startend, df, khz=1, centerFrac=0.3):
    """
    Use SW Harden's method of estimating Cm from voltage ramps.

    startend = start adn end indices of voltage ramp for Cm estimation
    df = dataframe for recording
    khz = sample frequency of data, in khz 
    centerFrac is the fractional time span to draw data from in the center of each ramp.
    """
    ramp = df.iloc[startend[0]:startend[1], :]

    # number of traces
    N = int(ramp.shape[1]/2)

    # check ramp dissection
    # plt.plot(ramp.iloc[:,N:])
    # ### plt.show()
    # exit()

    # find midpoint of ramp
    thalf = int(ramp.shape[0]/2)

    cm_vals = []
    for i in range(N):
        # split the ramp current into separate arms
        ramp1 = ramp.iloc[:thalf, i].values[::-1]
        ramp2 = ramp.iloc[thalf:, i].values

        # average of both arms
        ramp_avg = np.mean([ramp1, ramp2], axis=0)

        # figure out the middle of the data we wish to sample from
        centerPoint = int(len(ramp1))/2
        centerLeft = int(centerPoint*(1-centerFrac))
        centerRight = int(centerPoint*(1+centerFrac))

        # slope of the ramp in ms
        ramp_duration = ramp.shape[0]/(2*khz)  # duration

        # dV = np.ptp(ramp.iloc[:,N+i].values, axis=0)
        dV = ramp.iat[thalf+1, N+i] - ramp.iat[0, N+i]

        ramp_slope_ms = dV / (ramp_duration)

        # average slope deviation (distance from the mean)
        d_ramp = (ramp1 - ramp2)/2

        # deviation for the center
        d_ramp_center = d_ramp[centerLeft:centerRight]
        deviation = np.mean(d_ramp_center)

        cm_vals.append(np.abs(deviation / ramp_slope_ms))

    return cm_vals


def estimate_MT(startend, df, khz=1, pdf=None):
    """
    Using the method described by SW Harden and pClamp, estimate Ra, Rm, and Cm by fitting a single exponential to the capacitive transient of a membrane test step.
    """
    N = int(df.shape[1]/2)

    # define a single exponential
    def func(t, dI, tau, I_ss):
        return dI*np.exp(-t/tau) + I_ss

    params_SWH = []  # SW Harden methods
    params_MDC = []  # MDC methods
    MDC_Rm_error = []  # check correspondence between tau and Rm using tau ~ Rm*Cm

    def do_estimation(k):
        # isolate membrane test
        # to prevent SettingWithCopyWarning: https://stackoverflow.com/a/58829423
        df_MT = df.iloc[startend[k]:startend[k+1], :].copy()
        df_MT.index -= df_MT.index[0]
        time = df_MT.index.values.tolist()

        # fifth of the interval used for I_ss calculations
        fifth = int(df_MT.shape[0]/5)

        # invert transient if mean of first 20ms is less than last 20ms
        mu = df_MT.iloc[:, :N].mean(axis=0).values
        if mu[0] < mu[-1]:
            df_MT.iloc[:, :N] *= -1

        # check isolation
        # plt.plot(df_MT.iloc[:,:N])
        # ### plt.show()

        for i in range(N):
            I_t = df_MT.iloc[:, i].values

            # find index of capacitance peak
            peak_idx = int(np.argmax(I_t))

            # truncate time and current accordingly
            I_t = I_t.tolist()[peak_idx:]
            times = [t - time[peak_idx] for t in time[peak_idx:]]

            I_prev = df.iloc[startend[k]-fifth -
                             10*khz:startend[k]-10*khz, i].mean()
            dV = df.iat[startend[k]+2*fifth, N+i] - \
                df.iat[startend[k]-2*fifth, N+i]

            dI = max(I_t) - min(I_t)
            popt, pcov = curve_fit(func,
                                   times, I_t,
                                   p0=[dI, 10, np.mean(I_t[-fifth:])],
                                   bounds=([0, 1e-3, -1e3], [1e3, 100, 1e3])
                                   )
            # print(popt)

            # plot exponential fit of capacitive transient
            # ysim = [func(t, *popt) for t in times]
            # plt.plot(times, I_t)
            # plt.plot(times, ysim, ls='--', lw=3)
            # ### plt.show()
            # exit()

            dI, tau, I_ss = popt
            I_peak = I_ss + dI
            I_d = I_peak - I_prev
            I_dss = I_ss - I_prev

            # C_m = Q /dV, where Q is obtained by integrating the capacitive transient
            Q = simps([x - I_ss for x in I_t], times) + I_dss*tau
            Cm = abs(Q/dV)

            # from Molecular Devices (MDC)
            # tau = R * Cm, 1/R = 1/Rm + 1/Ra = (Ra + Rm)/(Ra * Rm)
            # tau/Cm = (Ra*Rm) / (Ra + Rm)
            # Ra^2 - Ra*Rt + Rt*(tau/Cm) = 0, R_t = Ra + Rm
            R_t = abs((dV/I_dss)*1e3)
            quad_factors = [1, -R_t, R_t*(tau/Cm)*1e3]
            # print(R_t, quad_factors)

            R_a = np.min(np.real(np.roots(quad_factors)))

            # Rm = Rt - Ra
            R_m = R_t - R_a

            # test solution
            R_t2 = 1/((1/R_a) + (1/R_m))
            MDC_Rm_error.append(R_t2*Cm - tau*1e3)

            params_MDC.append([R_a, R_m, Cm])

            # From SW Harden:
            # At time zero, access resistance is the thing limiting our ability to deliver current Id to a known dV (Cm doesn't come into play yet). Thus, Ra = dV / Id
            R_a = abs(dV/I_d)*1e3

            # From SW Harden:
            # the difference between this steady state current (Iss) and the last one (Iprev) is limited by the sum of Rm and Ra
            # R_m + R_a = dV / (I_ss - I_prev)
            # R_m = (dV - R_a * (I_ss - I_prev) ) / (I_ss - I_prev)
            R_m = abs((dV*1e-3 - R_a*I_dss*1e-6)/(I_dss*1e-12))*1e-6

            # From SW Harden:
            # When we raise the cell's voltage (Vm) by delivering current through the pipette (Ra), some current escapes through Rm. From the cell's perspective when we charge it though, Ra and Rm are in parallel.
            # C_m = tau / R, 1/R = 1/R_a + 1/R_m
            C_m = abs(tau / (1 / (1/R_a) + (1/R_m))) * 1e3

            # From SW Harden:
            # Rm leaks a small amount of the Id current that passes through Ra to charge Cm. We can calculate a correction factor as the ratio of Ra to Rm and multiply it by both of our resistances. Cm can be corrected by dividing it by the square of this ratio.

            if R_m >= 10*R_a:
                correction = 1 + (R_a / R_m)

                R_a *= correction
                R_m *= correction
                C_m *= 1/(correction**2)

            params_SWH.append([tau, R_a, R_m, C_m])

    for u in range(len(startend)-1):
        do_estimation(u)

    # take average of start and end capacitive transients
    params_SWH = [np.mean(params_SWH[2*i:2*i + 2], axis=0) for i in range(N)]
    params_MDC = [np.mean(params_MDC[2*i:2*i + 2], axis=0) for i in range(N)]

    f, ax = plt.subplots(2, 2, figsize=(12, 6))
    labels = [r"$\tau$ (ms)", r"$R_a$ (M$\Omega$)",
              r"$R_m$ (M$\Omega$)", r"$C_m$ (pF)"]

    h = 0
    sweeps = range(1, N+1)
    for i in range(2):
        for j in range(2):
            h = 2*i + j

            ax[i, j].plot(sweeps, [x[h] for x in params_SWH],
                          marker='o', c='r',
                          alpha=0.7, label="SWH")
            if h > 0:
                ax[i, j].plot(sweeps, [x[h-1] for x in params_MDC],
                              marker='x', markersize=8, c='white',
                              alpha=0.7, label="MDC")

            ax[i, j].legend()
            ax[i, j].set_title(labels[h])

    for i in range(2):
        ax[1, i].set_xlabel("Sweep #")

    f.suptitle("Estimation of Membrane Test Parameters")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if pdf is not None:
        pdf.savefig(bbox_inches='tight')

    # plt.show()
    plt.close()