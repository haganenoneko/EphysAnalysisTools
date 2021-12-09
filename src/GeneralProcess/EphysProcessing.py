# personal libraries, in the same directory
from scipy.stats import pearsonr, sem
from scipy.optimize import minimize
from PyPDF2 import PdfFileReader, PdfFileMerger
from scipy.integrate import simps, romb, trapz
from GeneralProcess.LeakSubtraction import leak_subtract
from GeneralProcess.EphysInfoFilter import EphysInfoFiltering
from GeneralProcess.ActivationCurves import activation_curve
from GeneralProcess.ExpFitting import exp_fitting
from GeneralProcess.AnalyzeRamp import analyze_ramp_dt
from GeneralProcess.PubPlotting import PubPlotting
from GeneralProcess.NormalizeForFitting import normalize_for_fitting
from GeneralProcess.ExtractTraces import apply_bessel, ExtractTraces
from GeneralProcess.FindCustomEpochs import FindCustomEpochs, DummyABF

import numpy as np
import pandas as pd
import re, math, glob, os, pyabf
from collections import OrderedDict

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages


def RC_defaults():
    rcParams['font.size'] = 12
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'Verdana'
    rcParams['font.weight'] = 'normal'
    rcParams['axes.linewidth'] = 2
    rcParams['axes.labelweight'] = 'bold'


# cmap = plt.cm.get_cmap()
cmap = plt.cm.get_cmap("gist_rainbow")


def DataframeStats(df, groupby=None, axis=0):
    """
    Print mean, sem, and count for columns of `df`
    pass `axis=1` if variables are in rows 
    """
    if groupby is None:
        print(pd.concat(
            [df.mean(axis=axis), df.sem(axis=axis), df.count(axis=axis)],
            axis=axis, keys=["Mean", "SEM", "Count"]))
    else:
        print(df.groupby(groupby).mean())
        print(df.groupby(groupby).sem())
        print(df.groupby(groupby).count())


def exp2(v, A1, s1, C):
    return A1*np.exp(-v/s1) + C


def remove_traces(dflis, khz, conds, to_drop, show=False):
    """
    dflis = list of activation and deactivation dataframes  
    conds = conditions, e.g. test voltages/durations  
    to_drop = test voltages/durations to drop, list
    """
    colnames = dflis[0].columns
    # print(colnames)

    for k in to_drop:
        # tuple -> truncation of given trace, (voltage/duration, truncation index)
        # keep values up to the truncation index, e.g. if < 0, then remove from the end
        if isinstance(k, tuple):
            # index of column name
            i = colnames.get_loc(k[0])

            # if 2nd element is a list, it specifies
            #   (0, x, 1, x),
            #   where x are truncation variables for activation (0) or deactivation (1)
            if isinstance(k[1], list):

                for j in range(0, len(k[1]), 2):
                    if k[1][j+1] == 0:
                        continue
                    else:
                        k[1][j+1] *= khz

                        # truncate start
                        if k[1][j+1] > 0:
                            print(k[1][j+1])
                            print(dflis[k[1][j]].iloc[:, i].shape[0])

                            dflis[k[1][j]].iloc[:k[1][j+1], i] = np.nan

                        # truncate end
                        else:
                            dflis[k[1][j]].iloc[k[1][j+1]:, i] = np.nan

            # apply truncation to both dataframes
            else:
                for j in range(2):
                    if k[1] > 0:
                        dflis[j].iloc[:k[1], i] = np.nan
                    else:
                        dflis[j].iloc[k[1][j+1]:, i] = np.nan

    # voltages to drop
    for k in to_drop:
        if isinstance(k, int) and k in conds:
            conds.remove(k)
            dflis[0].drop(k, axis=1, inplace=True)
            dflis[1].drop(k, axis=1, inplace=True)

    # remove nans and reset index, columns
    for i in range(2):
        dflis[i] = dflis[i].apply(lambda x: pd.Series(x.dropna().values))
        # ensure index matches sample frequency
        dflis[i].index *= 1/khz
        # dflis[i].columns = conds

    # check removal
    if show:
        f, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(dflis[0])
        ax[1].plot(dflis[1])
        plt.tight_layout()
        plt.show()
        plt.close()
        # exit()

    return dflis[0], dflis[1], conds


def BC_VoltageClampQuality(fname, params, pdfs=None, show=False):
    """
    Compute corner frequency and visualize frequency vs Vout/Vin plot.  

    fname = filenam   
    params = pd Series containing experimental parameters for file  
    pdfs = multipage pdf object to save plots  

    References:
    http://www.billconnelly.net/?p=310
    https://www.electronics-tutorials.ws/filter/filter_2.html
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

    f, ax = plt.subplots(figsize=(9, 4))
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

    plt.tight_layout()

    if pdfs is not None:
        pdfs.savefig(bbox_inches='tight')

    if show:
        plt.show()

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


def get_upper_index(df, pname):
    """
    Find upper index (+1, i.e. 1-indexed) that bounds the test pulse 
    `df` = current dataframe 
    `pname` = protocol name 

    NOTE: 
        - inverted "act...env" protocols estimate conductance with a maximally actvating current, rather than a deactivating tail current                                     
    """
    N = int(df.shape[1]/2)  # number of traces
    if "ramp_dt" in pname:
        return 3
    elif "_act" in pname:
        return 2
    elif "_de" in pname:
        # `ramp_de` and `de` are similar 2-step protocols where the 2nd is of interest
        return 3
    elif "env" in pname:
        # end of 2nd hyperpolarization
        return 4
    # elif "ramp_de" in pname:
    else:
        print("Upper bound for test pulse of {pname} not provided.".format(
            pname=pname))
        return None


class process():
    def __init__(self,
                 filter_criteria={},
                 show_protocols=True,
                 files_to_skip=[],
                 dates_to_save='None',
                 show_abf_segments=False,
                 show_csv_segments=False,
                 show_leak_subtraction=False,
                 do_pseudoleak_subtraction=False,
                 do_LJP_correction=(False, 3.178),
                 VoltageClampQuality=False,
                 show_Cm_estimation=False,
                 show_MT_estimation=False,
                 do_exp_kinetics={},
                 manual_cap_offset={},
                 do_activation_curves={},
                 do_ramp_stuff=False,
                 do_inst_IV=False,
                 save_AggregatedPDF=False,
                 do_pubplots={},
                 normalize=False,
                 remove_after_normalize={}
                 ):
        """
            # import and read data 
            `dates` = list of dates in ephys_info.xlsx, e.g. 20916, 209, etc.  
            `protocol_name` = protocols named in used_protocols_v#.csv  
            `files_to_skip` = files to skip, list of strings   
            `dates_to_save` = dates for which to use in summary figures  
            `show_protocols` = prints named protocols recorded in `dates`  

            # protocol segmentation 
            `show_abf_segments` = plots epoch segmentation of protocol  
            `show_csv_segments` = show segmentation of .csv file (data)  

            # preliminary processing 
            `show_leak_subtraction` = show data before and after leak-subtraction  
            `show_Cm_estimation` = show membrane capacitance estimations using voltage ramps (S.W. Harden's method)  
            `show_MT_estimation` = show estimation of Cm, Rm, and Ra by fitting exponential to a single membrane test step (+20)  
            `do_LJP_correction` = tuple (Boolean, Float). If first element is True, then subtracts the second value (if > 0) from all voltage commands. 

            # summary analyses 
            `do_exp_kinetics` = fit 1-3 order exponential functions to (leak-subtracted) current traces   
            `do_activaiton_curves` = compute G-V curves and Pmin values from leak-subtracted current traces. Fit with modified Boltzmann.  
            `do_ramp_stuff` = compute hysteresis statistics, e.g. integrated area, difference in current/integrated area, delay, etc.  
            `do_inst_IV` = compute I-V curves and related parameters (Erev, P_K, P_Na, I/Cm) using instantaneous current and membrane capacitance  

            # plotting 
            Each function generally has a plotting function `show.` Unless specified, these are separate plots.   
            `save_AggregatedPDF` = save multiple plots to one PDF file, [directory] \ [filename]_[protocol].pdf  
            `do_pubplots` = make nice looking plots for recordings with current + voltage data 

            # preparation for fitting
            `normalize` = apply normalization   
            `remove_after_normalize` = dictionary of {filename : test voltages/durations} that specifies traces to remove from dataframe before normalization  
                i.e. requires at least one prior run of normalization to identify traces to remove  
        """

        if len(filter_criteria.keys()) < 1:
            raise Exception(" No filter criteria provided.")
        else:
            read_ephys_info = EphysInfoFiltering(filter_criteria)
            filenames, ephys_info = read_ephys_info.filter()
            paired_files, exp_params = read_ephys_info.ExpParams(ephys_info)

        # path to main directory
        main_dir = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/Github_repos/hcn-gating-kinetics/"
        # path to csv files
        data_path = main_dir + r"data/current_time_course/"

        # load csv files containing corresponding data
        paths = [data_path + "%s.csv" % f for f in filenames]
        data_files = {}

        # find csv files
        to_remove = []  # indices of CSV files that weren't found
        for i, x in enumerate(paths):
            if os.path.exists(x):
                df = pd.read_csv(x, header=None, index_col=0)

                # check if index name is not 0; if so, remove header row
                if df.index[0] != 0:
                    df = df.iloc[1:, :]
                    # change index type to float
                    df.index = df.index.astype(float)

                df = file_specific_transform(filenames[i], df=df)

                data_files.update(
                    {filenames[i]: df})
            else:
                print("CSV file not found. ", filenames[i])
                print(" Removing %s from processing." % filenames[i])
                to_remove.append(i)

                # remove file not found from dataframes of recording parameters
                ephys_info = ephys_info[ephys_info['Files'] !=
                                        filenames[i]]
                exp_params.drop(filenames[i], axis=0, inplace=True)

        # remove from `filenames` if CSV file not found
        if len(to_remove) > 0:
            print("     Files kept: \n",
                  ephys_info.loc[:, ["Files", "Protocol"]])

            h = 0
            for t in to_remove:
                del filenames[t-h]
                h += 1

        # path to raw abf files
        abf_path = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/data_files/delbert/"

        # search subdirectories of entire `abf_path` for the original .abf files
        abf_files = [
            glob.glob(abf_path + "**/%s*.abf" % f, recursive=True) for f in filenames
        ]
        abf_files = [pyabf.ABF(a[0]) for a in abf_files]

        # check that abf files were found
        if len(abf_files) != len(filenames):
            print("Number of filenames is %d, but only found %d abf files" % (
                len(filenames), len(abf_files)))
            exit()

        # create dictionary that holds, for each unique filename, start and end of first leak ramp
        self.ramp_startend = {}
        # dict to hold start and end of +20mV membrane test steps
        self.mt_startend = {}
        # dictionary to hold filename : {epoch1:[t0, t1], ...}
        self.epoch_startends = {}
        # dictionary to hold sampling frequencies in kHz
        self.dataRates = {}

        # loop over .ABF files found, and find start and end of leak ramps
        for i, a in enumerate(abf_files):
            print("\n Reading...", filenames[i])
            df_data = data_files[filenames[i]]

            # sampling frequency in kHz
            khz = int(a.dataRate / 1000)
            self.dataRates.update({filenames[i]: khz})

            # number of sweeps in ith recording
            N = int(df_data.shape[1]/2)

            # check that protocol is contained in CSV file
            # 1. even number of columns in CSV file
            # 2. voltage is in range [-200, +100]
            try:
                HasProtocol = (df_data.shape[1] == 2*N) and \
                    (df_data.iloc[:, N:].min() >= -200).all() and \
                    (df_data.iloc[:, N:].max() <= 100).all()
            except:
                HasProtocol = False

            # check if epochs are available
            # 1. max epoch time exceeds duration of data
            # 2. sum of abf.sweepC and data_df.iloc[:,N] are nearly equal
            # note that 2) doesn't work unless `df_data.iloc[:,N]` is a voltage command
            a.setSweep(0)
            EpochTimes = a.sweepEpochs.p1s
            EpochLevels = a.sweepEpochs.levels
            EpochTypes = a.sweepEpochs.types

            try:
                HasEpochs = (df_data.shape[0] >= EpochTimes[-1]) or \
                    (abs(df_data.iloc[:, N].sum()/sum(a.sweepC)) > 0.9)
            except:
                HasEpochs = False

            # find epochs for custom protocol if used
            if HasEpochs and HasProtocol:
                pass
            else:
                print(
                    "Uneven number of columns of CSV. Protocol may be missing.\
                    \n First row: \n", df_data.iloc[0, :]
                )

                # protocol name
                pname = ephys_info.loc[ephys_info["Files"] == filenames[i],
                                       "Protocol"].values[0]

                # check if protocol corresponds to an existing CSV file
                if os.path.isfile(r"./data/protocols/%s.csv" % pname):

                    # retrieve epochs for custom protocol, either in CSV, or automated
                    CustomEpochs = FindCustomEpochs(
                        df_data, pname, filenames[i],
                        a, show_epochs=False, save=False
                    )

                    # abf variables (sweep epochs, times, levels, protocol time course)
                    a = CustomEpochs.ReplaceABFVariables(test=True)
                    abf_files[i] = a
                    khz = a.dataRate

                    # redefine epoch properties for first sweep
                    a.setSweep(0)
                    EpochTimes = a.sweepEpochs.p1s
                    EpochLevels = a.sweepEpochs.levels
                    EpochTypes = a.sweepEpochs.types

                    # redefine `df` if it doesn't contain protocol
                    if CustomEpochs.HasProtocol == False:
                        df_data = CustomEpochs.df
                        data_files[filenames[i]] = df_data

                        # redfine number of sweeps
                        N = int(CustomEpochs.df.shape[1]/2)

                    print("pyABF object replaced.")

                else:
                    raise Exception(
                        "`Protocol` entry in `ephys_info` < %s > does not correspond\
                        to a CSV file at './data/protocols/'. If it is a custom protocol,\
                        then make sure such a file is available." % pname
                    )
                        
            # apply LJP correction
            # for pyABF (non-custom protocol) objects, a DummyABF object is created,
            # so some slowness is expected
            if all(x > 0 for x in do_LJP_correction):
                # subtract from voltage command
                df_data.iloc[:, N:] -= do_LJP_correction[1]

                # create DummyABF object to replace original pyABF
                # because, pyABF object doesn't mutate well
                a = DummyABF(a, [], [], khz)

                # apply LJP subtraction to each sweep of voltage command in ABF,
                # regardless of whether it's included in CSV file or not
                a.pro -= do_LJP_correction[1]
                for j in a.sweepList:
                    a.EpochLevels[j] -= do_LJP_correction[1]

                # redefine 0-th sweep properties for finding membrane test and leak ramps
                a.setSweep(0)
                EpochTimes = a.sweepEpochs.p1s
                EpochLevels = a.sweepEpochs.levels
                EpochTypes = a.sweepEpochs.types

                # replace in dictionary
                data_files[filenames[i]] = df_data

            # find first voltage ramp for the first trace, then assume this is constant
            # plot segmented epochs from protocol
            if show_abf_segments:
                f, ax = plt.subplots(2, 1, figsize=(8, 6))

                for sweepNumber in a.sweepList:
                    a.setSweep(sweepNumber)

                    times = np.array(a.sweepX) * 1000
                    ax[0].plot(times[::2], a.sweepY[::2],
                               c='gray', alpha=0.4)
                    ax[1].plot(times[::2], a.sweepC[::2], c='r', alpha=0.4)

                    # print("Step times... \n", a.sweepEpochs.p1s)
                    for x, p1 in enumerate(a.sweepEpochs.p1s):
                        for j in range(2):
                            ax[j].axvline(p1/khz, c='white', ls='--', lw=2)

                        # ax[1].axhline(a.sweepEpochs.levels[x], c='w', ls='--', alpha=0.3)

                ax[0].set_title("show_abf_segments: %s" % filenames[i])
                ax[0].set_ylabel("Current")
                ax[1].set_ylabel("Command Voltage")
                ax[1].set_xlabel("Time")
                plt.tight_layout()

                print(
                    "Showing ABF segments. White dashes show epochs in voltage protocol.")
                plt.show()
                plt.close()

            # find indices for the start and end of the leak ramps
            if "Ramp" in EpochTypes[:10]:
                # find index of first ramp epoch
                # fri = 'first ramp index'
                fri = EpochTypes[:10].index('Ramp')
                # print(EpochTypes[:10], fri)

                # last index of initial ramps; lri = 'last ramp index'
                lri = fri + 1
                while EpochTypes[lri+1] == 'Ramp' and lri <= 10:
                    lri += 1

                t0 = EpochTimes[fri]
                t1 = EpochTimes[lri+1]

                self.ramp_startend.update(
                    {filenames[i]: file_specific_transform(
                        filenames[i], times=[t0, t1])}
                )
            else:
                print("No leak ramp found in first 10 steps of %s." %
                      filenames[i])

                # no leak ramp is present, so we set to `None` to facilitate later processing
                fri = None
                lri = None

            # check for presence of membrane test step and when relevant protocol epochs start
            # if no leak ramp, then we skip testing for membrane test
            if fri is None:

                # when no leak ramp is found, we set `fri` = 0
                # this causes epochs to be cataloged starting from the very beginning
                fri = 0

                # we pre-emptively set `dt_` to None so we can look for it later
                dt_ = None

            else:
                # check if membrane test step is present within 1-5 epochs after the ramp
                # 1. duration < 1s
                # 2. level > 0 mV
                # 3. type = step

                mt_idx = next(
                    (j for j in range(1, 3) if EpochLevels[lri+j] == 20 and
                     (EpochTimes[lri+j+1] - EpochTimes[lri+j]) <= 500*khz and
                     EpochTypes[lri+j] != "Ramp"),
                    None
                )
                
                if mt_idx is None:
                    dt_ = None
                else:
                    mt_idx += lri

                    dt_ = mt_idx + next(
                        (j for j in range(1, 3) if
                         EpochLevels[mt_idx+j] != EpochLevels[0]), 1
                    )

                    self.mt_startend.update({
                        filenames[i]:
                        EpochTimes[mt_idx:dt_]
                    })

            # `dt_` = number of epochs b/w end of leak ramp and start of protocols
            # if `dt_` not found based on where leak ramp and membrane test are,
            # then relevant pulses start after the last -35mV step
            if dt_ is None:
                if lri is None:
                    dt_ = 1 + next(
                        (j for j in range(1, 3)
                         if EpochLevels[j+1] != EpochLevels[0]),
                        0)
                else:
                    dt_ = lri + next(
                        (j for j in range(1, 3)
                         if EpochLevels[j+lri] != EpochLevels[0]),
                        0)
            
            if isinstance(dt_, int):
                print(" %d epochs to start of relevant protocol" % dt_)
            else:
                raise Exception("`dt_` is not an integer: \n ", dt_)

            # 0-indexed list of indices of sweeps, e.g. range(# sweeps)
            # print(a.sweepList)

            # add epoch start and end times for each sweep
            # if csv and abf are 1:1, `h` stays 0
            # else, `h += 1` so that we keep indexing csv at the jth column 
            # while proceeding through the abf file
            h = 0

            # add epoch start and end times for each sweep
            for j in a.sweepList:
                a.setSweep(j)

                # we don't need to compare CSV and ABF sweeps if the number of sweeps are equal
                if N < len(a.sweepList):

                    # don't exceed number of traces in the .csv file
                    if (N + j - h) >= 2*N:
                        print("Exceeded CSV columns while reading epoch intervals")
                        break

                    # sum ABF voltage command
                    abf_sum = np.sum(a.sweepC)
                    if not isinstance(abf_sum, float):
                        raise Exception(
                            "`abf_sum` is not a float.\
                            Check the type of `a.sweepC` : ", type(a.sweepC)
                        )

                    # sum jth sweep of voltage protocol
                    csv_sum = df_data.iloc[:, N+j-h].sum()

                    # apply file-specific transform if necessary to enforce equality
                    csv_sum = file_specific_transform(
                        filenames[i], times=(abf_sum, csv_sum))

                    # print(df_data.iloc[0,:])
                    # plt.plot(df_data.index, a.sweepC, ls='--', lw=2)
                    # plt.plot(df_data.iloc[:,N+j-h], alpha=0.5)
                    # plt.title("abf = %d, csv = %d" % (j, N+j-h))
                    # plt.show()

                    # check if jth sweep is in both abf and csv files by
                    # subtracting their sums (voltage commands)
                    # print(j, h, ((abf_sum - csv_sum)/abf_sum))
                    if abs((abf_sum - csv_sum)/abf_sum) < 0.01:
                        if filenames[i] in self.epoch_startends.keys():
                            self.epoch_startends[filenames[i]].update({
                                j: a.sweepEpochs.p1s[dt_:]
                            })
                        else:
                            self.epoch_startends.update({
                                filenames[i]: {
                                    j: a.sweepEpochs.p1s[dt_:]
                                }
                            })
                    # if the jth sweep isn't found in the .csv file, we skip it
                    else:
                        h += 1
                    
                else:

                    if filenames[i] in self.epoch_startends.keys():
                        self.epoch_startends[filenames[i]].update({
                            j: a.sweepEpochs.p1s[dt_:]
                        })
                    else:
                        self.epoch_startends.update({
                            filenames[i]: {
                                j: a.sweepEpochs.p1s[dt_:]
                            }
                        })

            # verify that epochs have been added for the current file
            if filenames[i] not in self.epoch_startends.keys():
                raise Exception(
                    "No epochs were saved for file < %s >,\
                    this may mean that the protocol was not saved in the CSV file.\
                    if a custom protocol was used, check that the protocol was parsed correctly. \
                    finally, if LJP subtraction is enabled, check that it is applied equally for both CSV and ABF data." % filenames[i]
                )
            
            # check same number of sweep intervals saved as number of sweeps 
            if len(self.epoch_startends[filenames[i]].keys()) < df_data.shape[1]/2:
                print(df_data.shape)
                print(self.epoch_startends[filenames[i]])
                raise Exception("Intervals were not saved for every sweep.")
            
            # apply transform to `mt_startend` and `epoch_startends` if file requires it
            self.mt_startend = file_specific_transform(
                filenames[i], times=self.mt_startend)
            self.epoch_startends = file_specific_transform(
                filenames[i], times=self.epoch_startends)
            
            # plot segmented epochs on .csv file
            # this goes after `show_abf_segments` because it requires `self.epoch_startends` to be filled out
            if show_csv_segments:

                # number of traces
                # N = int(df_data.shape[1]/2)

                f, ax = plt.subplots(2, 1, figsize=(14, 5))
                for j, g in enumerate(self.epoch_startends[filenames[i]].keys()):
                    ax[0].plot(df_data.iloc[::5, j], lw=2, alpha=0.4)
                    ax[1].plot(df_data.iloc[::5, N+j], lw=2, alpha=0.4)

                    for k in range(2):
                        # epochs for sweep `g` of file `i`
                        E = self.epoch_startends[filenames[i]][g]

                        for n, u in enumerate(E):
                            # remove epoch if it (e.g. transformed) exceeds dimensions
                            if u >= df_data.shape[0]:
                                self.epoch_startends[filenames[i]][g] = E[:n]
                                break
                            else:
                                ax[k].axvline(u/khz, c='white', ls='--')

                ax[0].set_title("show_csv_segments: %s" % filenames[i])
                ax[0].set_ylabel("Current")
                ax[1].set_ylabel("Voltage")
                ax[1].set_xlabel("Time (ms)")
                plt.tight_layout()

                print(
                    "Showing CSV segments. White dashes show epochs in voltage protocol.")
                plt.show()
                plt.close()
                # exit()

        # assign self variables for class
        self.dates_to_save = dates_to_save
        self.main_dir = main_dir
        self.save_path = main_dir + "output/Processing/Pooled_Analyses/"
        self.ephys_info = ephys_info

        # experimental recording parameters
        self.exp_params = exp_params

        # recordings from same cell
        # {parent filename : [[subsequent files], [activation files]]}
        self.paired_files = paired_files
        self.filenames = filenames
        self.data_files = data_files
        self.abf_files = abf_files

        # create prefix for output without skipped filenames or exclusion criteria
        self.output_prefix = read_ephys_info.CreatePrefix(
            exclusion=False, skip=False)

        # for downstream options
        # self.show_protocols = show_protocols
        self.show_abf_segments = show_abf_segments
        self.show_csv_segments = show_csv_segments
        self.do_pubplots = do_pubplots
        self.VoltageClampQuality = VoltageClampQuality
        self.do_pseudoleak_subtraction = do_pseudoleak_subtraction
        self.show_leak_subtraction = show_leak_subtraction
        self.manual_cap_offset = manual_cap_offset
        self.show_Cm_estimation = show_Cm_estimation
        self.show_MT_estimation = show_MT_estimation
        self.do_ramp_stuff = do_ramp_stuff
        self.do_exp_kinetics = do_exp_kinetics
        self.do_activation_curves = do_activation_curves
        self.do_inst_IV = do_inst_IV
        self.normalize = normalize
        self.remove_after_normalize = remove_after_normalize
        self.save_AggregatedPDF = save_AggregatedPDF

        # for computations
        self.exp_fit_params = []    # parameters for exp1-3 fitting
        self.exp_fit_delay = {}     # delay in exp1
        self.ac_fit_params = []     # parameters for activation curve
        self.ac_norm_data = []      # normalized activation curve
        self.tail_post_pmins = []   # Pmin from post-pulse currents
        self.ramp_stats = []        # Ramp hysteresis statistics
        self.IV_params = []         # Summary statistics from I-V curves: Erev, P_K, P_Na
        self.IV_currents = []       # instantaneous current and current dnesity vs. voltage

    def etc(self, return_traces=False, save_normalized=False,
            save_leaksub=True, save_extracted=True):
        """
        Runs processing functions.  

        `return_leaksub` = return leak subtracted traces and extracted test pulses, each dictionaries with filenames as keys 
            leaksub = {filename : dataframe}
            extracted = {filename : {'data' : dataframe, 'protocol' : dataframe}}
        `save_leaksub` = whether to save leak-subtracted data  
        `save_extracted` = whether to save leak-subtracted, extracted data  
        """

        filenames = self.filenames
        ephys_info = self.ephys_info
        data_files = self.data_files

        # hold leak subtracted dataframes, {filename : leak-subtracted dataframe}
        data_files_Lsub = {}
        extracted_data = {}

        # name for saving temporary files when fitting exp kinetics
        if "save_path" in self.do_exp_kinetics.keys():
            if self.do_exp_kinetics["save_path"] is not None:
                tmp_exp_name = self.do_exp_kinetics["save_path"] + \
                    self.output_prefix
            else:
                tmp_exp_name = self.save_path + self.output_prefix

        for i, df in enumerate(data_files.values()):
            plt.style.use('default')
            RC_defaults()
            print("\n Leak subtracting... ", filenames[i])

            khz = int(self.dataRates[filenames[i]])
            N = int(df.shape[1]/2)

            # copy dataframe
            df1 = df.copy()

            # step intervals for sweeps in current file
            try:
                intervals = self.epoch_startends[filenames[i]]
            except:
                print("Files with defined epochs: \n   ",
                      self.epoch_startends.keys())
                raise Exception(
                    "%s does not have any epochs set." % filenames[i])

            # get name of protocol for the current recording
            try:
                pname = ephys_info.loc[ephys_info['Files']
                                       == filenames[i], "Protocol"].iat[0]
            except:
                plt.plot(df1.iloc[:, :int(df1.shape[1]/2)])
                # plt.show()
                plt.close()

                print("Couldn't find protocol name for %s" % filenames[i])
                print(filenames[i], ephys_info.loc[:, ['Files', 'Protocol']])
                raise

            # create PDF for holding multiple plots
            if self.save_AggregatedPDF:
                s = r"{savedir}/PDFs/{fname}-{pname}.pdf".format(
                    savedir=self.save_path, fname=filenames[i], pname=pname)
                PDFPlots = PdfPages(s)

                print("Created PdfPages object, which will be saved at: \n  {pdfloc}".format(
                    pdfloc=s))
            else:
                PDFPlots = None

            # if no leak ramp, skip leak subtraction
            Lsub = None
            if filenames[i] in self.ramp_startend.keys():
                # create leak subtract object, then perform subtraction
                Lsub = leak_subtract(
                    self.ramp_startend[filenames[i]],
                    khz=khz, epochs=intervals, residual=False
                )
                # perform leak subtraction
                df1 = Lsub.do_leak_subtraction(
                    df1, method="ohmic",
                    plot_results=self.show_leak_subtraction, pdfs=PDFPlots
                )

                # save leak-subtracted dataframe
                s = f"{self.main_dir}/output/Processing/Processed_Time_Courses/leaksub/"
                s += f"{filenames[i]}_leaksub.csv"
                
                if save_leaksub:
                    df1.to_csv(s)

                data_files_Lsub.update({filenames[i]: df1})

            elif self.do_pseudoleak_subtraction and "_act" in pname:
                # exclude sine, RR, ramp or env
                if all(s not in pname for s in ['sine', 'RR', 'ramp', 'env']):
                    print("%s is an activation protocol, but no leak ramp found. A pseudo-leak subtraction will be done using holding, activation onset, \
                        and tail offset current/voltage levels." % filenames[i])

                    # initialize leak_subtract class with None as ramp start/end times
                    Lsub = leak_subtract(
                        None, khz=khz, epochs=intervals, residual=False)

                    # don't use tails for F431A
                    if "F431A" in self.ephys_info.loc[self.ephys_info["Files"] == filenames[i], "Construct"]:
                        df1 = Lsub.PseudoLeakSubtraction(
                            df, use_tails=False, plot_results=self.show_leak_subtraction)
                    else:
                        df1 = Lsub.PseudoLeakSubtraction(
                            df, use_tails=True, plot_results=self.show_leak_subtraction)

                    data_files_Lsub.update({filenames[i]: df1})

            # experimental seal/whole cell parameters
            # if any lists in experimental parameters, take the mean of each element
            if np.any([isinstance(x, list) for x in self.exp_params.loc[filenames[i], :]]):
                expp = [np.mean(x)
                        for x in self.exp_params.loc[filenames[i], :]]
            else:
                expp = self.exp_params.loc[filenames[i], :]

            # quality of voltage clamp wrt frequency, shows corner frequency
            if self.VoltageClampQuality:
                BC_VoltageClampQuality(
                    filenames[i], expp, pdfs=PDFPlots, show=False)

            if self.show_leak_subtraction:
                if len(self.ramp_startend.keys()) < 1 or \
                        filenames[i] not in self.ramp_startend.keys():

                    print(
                        " No leak subtraction can be done, because current file < %s > does not have leak ramp epochs." % filenames[i])
                    print(
                        "Be careful about GV results. Baseline may be unreliable without leak subtraction.")

                else:
                    print(" Showing leak subtraction for %s..." % filenames[i])
                    RC_defaults()

                    f, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
                    N = int(df.shape[1]/2)
                    
                    for j in range(N):
                        ax.plot(df.iloc[::4, j], lw=1.2, c='gray', 
                                label="Original" if j == 0 else "")
                        ax.plot(df1.iloc[::4, j], lw=1.8, c='r', 
                                label="Subtracted" if j == 0 else "")

                    ax.set_ylabel("Current (pA)")
                    ax.legend(loc="lower right", bbox_to_anchor=[0.81, 0])

                    ax.set_title("Leak Subtraction using Linear Voltage Ramps")
                    ax.set_xlabel("Time (ms)")

                    # Label for experimental parameters 
                    s = f"{filenames[i]}\n" + expp.to_string()
                    s = s.replace("(M)", "(M$\Omega$)")
                    s = s.replace("(G)", "(G$\Omega$)")
                    
                    ax.text(0.987, 0.35, s, ha='right', 
                            va='center', transform=ax.transAxes, )

                    ax.locator_params(axis='x', nbins=5)
                    ax.locator_params(axis='y', nbins=4)

                    if self.save_AggregatedPDF:
                        PDFPlots.savefig(bbox_inches='tight')

                    plt.show()
                    plt.close()

            if self.show_Cm_estimation:
                print(" Cm estimation for %s..." % filenames[i])

                cm_values = estimate_Cm(
                    self.ramp_startend[filenames[i]], df, khz=khz)
                plt.plot(range(N), cm_values, marker='o')

                plt.xticks(range(N))

                plt.title(r"$C_m$ estimation")
                plt.tight_layout()

                if self.save_AggregatedPDF:
                    PDFPlots.savefig()

                plt.show()
                plt.close()

            if self.show_MT_estimation:
                if filenames[i] in self.mt_startend.keys():
                    print(" Membrane test estimation for %s..." % filenames[i])
                    estimate_MT(
                        self.mt_startend[filenames[i]], df, khz=khz, pdf=PDFPlots)

            # get upper index that bounds test pulse in protocol `pname`
            u = get_upper_index(df, pname)
            if u is None:
                print("No upper index found for file < %s >" % filenames[i])
                continue

            if self.do_pubplots:
                print(" Making pub plots for %s..." % filenames[i])

                if self.do_pubplots["leaksub"] == True:
                    if filenames[i] in self.ramp_startend.keys():
                        pubplot = PubPlotting(df1, intervals, filenames[i], khz=khz,
                                              pdfs=PDFPlots, **self.do_pubplots)

                    # if leak subtraction was not done, change filename of saved figures to reflect this
                    else:
                        tmp_ = self.do_pubplots
                        tmp_["leaksub"] = False

                        print(
                            "`leaksub = True` for self.do_pubplots, but leak subtraction wasn't done for %s, so pubplots will be saved with suffix '_x' instead of '_leaksub'." % filenames[i])

                        pubplot = PubPlotting(df1, intervals, filenames[i], khz=khz,
                                              pdfs=PDFPlots, **tmp_)

                else:
                    pubplot = PubPlotting(df, intervals, filenames[i], khz=khz,
                                          pdfs=PDFPlots, **self.do_pubplots)

                # generate figure
                pubplot.make_figure()

            # extract test pulses from data (leak-subtracted if leak subtraction occurred)
            print(" Extracting test pulses for %s..." % filenames[i])

            if "env" in pname:
                if len(intervals[0][:u]) < 4:
                    u += 1
                # print(u, intervals)

                # df_to_fit = list of concatenated dataframes containing:
                v_, env_times, df_to_fit, df_protocol = ExtractTraces(df1, u,
                                intervals, N, khz, filenames[i], env=True, show=False,
                                manual_cap_offset=self.manual_cap_offset).extract(return_voltages=True)

            else:
                # if ramp, return half-durations along with current
                if "ramp_dt" in pname:
                    tmids, df_to_fit, df_protocol = ExtractTraces(df1, u, intervals, N, khz,
                                                                  filenames[i], ramp="dt", manual_cap_offset=self.manual_cap_offset).extract(return_voltages=True)
                elif "ramp_de" in pname:
                    tmids, df_to_fit, df_protocol = ExtractTraces(df1, u, intervals, N, khz,
                                                                  filenames[i], ramp="de", manual_cap_offset=self.manual_cap_offset).extract(return_voltages=True)
                # else, return test pulse voltages
                else:
                    vtest, df_to_fit, df_protocol = ExtractTraces(df1, u, intervals, N, khz,
                                                                  filenames[i], show=False, manual_cap_offset=self.manual_cap_offset).extract(return_voltages=True)

            # save extracted data
            if type(df_to_fit) == list:
                extracted_data.update(
                    {filenames[i]: {"data": df_to_fit,
                                    "protocol": df_protocol}}
                )
            else:
                extracted_data.update(
                    {filenames[i]: {"data": [df_to_fit],
                                    "protocol": df_protocol}}
                )

            # save extracted data (leak-subtracted if leak subtraction was done)
            if save_extracted:
                # path to directory to save files in
                s = r"{maindir}/output/Processing/Processed_Time_Courses/leaksub/extracted/{fname}_leaksub_extracted.csv".format(
                    maindir=self.main_dir, fname=filenames[i])

                # append voltage protocol columnwise
                try:
                    pd.concat([df_to_fit, df_protocol], axis=1).to_csv(s)
                    print("Saved extracted data successfully at < %s > " % s)
                except:
                    print("Failed to save extracted data.")

            if self.do_ramp_stuff and ("ramp_dt" in pname):
                print(" Ramp analysis for %s..." % filenames[i])

                # _, df_, pro = ExtractTraces(df1, u, intervals, N,
                #                           khz, ramp="dt").extract(return_voltages=True)

                df_ = extracted_data[filenames[i]]["data"][0]
                pro = extracted_data[filenames[i]]["protocol"]

                # initialize class object for hysteresis summaries
                ramp_H = analyze_ramp_dt(pro, df_, tmids, N, khz)

                # run analyses and return statistics
                df_ramp_stats, _ = ramp_H.H(plot=True, pdf=PDFPlots)
                df_ramp_stats["Rates"] = df_ramp_stats.index
                df_ramp_stats.reset_index(drop=True, inplace=True)

                # switch
                df_ramp_stats.index = [
                    filenames[i]] * df_ramp_stats.shape[0]
                self.ramp_stats.append(df_ramp_stats)

            if self.do_exp_kinetics:
                if "env" in pname:
                    # retrieve extracted pulses
                    df_lis = extracted_data[filenames[i]]["data"]

                    # there should be 3 elements: activation, tail, and 2nd activation
                    if len(df_lis) != 3:
                        raise Exception(
                            "Extracted envelope data should have 3 elements, but currently found %d" % len(df_lis))

                    # assign envelope durations to column labels of extracted data
                    for j in range(3):
                        df_lis[j].columns = env_times

                    # sort envelope times, then re-order columns of df_lis
                    env_times.sort()
                    for j in range(3):
                        df_lis[j] = df_lis[j].loc[:, env_times]

                    fitting = exp_fitting(df_lis, filenames[i], khz)
                    fit_params = fitting.FitEnvelope(env_times, show=True)

                    v_ = [math.ceil(v) for v in v_]
                    v_.extend(fit_params)
                    v_.insert(0, filenames[i])

                    self.exp_fit_params.append(
                        pd.Series(data=v_,
                                  index=["FileName", "Activation", "Deactivation",
                                         "A", "tau", "C"])
                    )

                else:
                    # retrieve extracted data, and make a copy
                    df_ = extracted_data[filenames[i]]["data"][0]

                    # assign test voltages as column labels
                    df_.columns = vtest
                    # sort test voltages, then re-arrange column labels
                    vtest.sort()
                    df_ = df_.loc[:, vtest]

                    # initialize exponential fitting object
                    fitting = exp_fitting(df_, filenames[i], khz, volts=vtest)
                    # do fitting
                    fit_params, exp_delay = fitting.fit_traces(
                        pdf=PDFPlots, **self.do_exp_kinetics)

                    # convert exp fit params into dataframe, then add
                    fit_params = pd.DataFrame.from_dict(fit_params)
                    fit_params.columns = vtest

                    # add filename column
                    fit_params.insert(loc=0, column='FileName',
                                      value=[filenames[i]]*fit_params.shape[0])
                    self.exp_fit_params.append(fit_params)

                    # save every 5 files (overwrite)
                    if "save_path" in self.do_exp_kinetics and \
                            len(self.exp_fit_params) % 5 == 0:

                        pd.DataFrame.from_dict(self.exp_fit_delay).to_csv(
                            tmp_exp_name + "_exp_delay__tmp.csv"
                        )
                        pd.concat(self.exp_fit_params, axis=0).to_csv(
                            tmp_exp_name + "_exp_params__tmp.csv"
                        )

                        print("Temporary data saved successfully at < %s >" %
                              tmp_exp_name)

                    # extract delay for exp1, then add
                    exp_delay = {vtest[j]: x for j, x in enumerate(exp_delay)}
                    self.exp_fit_delay.update({filenames[i]: exp_delay})

            # compute activation curve and tail pmins from leak-subtracted data `df1`
            if self.do_activation_curves:
                # skip `ramp_dt` (ie hysteresis) protocols, which might slip through
                if ("ramp" in pname) or ("env" in pname):
                    pass
                else:
                    if "_act" in pname:
                        print(" Activation curve and/or Pmins for %s..." %
                              filenames[i])

                        # use `u+1`, since GV calculation is concerned with the tail steps, not the test pulses
                        print("Extracting tail pulses...")

                        v_, df_to_ac = ExtractTraces(df1, u+1, intervals, N, khz,
                                                     filenames[i], show=True, ramp="x",
                                                     manual_cap_offset=self.manual_cap_offset
                                                     ).extract()

                        # append tails to extracted data
                        extracted_data[filenames[i]]["data"].append(df_to_ac)

                        if df_to_ac.shape[0] < 1:
                            raise Exception(
                                "No rows kept after extraction of leak-subtracted dataframe for file < %s >" % fienames[i])

                        # compute activation curve
                        ac = activation_curve(vtest, df_to_ac, khz, plot_results=True,
                                              pdf=PDFPlots, **self.do_activation_curves)

                        # fitting is done in the previous step; here, we just get the parameters and normalized current values
                        norm_tails, ac_pars = ac.do_fit()
                        # pmins
                        # tail_pmins = ac.tail_pmins()

                        # we have to redefine `vtest` here because `activation_curves` sorts voltages from low to high, which may not be the order of the test pulses
                        norm_tails = pd.DataFrame(
                            data={filenames[i]: norm_tails})
                        vtest = ac.return_test_voltages()
                        norm_tails.index = vtest

                        # create dataframe for boltzmann parameters
                        # columns = parameter names, index = filename
                        ac_pars = pd.DataFrame(
                            data=ac_pars, index=[filenames[i]])

                        self.ac_norm_data.append(norm_tails)
                        self.ac_fit_params.append(ac_pars)

                    elif "_de" in pname:
                        v_, _ = ExtractTraces(df1, u, intervals, N, khz,
                                              filenames[i], ramp="x").extract()
                        df_tails = extracted_data[filenames[i]
                                                  ]["data"][0].copy()

                        v_post, df_post = ExtractTraces(df1, u+1, intervals, N,
                                                        khz, filenames[i], ramp="x").extract()
                        extracted_data[filenames[i]]["data"].append(df_post)

                        # check if all post-tail voltages are the same
                        if sum(v_post) == v_post[0]*len(v_post):

                            # check if the post-tail voltage is among the test tail voltages
                            if v_post[0] in vtest:

                                ac = activation_curve(vtest, df_tails, khz, post_tails=df_post,
                                                      post_tail_voltage=v_post[0], show_pmin=False, pdf=PDFPlots,
                                                      **self.do_activation_curves)

                                d = ac.tail_pmins()

                                if isinstance(d[0], float):
                                    df_pmins = pd.DataFrame(
                                        data={
                                            "Voltages": ac.return_test_voltages(),
                                            "FileName": filenames[i],
                                            "Tail_Pmin": d}
                                    )
                                else:
                                    tail_mins, post_mins = d
                                    df_pmins = pd.DataFrame(
                                        data={
                                            "Voltages": ac.return_test_voltages(),
                                            "FileName": filenames[i],
                                            "Tail_Pmin": tail_mins,
                                            "PostTail_Pmin": post_mins}
                                    )

                                self.tail_post_pmins.append(df_pmins)

            if self.do_inst_IV:
                if Lsub is None or any(x in pname for x in ["ramp", "env"]):
                    pass
                else:
                    # do I-V analysis
                    IV_params, Iinst_df = Lsub.IV_analysis(df_to_fit, df_protocol,
                                                           self.exp_params.at[filenames[i], "C_m (pF)"],
                                                           khz=khz, plot_results=True, pdfs=PDFPlots, output=True)

                    self.IV_params.append(IV_params)
                    self.IV_currents.append(Iinst_df)

            # close the multipage pdf object
            if self.save_AggregatedPDF:
                try:
                    PDFPlots.close()
                    print("PDF object successfully closed. \n")
                except:
                    print(
                        "PDF object was not closed properly. Maybe it was closed during processing above? \n")

        # convert summarized analytics into dataframes
        if self.do_activation_curves:
            print("GV kwargs: \n", self.do_activation_curves)

            if len(self.ac_norm_data) > 0:

                self.ac_norm_data = pd.concat(
                    [d.dropna() for d in self.ac_norm_data], axis=1
                )

                self.ac_fit_params = pd.concat(self.ac_fit_params, axis=0)

                DataframeStats(self.ac_norm_data, axis=1)
                DataframeStats(self.ac_fit_params)

            if len(self.tail_post_pmins) > 0:
                # if len(self.tail_post_pmins) > 1:
                self.tail_post_pmins = pd.concat(self.tail_post_pmins, axis=0)
                self.tail_post_pmins.set_index(
                    "Voltages", drop=True, inplace=True)

                # convert filename column to strings
                self.tail_post_pmins["FileName"] = self.tail_post_pmins["FileName"].astype(
                    str)

        if self.do_ramp_stuff and (len(self.ramp_stats) > 0):
            self.ramp_stats = pd.concat(self.ramp_stats, axis=0)
            print(self.ramp_stats)

            DataframeStats(self.ramp_stats, groupby="Rates")

        if self.do_exp_kinetics:
            print("Exp. kinetics kwargs: \n", self.do_exp_kinetics)

            # standard activation/deactivation fitting
            if len(self.exp_fit_params) > 0 and len(self.exp_fit_delay) > 0:
                self.exp_fit_params = pd.concat(self.exp_fit_params, axis=0)
                self.exp_fit_delay = pd.DataFrame.from_dict(self.exp_fit_delay)

            # envelope
            else:
                if len(self.exp_fit_params) > 1:
                    self.exp_fit_params = pd.concat(
                        self.exp_fit_params, axis=1)
                print(self.exp_fit_params)

        if self.do_inst_IV:
            if len(self.IV_params) > 0:
                self.IV_params = pd.concat(self.IV_params, axis=1)

            if len(self.IV_currents) > 0:
                self.IV_currents = pd.concat(self.IV_currents, axis=1)

        if self.normalize:
            """ 
            Probably do normalization/reduction here, so we can access summary GV, boltz params, tail_pmins, etc.
            Data structures
                1. Step intervals = self.epoch_startends = {filename : list}
                2. Leak-subtracted data = data_files_Lsub = {filename : dataframe}
                3. Normalized tail peak values (for GV curves) = self.ac_norm_data = voltage index x filename columns  
                4. Boltzmann fit parameters = self.ac_fit_params = filename index x Vh/s/c columns
                5. Activation name = self.paired_files = {parent filename : [[subsequent filenames], [activation filename]]}
                    5a. Check if last element of value is a list or str. If str, then no activation filenames.
                    5b. If list, then we have an activation recording from the same cell. Index the first element of activation filenames and use for input in call to `normalization`
                6. Tail Pmins = self.tail_post_pmins = FileName index x columns `Tail_Pmin`, and occasionally `PostTail_Pmin`
            """
            print("Normalization kwargs: \n", self.normalize)

            # reset pyplot style
            plt.style.use('default')
            RC_defaults()

            # check that activation curve info is available
            if len(self.do_activation_curves.keys()) < 1:
                raise Exception(
                    "Normalization requires activation curve info. Add appropraite keys to `self.do_activation_curves` in call to `process`, e.g. {'show' : True} ")

            for i, df in enumerate(data_files.values()):
                try:
                    pname = ephys_info.loc[ephys_info['Files']
                                           == filenames[i], "Protocol"].iat[0]
                except:
                    raise Exception(
                        "Couldn't find protocol name for %s" % filenames[i])

                if "ramp_dt" in pname:
                    print("Skipping normalization of {f}, with protocol {p}, which is currently not normalized.".format(
                        f=filenames[i], p=pname))
                    continue
                elif "ramp_de" in pname:
                    extract_option = "de"
                else:
                    extract_option = "x"

                # open PDF object if available
                if self.save_AggregatedPDF:
                    # path to PDF of previously generated plots
                    PDFPlots = r"{savedir}/PDFs/{fname}-{pname}.pdf".format(
                        savedir=self.save_path, fname=filenames[i], pname=pname)
                else:
                    PDFPlots = None

                print("Normalizing... %s \n     Protocol = %s " %
                      (filenames[i], pname))

                khz = int(self.dataRates[filenames[i]])
                N = int(df.shape[1]/2)

                # get leak subtracted dataframe
                if filenames[i] in data_files_Lsub.keys():
                    df_ = data_files_Lsub[filenames[i]]
                else:
                    df_ = df

                # step intervals for sweeps in current file
                intervals = self.epoch_startends[filenames[i]]

                # x1, x2 are either voltages or ramp durations
                # ind = 2, 3 -> upper bound + 1 for indexing epochs
                #   i.e. ind=2 implies test interval is bounded by epochs 0 and 1
                #   e.g. intervals[0][0:2]
                #   e.g. ind=3 -> intervals[0][1:3]
                # activation
                x1, _ = ExtractTraces(df_, 2, intervals, N, khz, filenames[i],
                                      ramp=extract_option).extract()
                # deactivation
                x2, _ = ExtractTraces(df_, 3, intervals, N, khz, filenames[i],
                                      ramp=extract_option).extract()

                if "de" in pname:
                    # get upper index that bounds test pulse in protocol `pname`
                    u = get_upper_index(df, pname)
                    if u is None:
                        raise Exception(
                            "No upper index found for file < %s >" % filenames[i])

                    # enable manual_cap_offset = 'top' when large caps present
                    if filenames[i] in self.manual_cap_offset.keys() and \
                            self.manual_cap_offset[filenames[i]] == "top_act":
                        manual_cap = {filenames[i]: "top"}
                    else:
                        manual_cap = {}

                    _, df_act = ExtractTraces(df_, u-1, intervals, N, khz,
                                              filenames[i], ramp="x", show=True,
                                              manual_cap_offset=manual_cap
                                              ).extract()

                    df_de = extracted_data[filenames[i]]["data"][0]

                else:
                    df_act, df_de = extracted_data[filenames[i]]["data"]

                if "de" in pname:
                    if "env" in pname:
                        print(
                            "Skipping normalization for `env` protocol for file < %s >" % filenames[i])
                        continue

                    # rename columns of (leak-subtracted) data with test voltages
                    df_act.columns = x2
                    df_de.columns = x2
                    # since voltages may not be sorted, so we sort `x2`, then re-arrange the dataframes
                    x2.sort()
                    # sort columns of df_act and df_de by voltages `x2`
                    df_act = df_act.loc[:, x2]
                    df_de = df_de.loc[:, x2]

                    # remove traces if specified
                    if filenames[i] in self.remove_after_normalize.keys():
                        df_act, df_de, x2 = remove_traces([df_act, df_de], khz, x2,
                                                          self.remove_after_normalize[filenames[i]], show=False)

                    # prepulse for deactivation
                    #   e.g. intervals[1][0] = epoch (start of hyperpolarization) for 1st trace
                    #        N+1 = voltage command for 2nd trace
                    # key in `intervals` for first sweep
                    k_ = list(intervals.keys())[0]
                    v = int(df_.iat[intervals[k_][0]+100*khz, N])
                    print("Prepulse voltage for deactivation", v)

                    # look for filename of activation protocol from the same cell, if available
                    paired = None
                    for key, val in self.paired_files.items():
                        if len(val) > 1 and filenames[i] in val and isinstance(val[1], list):
                            paired = val[1][0]
                            break

                    # `ramp_de` protocols
                    if "ramp_de" in pname:
                        nrm = normalize_for_fitting(pname, filenames[i], [df_act, df_de],
                                                    khz, volts=x2, prepulse=v, GV=self.ac_norm_data, boltz_params=self.ac_fit_params, paired=paired)
                        dflis = nrm.do_norm(pdf_dir=PDFPlots, **self.normalize)

                    # standard deactivation
                    else:
                        # get Pmins corresponding to given file as mean of tail pmin and post peak
                        if type(self.tail_post_pmins) == list:
                            print("No pmins were recorded.")
                            continue

                        elif not self.tail_post_pmins["FileName"].isin([filenames[i]]).any():
                            print("No pmins available for < %s >." %
                                  filenames[i])
                            continue

                        # select tail_pmins and post_pmins
                        Pmins = self.tail_post_pmins.copy()
                        # select file
                        Pmins = Pmins.loc[Pmins["FileName"] == filenames[i],
                                          ["Tail_Pmin", "PostTail_Pmin"]]

                        # Drop empty rows
                        Pmins.dropna(how="all", axis=0, inplace=True)

                        # plot Pmins
                        fig, ax = plt.subplots(
                            figsize=(8, 5), constrained_layout=True)
                        ax.set_title("Pmins: %s" % filenames[i])
                        ax.set_xlabel("Voltage (mV)")
                        ax.set_ylabel("Pmin")

                        for j in range(2):
                            ax.plot(Pmins.iloc[:, j], marker='o',
                                    ls='none', label=Pmins.columns[j])

                        # take the mean if difference is < 0.2, or minimum otherwise
                        if Pmins.shape[1] > 1:
                            for j in range(Pmins.shape[0]):
                                if abs(Pmins.iat[j, 0] - Pmins.iat[j, 1]) > 0.2:
                                    Pmins.iat[j, 0] = Pmins.iloc[j, :].min()
                                else:
                                    Pmins.iat[j, 0] = Pmins.iloc[j, :].mean()

                            # keep filtered values
                            Pmins = Pmins.iloc[:, 0]

                        # plot mean/min Pmin
                        ax.plot(Pmins, ls='--', lw=2,
                                alpha=0.7, label="Mean or Min")

                        # sort index (test voltages)
                        Pmins.sort_index(inplace=True)

                        # fit Pmin with 2exp (actually 1exp)
                        """
                        popt, _ = curve_fit(exp2, Pmins.index.values, Pmins.values,
                                            p0=[Pmins.iloc[0], 30, Pmins.iloc[-1]],
                                            bounds=([0, 5, 0], 
                                                    [0.5, 100, Pmins.iloc[0]])
                                        )
                        fit_v = range(-120, 55, 5)
                        fit_y = [exp2(v, *popt) for v in fit_v]
                        
                        ax.plot(fit_v, fit_y, c='b', alpha=0.5, label=popt)
                        """

                        # check that test voltages and voltages of tail pmins are the same
                        if x2 != Pmins.index.values.tolist():
                            print("Test voltages: ", x2)
                            print("Tail pmins: ", Pmins)
                            print(
                                "\n Test voltages do not fully match voltages of tail pmins.")

                            # select test voltages that are present
                            Pmins = Pmins.loc[x2]

                        # normalize
                        nrm = normalize_for_fitting(pname, filenames[i], [df_act, df_de],
                                                    khz, volts=x2, prepulse=v, pmins=Pmins, PminPlot=[
                                                        fig, ax],
                                                    GV=self.ac_norm_data, boltz_params=self.ac_fit_params,
                                                    paired=paired)

                        dflis = nrm.do_norm(pdf_dir=PDFPlots, **self.normalize)

                else:
                    # remove traces if specified
                    df_act.columns = x1
                    df_de.columns = x1
                    if filenames[i] in self.remove_after_normalize.keys():
                        df_act, df_de, x1 = remove_traces([df_act, df_de], khz, x1,
                                                          self.remove_after_normalize[filenames[i]], show=False)

                    # deactivating voltage for activation
                    # time = 2nd epoch (end of hpol, start of depol) of 1st trace
                    # voltage = command of 2nd trace
                    v = list(intervals.keys())[0]
                    v = intervals[v][1] + 100*khz
                    v = int(df.iat[v, N])

                    print("Deactivating voltage for activation = ", v)

                    nrm = normalize_for_fitting(pname, filenames[i], [df_act, df_de],
                                                khz, volts=x1, postpulse=v,
                                                GV=self.ac_norm_data, boltz_params=self.ac_fit_params)

                    dflis = nrm.do_norm(pdf_dir=PDFPlots, **self.normalize)

                # save normalized dataframes as CSVs
                # destination
                dest = r"{m}/output/Processing/Processed_Time_Courses/{f}".format(
                    m=self.main_dir, f=filenames[i])

                if "ramp_de" in pname:
                    dest += "_rampde_"
                elif "_de" in pname:
                    dest += "_de_"
                elif "_act" in pname:
                    dest += "_act_"

                if save_normalized == True:

                    dflis[0].to_csv(dest + "act_norm.csv")
                    dflis[1].to_csv(dest + "de_norm.csv")

                    if len(dflis) > 2:
                        dflis[2].to_csv(dest + "act_norm_reduced.csv")
                        dflis[3].to_csv(dest + "de_norm_reduced.csv")

                    print(
                        "Successfully saved normalized current traces to < %s... >" % dest)

        if return_traces:
            return data_files_Lsub, extracted_data

    def go(self, idle=False, save_traces=False, save_output_csv=False, append=False):
        """
        Run processing pipeline

        idle = if true, do nothing
        append = if true, try to append to existing .csv, else write new .csv
        save_traces = if True, saves normalized, extracted, and leak-subtracted current traces 
        save_output_csv = if true, saves output dataframes to csv files 
        """

        if not idle:
            self.etc(save_normalized=save_traces,
                     save_extracted=save_traces, save_leaksub=save_traces)

            if save_output_csv:
                # path to save to
                path = self.save_path + "/CSV_output/"

                # specify filenames and data to save
                labels = []
                output = []

                print("`save_output_csv` is enabled. output will be saved at {p} with file prefix {pf}".format(
                    p=path, pf=self.output_prefix))

                if self.do_activation_curves:
                    labels.extend(
                        ["act_norm.csv", "boltz_params.csv", "tail_post_pmins.csv"])
                    output.extend(
                        [self.ac_norm_data, self.ac_fit_params, self.tail_post_pmins])
                elif self.do_ramp_stuff:
                    labels.append("ramp_stats.csv")
                    output.append(self.ramp_stats)
                elif self.do_exp_kinetics:

                    if type(self.exp_fit_delay) == list:
                        labels.append("env_exp_params.csv")
                        output.append(self.exp_fit_params)
                    else:
                        labels.extend(["exp_delay.csv", "exp_params.csv"])
                        output.extend(
                            [self.exp_fit_delay, self.exp_fit_params])

                elif self.do_inst_IV:
                    labels.extend(["IV_params.csv", "IV_currents.csv"])
                    output.extend([self.IV_params, self.IV_currents])

                print(" Output that will be saved: ", labels)

                # add prefix to output filenames to specify the files and protocols used as input to the call to `process`
                if len(labels) + len(output) > 0:
                    labels = [
                        "{prefix}__{s}".format(prefix=self.output_prefix, s=label) for label in labels
                    ]
                # print(labels)

                # avoid ellipses when saving long arrays (e.g. lists of parameters) in .csv
                np.set_printoptions(threshold=1e9)

                for (i, o) in enumerate(output):
                    if len(o) < 1:
                        print(
                            " %s is empty. Skipping creation of .csv file." % labels[i])
                        continue
                    else:
                        if append:
                            try:
                                o.to_csv(path + labels[i],
                                         mode='a', header=False)
                                print(" %s appended successfully." % labels[i])
                            except:
                                print(
                                    " Could not append %s. Creating new file." % labels[i])
                                o.to_csv(path + labels[i])
                                print(" %s created successfully." % labels[i])
                        else:
                            try:
                                o.to_csv(path + labels[i])
                                print(" %s created successfully." % labels[i])
                            except:
                                print(o)
                                print(" Could not write %s. Skipping." %
                                      labels[i])
                                continue
