# personal libraries, in the same directory 
from LeakSubtraction import leak_subtract 
from EphysInfoFilter import EphysInfoFiltering 
from ActivationCurves import activation_curve

from collections import OrderedDict
import numpy as np 
import pandas as pd 
import re 
import math 
import glob 
import os 
import pyabf 

import matplotlib.pyplot as plt 
from matplotlib import rcParams 
import matplotlib.patheffects as pe 
from matplotlib.backends.backend_pdf import PdfPages 

rcParams['font.size'] = 12 
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Verdana'
rcParams['font.weight'] = 'normal'
rcParams['axes.linewidth'] = 2
rcParams['axes.labelweight'] = 'bold' 

# cmap = plt.cm.get_cmap()
cmap = plt.cm.get_cmap("gist_rainbow")
plt.style.use("dark_background")
# plt.style.use("seaborn-colorblind")

from scipy.optimize import curve_fit 
from scipy.optimize import minimize 
from scipy.integrate import simps, romb, trapz  
from scipy.stats import pearsonr, sem 
from scipy.signal import bessel, filtfilt

from PyPDF2 import PdfFileReader, PdfFileMerger

def apply_bessel(df, khz, desired_freq=0.1, show=False):
    """
    Create and apply 4th order Bessel filter. Sampling parameters can be viewed in pClamp. 
    
    df = dataframe  
    khz = sample frequency -> 'angular frequency'  
    desired_freq = target frequency after filtering, in khz 
    """
    # normalize target frequency 
    # desired_freq = desired_freq / (khz/2)
    desired_freq *= 1/ (khz / 2)
    
    # norm = mag -> normalize so that gain magnitude is -3dB at khz (angular frequency)
    # b, a = numerator and denominator (respectively) polynomials of the filter
    # return digital filter
    # fs = sample frequency of data 
    b, a = bessel(4, desired_freq, btype="lowpass", 
                analog=False, output="ba", 
                # norm="mag", 
                fs=khz)
    output = filtfilt(b, a, df)
    
    if show:
        x = np.arange(df.shape[0]) / khz 
        plt.plot(x, df, alpha=0.5, label="Data")
        plt.plot(x, output, lw=2, label="Filtered")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
    
    return output 

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
                            print(dflis[ k[1][j] ].iloc[:, i].shape[0])
                            
                            dflis[ k[1][j] ].iloc[:k[1][j+1], i] = np.nan 
                            
                        # truncate end 
                        else:
                            dflis[ k[1][j] ].iloc[k[1][j+1]:, i] = np.nan 
            
            # apply truncation to both dataframes 
            else:
                for j in range(2):
                    if k[1] > 0:
                        dflis[j].iloc[:k[1],i] = np.nan
                    else:
                        dflis[j].iloc[k[1][j+1]:, i] = np.nan 
                
    # voltages to drop 
    for k in to_drop:
        if isinstance(k, int):
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
    
    f, ax = plt.subplots(figsize=(9,4))
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
    ax.set_ylabel(r"$\mathbf{\frac{V_{out}}{V_{in}}}$", fontsize=20, rotation=0, labelpad=22)
    ax.set_title(fname)
    
    ax.set_xscale('log')
    ax.grid(b=True, which='both', axis='both', alpha=0.3)
    
    plt.tight_layout()
    
    if pdfs is not None:
        pdfs.savefig(Figure=f, bbox_inches='tight')
    
    if show:
        plt.show()
        
    plt.close()
    
def file_specific_transform(fname, times):
    """
    Apply file-specific transformation to times, as necessary.  
    `fname` = filename  
    `times` = bounding intervals for leak ramp  
    """
    to_transform = ["20903005"]
    
    if fname in to_transform:
        if fname == "20903005":
            # after the first trace, multiply all epochs by 2 
            if isinstance(times, list):
                t0, t1 = times 
                return [t0, t1, 2*t0, 2*t1, 2*t0, 2*t1]
            elif isinstance(times, dict):
                if fname in times.keys():
                    d = times[fname]
                    for i in range(1, 3):
                        d[i] = [2*x for x in d[i]]
                        
                    times[fname] = d 
                return  times 
            elif isinstance(times, tuple):
                # tuple = abf_sum, csv_sum -> force abf_sum = csv_sum 
                return times[0] 
        
    else:
        if isinstance(times, tuple):
            # abf_sum, csv_sum -> keep csv_sum 
            return times[1]
        else:
            # don't change anything 
            # print(" `file_specific_transform` was called, but `fname` is None. Please provide a filename.")
            return times  
 
def make_figure(df, intervals, khz=2, save_path=None, pdfs=None, show=False,
                current_yaxis_label="Leak-subtracted \n Current (pA)",
                seconds=True,
                format="leaksub"):
    """
    make figures for ephys traces   
        df = data 
        intervals = epochs for each trace in the file 
        khz = sample frequency
        
        save_path = if not NOne -> where the figures will be saved
        pdfs = pdf multipage object if not None; else will save png/svg 
        seconds = True -> uses seconds in time axis 
        format = if "leaksub" -> adds '_leaksub' as a suffix to saved files 
    """    
    N = int(df.shape[1]/2)
    
    fig = plt.figure(figsize=(12,5.5))
    gs = fig.add_gridspec(7,1)
    
    #convert ms to seconds 
    if seconds:
        df.index *= 1/1000 
    
    ax1 = plt.subplot(gs[:5]) #current 
    ax2 = plt.subplot(gs[5:]) #voltage protocol
    # plt.subplots_adjust(hspace=0.8) 
    
    # set up visuals, e.g. tick locations, appearance of spines 
    all_axes = [ax1, ax2] 
    for j, a in enumerate(all_axes):
        if j == 1: #voltage 
            # a.tick_params(axis='y', which='both', left=False)
            # a.tick_params(axis='x', length=6, width=3, labelsize=12)
            # a.spines['left'].set_visible(False) 
            
            ybot, ytop = [int(x) for x in a.get_ylim()] 
            ypos = range(ybot-20, ytop+20, 50)
            a.set_yticks(ypos, ypos)
            a.locator_params(axis='y', nbins=3)
            
            if seconds:
                a.set_xlabel("Time (s)", fontsize=12, labelpad=10)
            else:
                a.set_xlabel("Time (ms)", fontsize=12, labelpad=10)
            
            # add downwards offset to bottom spines  
            a.spines['bottom'].set_position(('outward', 10))
        else:
            # a.tick_params(axis='both', length=6, width=3, labelsize=12)
            a.yaxis.set_ticks_position('left') 
            a.spines['bottom'].set_visible(False)
            a.locator_params(axis='y', nbins=4)
    
        a.tick_params(axis='both', length=6, width=3, labelsize=12)
        
        #remove top and right borders 
        a.spines['top'].set_visible(False)
        #a.spines['left'].set_visible(False)
        a.spines['right'].set_visible(False)                
        a.xaxis.set_ticks_position('bottom')
    
        #four tick marks per axis 
        a.locator_params(axis='x', nbins=6)
                
        #empty xtick labels for current 
        if j % 2 == 0:
            # a.set_xticklabels(['']*4)
            a.set_xticklabels([""])
            a.set_xticks([]) 
        
    df_i = df.iloc[:, :N] 
    df_v = df.iloc[:, N:] 
    
    # truncate last 25% if total duration is > 50s and change in current < 50 pA 
    if df_v.shape[0]/khz > 50e3:
        t = int(0.75*df_v.shape[0])
        
        i0 = df_v.iloc[t:, :].min(axis=0).values
        imax = df_v.iloc[t:, :].max(axis=0).values
        
        if (np.abs(imax - i0) < 50).all():
            df_i = df_i.iloc[:t, :]
            df_v = df_v.iloc[:t, :] 
    
    # start of test pulses; add 200ms of holding to the start
    t0 = intervals[0][0] - 200*khz 
    
    ax1.plot(df_i.iloc[t0:, 0], c='r', lw=2)
    ax2.plot(df_v.iloc[t0:, 0], c='r', lw=2)
    ax1.plot(df_i.iloc[t0:, 1:], c='w', lw=1.5)
    ax2.plot(df_v.iloc[t0:, 1:], c='w', lw=1.5)
    
    plt.tight_layout(rect=[0.05, 0.02, 1, 1])
    
    ax1.set_ylabel(current_yaxis_label, rotation=90, labelpad=12, fontsize=14)
    ax2.set_ylabel("Voltage (mV)", rotation=90, labelpad=12, fontsize=14)
    
    if pdfs is not None:
        pdfs.savefig(bbox_inches='tight')
    else:
        if save_path is not None:
            if format == "leaksub":
                if pdfs is None:
                    plt.savefig(save_path + r"%s_leaksub.png" % fname, bbox_inches='tight')
                    plt.savefig(save_path + r"%s_leaksub.svg" % fname, bbox_inches='tight')
            else:
                if pdfs is None:
                    plt.savefig(save_path + r"%s.png" % format, bbox_inches='tight')
                    plt.savefig(save_path + r"%s.svg" % format, bbox_inches='tight')
    
    if show:
        plt.show()
        
    plt.close('all')       
    # exit()
        
def estimate_Cm(startend, df, khz=1, centerFrac=0.3):
    """
    Use SW Harden's method of estimating Cm from voltage ramps.
    
    startend = start adn end indices of voltage ramp for Cm estimation
    df = dataframe for recording
    khz = sample frequency of data, in khz 
    centerFrac is the fractional time span to draw data from in the center of each ramp.
    """
    ramp = df.iloc[startend[0]:startend[1],:] 
    
    #number of traces 
    N = int(ramp.shape[1]/2)
    
    #check ramp dissection 
    # plt.plot(ramp.iloc[:,N:])
    # ### plt.show()
    # exit()
        
    #find midpoint of ramp 
    thalf = int(ramp.shape[0]/2)
    
    cm_vals = [] 
    for i in range(N):
        # split the ramp current into separate arms 
        ramp1 = ramp.iloc[:thalf,i].values[::-1]
        ramp2 = ramp.iloc[thalf:,i].values 

        # average of both arms 
        ramp_avg = np.mean([ramp1, ramp2], axis=0) 
        
        # figure out the middle of the data we wish to sample from
        centerPoint = int(len(ramp1))/2
        centerLeft = int(centerPoint*(1-centerFrac))
        centerRight = int(centerPoint*(1+centerFrac))
        
        # slope of the ramp in ms 
        ramp_duration = ramp.shape[0]/(2*khz) # duration 

        # dV = np.ptp(ramp.iloc[:,N+i].values, axis=0)
        dV = ramp.iat[thalf+1,N+i] - ramp.iat[0,N+i]
        
        ramp_slope_ms = dV / (ramp_duration)
        
        # average slope deviation (distance from the mean)
        d_ramp = (ramp1 - ramp2)/2
        
        # deviation for the center
        d_ramp_center = d_ramp[centerLeft:centerRight]
        deviation = np.mean(d_ramp_center)
        
        cm_vals.append( np.abs(deviation / ramp_slope_ms) )
    
    return cm_vals 

def estimate_MT(startend, df, khz=1, pdf=None):
    """
    Using the method described by SW Harden and pClamp, estimate Ra, Rm, and Cm by fitting a single exponential to the capacitive transient of a membrane test step.
    """
    N = int(df.shape[1]/2)
    
    # define a single exponential
    def func(t, dI, tau, I_ss): 
        return dI*np.exp(-t/tau) + I_ss 
    
    params_SWH = [] #SW Harden methods
    params_MDC = [] #MDC methods 
    MDC_Rm_error = [] # check correspondence between tau and Rm using tau ~ Rm*Cm 
    
    def do_estimation(k):
        # isolate membrane test 
        # to prevent SettingWithCopyWarning: https://stackoverflow.com/a/58829423
        df_MT = df.iloc[startend[k]:startend[k+1],:].copy()
        df_MT.index -= df_MT.index[0] 
        time = df_MT.index.values.tolist()
                        
        #fifth of the interval used for I_ss calculations
        fifth = int(df_MT.shape[0]/5) 
        
        #invert transient if mean of first 20ms is less than last 20ms 
        mu = df_MT.iloc[:,:N].mean(axis=0).values 
        if mu[0] < mu[-1]:
            df_MT.iloc[:,:N] *= -1 
        
        # check isolation
        # plt.plot(df_MT.iloc[:,:N])
        # ### plt.show()

        for i in range(N):
            I_t = df_MT.iloc[:,i].values
            
            #find index of capacitance peak 
            peak_idx = int(np.argmax(I_t))
            
            #truncate time and current accordingly 
            I_t = I_t.tolist()[peak_idx:]
            times = [t - time[peak_idx] for t in time[peak_idx:]] 
            
            I_prev = df.iloc[startend[k]-fifth-10*khz:startend[k]-10*khz, i].mean() 
            dV = df.iat[startend[k]+2*fifth, N+i] - df.iat[startend[k]-2*fifth, N+i]
            
            dI = max(I_t) - min(I_t) 
            popt, pcov = curve_fit(func, 
                                times, I_t,
                                p0 = [dI, 10, np.mean(I_t[-fifth:])],
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
            
            #test solution
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
            R_m = abs( (dV*1e-3 - R_a*I_dss*1e-6)/(I_dss*1e-12) )*1e-6
            
            # From SW Harden:
            # When we raise the cell's voltage (Vm) by delivering current through the pipette (Ra), some current escapes through Rm. From the cell's perspective when we charge it though, Ra and Rm are in parallel.
            # C_m = tau / R, 1/R = 1/R_a + 1/R_m 
            C_m = abs( tau / (1/ (1/R_a) + (1/R_m)) ) *1e3
            
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
            
            ax[i,j].plot(sweeps, [x[h] for x in params_SWH], 
                        marker='o', c='r',
                        alpha=0.7, label="SWH")
            if h > 0:
                ax[i,j].plot(sweeps, [x[h-1] for x in params_MDC], 
                            marker='x', markersize=8, c='white',
                            alpha=0.7, label="MDC")
                        
            ax[i,j].legend() 
            ax[i,j].set_title(labels[h])
    
    for i in range(2):
        ax[1,i].set_xlabel("Sweep #")
    
    f.suptitle("Estimation of Membrane Test Parameters")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if pdf is not None:
        pdf.savefig()
    
    ### plt.show()
    plt.close()
        
def find_cap_dt(df, N, khz=2, window=15):
    """
    `df` = dataframe of extracted test pulses 
    `N` = number of sweeps 
    
    Attempt to find duration of capacitive spikes from extracted test pulses.
        Loop over traces, 
        1. Use range (start - end current) to determine polarity, i.e. activation vs deactivation
        2. Find timepoint after which current change between timepoints is on the order of the mean of the first 500ms of the pulse
    
    If the range of durations (max - min) is greater than 5ms, returns the median duration.
    Else, returns the maximum duration. 
    """
    khz = int(khz) 
    dfa = df.iloc[:,:N].abs().values 
    # dI = dfa.iloc[50*khz,:] - dfa.iloc[-50*khz,:] 
    
    # find rate of change between `window` and `window + 500ms`
    dT = 500 
    # if length of pulse is less than 500ms, iteratively reduce by 25ms 
    while dfa.shape[0] < (dT + window)*khz:
        dT -= 25 
    
    df_dt = (dfa[window*khz:(dT+window)*khz, :] - dfa[:dT*khz, :]) / window*khz     
    mu = np.mean(np.abs(df_dt), axis=0) 

    caps = [] 
    for i in range(N):
        # polarity = dI.iloc[i] < 0 
        
        for j in range(df_dt.shape[0]):
            # J = df_dt[j, i]

            if  abs(df_dt[j, i]) > mu[i]: 
                continue 
            else:
                caps.append(j+1) 
                break 
    
        # plt.plot(df.iloc[caps[i]:, i])

    if max(caps) - min(caps) > 5*khz:
        return np.median(caps)
    else:
        return max(caps) 
                
def extract_traces(df, ind, intervals, ntraces, khz, 
                return_voltages=False,  
                ramp="x", env=False):
    """
    Extract pulses from a dataframe given capacitance duration and bounding interval.
    For `ramp`, `de`, and `act` protocols, returns test voltages (or ramp half-durations for ramps), extracted test pulses with current and voltage columns` 
    
    `ind` = upper index that bounds test pulse, i.e. [u-2, u] 
    `intervals` = dict {sweeps : {epochs : durations}} 
    `ntraces` = number of traces 
    `khz` = sampling frequency in khz 
    `return_voltages` = True -> returns corresponding voltage protocols  
    
    ### `ramp` = "x", "dt", or "de" 
    * "x" -> Nothing  
    * "dt" -> Equal-duration ramps. Returns half-durations instead of voltages. `ind` is the end of second ramp. Extracts both first and second ramps.  
    * "de" -> Deactivating ramps (fixed prepulse followed by varying-duration depolarizing ramp.) `ind` is end of ramp. Returns duration of and Extracts depolarizing ramps.   
        
    ### `env` = bool; if `True`, treats file as an envelope of tails protocol:  
    For `env` protocols, returns `[act_volt, tail_volt], test_times, df_to_fit` when return_voltages=False.  
    * `act_volt` and `tail_volt` are voltages for activation and deactivation, respectively, and taken from the first sweep.  
    * `test_times` contains four epochs between the start of the first activation and end of the second activation. `df_to_fit` takes current data from this entire range.  
    """    
    df_to_fit = [] 
    
    # envelope 
    if env:
        act_volt = -120 # activation voltage 
        tail_volt = 0   # voltage of tail pulse 
        test_times = [] # list of 4 epochs: 1st hpol, start of tail, start and end of 2nd hpol
    else:
        # non-ramp -> 2-step protocols where steps vary in voltage
        if ramp == "x":
            test_voltages = [] 
        # ramp -> 2-step protocols with varying duration which we track as half-duration in `tmids`
        else:
            tmids = [] 
        
    protocol = [] 
    for j, k in enumerate(intervals.keys()):
        if j >= ntraces:
            break 
        
        if env:
            # 1st activation, start of tail, start of 2nd activation, end of 2nd activation
            ts = intervals[k][ind-4:ind] 
            test_times.append(ts) 
            
            if j == 0:
                act_volt = df.iat[ts[0]+50, ntraces+j]
                tail_volt = df.iat[ts[1]+50, ntraces+j]
        else:
            # start and end of jth test pulse
            if ramp == "dt":
                # start of 1st ramp, middle of ramp, end of 2nd ramp 
                t0, tmid, t1 = intervals[k][ind-3:ind]    
                tmids.append(tmid-t0) 
            elif ramp == "de":
                t0, t1 = intervals[k][ind-2:ind]    
                tmids.append(t1-t0)
            else:
                t0, t1 = intervals[k][ind-2:ind] 
                test_voltages.append( int(df.iat[t1-100, ntraces+j]) ) 
                
        # isolate current and voltage protocol from test pulse 
        if env: 
            trace = df.iloc[ts[0]:ts[-1]+1, [j, ntraces+j]].dropna().reset_index(drop=True)     
        else:
            trace = df.iloc[t0:t1+1, [j, ntraces+j]].dropna().reset_index(drop=True) 
        
        df_to_fit.append(trace.iloc[:,0])
        protocol.append(trace.iloc[:,1])
    
    #concatenate traces and zero index     
    df_to_fit = pd.concat(df_to_fit, axis=1)
    protocol = pd.concat(protocol, axis=1)
    df_to_fit.index *= 1/khz 
    protocol.index *= 1/khz 

    #find duration of capacitive spikes 
    print(" size of extracted dataframe \n", df_to_fit.shape)
    c = int( find_cap_dt(df_to_fit, ntraces, khz) )
    df_to_fit = df_to_fit.iloc[int(c*khz):,:].reset_index(drop=True) 
    protocol = protocol.iloc[int(c*khz):,:].reset_index(drop=True) 

    # get dataframe as array 
    df_dt = np.abs(df_to_fit.iloc[:,:ntraces].values)
    
    # average rate of current change over 500ms  
    dT = 500 
    # iteratively reduce time window for rate of current change if exceeds length of prepulse
    while df_dt.shape[0] < (2+dT)*khz:
        dT -= 25 
        
    d_base = np.abs(df_dt[2*khz:(2+dT)*khz,:] - df_dt[:dT*khz]) / (2*khz) 
    mu = np.mean(d_base)
    
    w = 0 
    # print((abs(np.mean(d_base[(w*khz):(10+w)*khz]) - mu)/mu))
    while (abs(np.mean(d_base[(w*khz):(10+w)*khz]) - mu)/mu) > 0.1:
        w += 1
    
    if w > 20:
        print("During fine-tuning of cap. estimation, interval to delete was estimated at \n ` w = %d `, \n     which exceeds the pre-defined condition \n `w < 20`. \n Showing results..." % w)
        
        f, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(df_to_fit)
        ax[0].axvline(w*khz, c='white', ls='--', label="Predicted Cap. Duration")
        ax[0].legend(loc='lower left')
        
        ax[1].plot(df_to_fit.iloc[:100*khz,:], lw=2, alpha=0.5)
        ax[1].plot(df_to_fit.iloc[w*khz:100*khz,:], lw=2)
        ax[1].axvline(w*khz, c='white', ls='--', alpha=0.5)
        
        # plt.show()
        plt.close()   
        # exit()
        
    else:
        df_to_fit = df_to_fit.iloc[(w*khz):,:].reset_index(drop=True)
        df_to_fit.index *= 1/khz 
        protocol = protocol.iloc[(w*khz):,:].reset_index(drop=True)
        protocol.index *= 1/khz 

    if env:
        if return_voltages:
            return [act_volt, tail_volt], test_times, df_to_fit, protocol 
        else:
            return [act_volt, tail_volt], test_times, df_to_fit
    else:
        if ramp == "x":
            if return_voltages:
                return test_voltages, df_to_fit, protocol 
            else:
                return test_voltages, df_to_fit
        else:
            if return_voltages:
                return tmids, df_to_fit, protocol 
            else:
                return tmids, df_to_fit
    
def get_upper_index(df, pname):
    """
    Find upper index (+1, i.e. 1-indexed) that bounds the test pulse 
    `df` = current dataframe 
    `pname` = protocol name 
    
    NOTE: 
        - inverted "act...env" protocols estimate conductance with a maximally actvating current, rather than a deactivating tail current                                     
    """
    N = int(df.shape[1]/2) # number of traces 
    if "ramp_dt" in pname:
        return 3 
    elif "_act_" in pname:
        return 2 
    elif "_de_" in pname:
        # `ramp_de` and `de` are similar 2-step protocols where the 2nd is of interest
        return 3 
    elif "env" in pname:
        # end of 2nd hyperpolarization
        return 4 
    # elif "ramp_de" in pname:
    else:
        print("Upper bound for test pulse of {pname} not provided.".format(pname=pname))
        return None 
    
class analyze_ramp_dt():
    def __init__(self, protocol, df, tmid, ntraces, khz):
        """
        protocol = dataframe of voltage command 
        df = dataframe containing leak-subtracted data for a ramp_dt protocol 
        tmid = list of indices for middle of ramps 
        ntraces = number of traces 
        khz = sampling frequency in khz 
        """
        self.df = df 
        self.tmid = tmid 
        self.ntraces = int(ntraces)
        self.khz = int(khz)
        self.protocol = protocol 
        
        # plt.plot(df)
        # for t in tmid:
        #     plt.axvline(t)
        # plt.show()
        # exit()
    
    def split_ramp(self, time, df, tmid, flip=True):
        """
        Return ramp split at tmid and corresponding timepoints 
            time = np.array 
        Post-split section of ramp is automatically flipped unless `flip`=False
        """
        t1 = np.array(time[:tmid]) 
        t2 = np.array(time[tmid:df.shape[0]]) - time[tmid]  
        
        df = df.values 
        r1 = df[:tmid] 
        if flip:
            r2 = np.flip(df[tmid:])
        else:
            r2 = df[tmid:] 
        
        n = abs(len(t2) - len(t1))
        if n > 0:
            if len(t1) < len(t2):
                t2 = t2[:-n]
                r2 = r2[:-n]
            else:
                t1 = t1[n:]
                r1 = r1[n:] 
            
        # plt.plot(t1, r1)
        # plt.plot(t2, r2)
        # plt.show()
        # exit()
        
        return t1, t2, r1, r2 
    
    def find_RampMids(self, split_outputs):
        """
        Find midpoint of ramps, tmid. 
        Find time where we achieve 50\%\ current drop 

        split_output = list of outputs from self.split_ramp 
        """        
        def I_sig(t, tmid, s, c):
            return (1-c)/(1 - np.exp((t-tmid)/s)) + c 

        def find_mid(T, R):
            I0 = max(R) 
            Imid = I0 + 0.5*(min(R) - I0)
            Imid_idx = min(range(len(R)), 
                        key=lambda i: abs(R[i] - Imid))
            return Imid_idx 

        Tmids = np.zeros((len(split_outputs), 2))
        for i in range(len(split_outputs)):
            t1, t2, r1, r2 = split_outputs[i]  
            
            # print(find_mid(t1, r1))
            Tmids[i,0] = t1[find_mid(t1, r1)]
            Tmids[i,1] = t2[find_mid(t2, r2)] 
        
        return Tmids 
        
    def moving_avg(self, x, w):
        """
        Compute moving average for given window size, w, and data array `x`
        """
        return np.convolve(x, np.ones(w), 'valid') / w 
        
    def get_int(self, dI, dx=1, t1=0):
        """
        Return area between ramp arms, with respect to variable `dx`, which is the step difference between timepoints.
        dI = difference in current between ramp arms 
        
        If rombs doesn't work, use simps, which requires the sampling times of the first ramp, `t1`
        If simps doesn't work, use trapz, which is the least accurate. 
        """
        # romberg method 
        try:
            int_dI = romb(dI, dx=dx)
        # simpsons method 
        except:
            int_dI = simps(dI, x=t1)    
        
        if math.isnan(int_dI) or abs(int_dI) == np.inf:
            int_dI = trapz(dI, x=t1)
            
        return int_dI 
        
    def H(self, plot=False, pdf=None, filter=False):
        """
        Overlap (flip) ramp arms, then either subtract current directly, and integrate areas to subtract charge  
        
        `plot` = bool; whether to show visualization  
        `pdf` = multipage PDF object to save figures in  
        `filter` = bool; whether to apply a 100khz low-pass Bessel filter to data before fitting. Try viewing results with `plot = True` once before thinking about enabling this.  
        
        *Returns*: `rates`, `dA`, `vA`, `dI`, `dT`, `Tmids_linear`  
        `rates` = rate of voltage ramps 
        `dA` = difference in area under each ramp (\int I dt = Q)  
        `vA` = same as above, but over the voltage axis, i.e. \int I dv = P (W)  
        `dI` = differencei n current  
        `dT` = difference in time of (negative) peak current in depolarizing ramp  
        `Tmids_linear` = difference in time for current under the ramp protocol to develop by 50% 
        """
        time = self.df.index.values 
        time -= time[0] 
        dt = time[1] 

        # create plots 
        if plot:
            fig, ax = plt.subplots(2, 3, figsize=(12, 7)) 
                        
            ax[0,0].set_title("Current (pA)")
            ax[1,0].set_title(r"Difference in Current (pA)")
            ax[1,0].set_xlabel("Time (s)", labelpad=12)
            
            ax[0,1].set_title(r"Total $Q$ in Difference Current (nC)")
            ax[1,1].set_title("Delay in Peak Current (ms)")
            ax[1,1].set_xlabel("Ramp Slope (mV/s)", labelpad=12)
            
            ax[0,2].set_title("Ramp Midpoints (s)")
            ax[1,2].set_title("Difference in Ramp Midpoints (s)")
            ax[1,2].set_xlabel("Ramp Slope (mV/s)", labelpad=12)
            
        dA = []     # difference in area under each ramp (\int I dt = Q)
        vA = []     # same as above, but over the voltage axis, i.e. \int I dv = P (W) 
        dI = []     # differencei n current 
        dT = []     # difference in time of (negative) peak current in depolarizing ramp 
        rates = []  # rate of change in voltage over time, in mV/s
        
        split_outputs = [] 
        for i in range(self.ntraces):
            y = self.df.iloc[:,i].dropna()
            
            # apply bessel filter to ith trace 
            if filter:
                y = pd.Series(apply_bessel(y, self.khz))
            
            # get times and current values for each arm of the ramp 
            t1, t2, r1, r2 = self.split_ramp(time, y, self.tmid[i]) 
            split_outputs.append([t1, t2, r1, r2])
            
            # get delay between minima (peak current) in either ramp
            # dT.append( t2[-1] - np.argmin(r2) )
            # take moving average over 5ms 
            dT_i = np.array( [self.moving_avg(r, 5*self.khz) for r in [r1, r2]] )
            dT_i = np.argmin(dT_i, axis=1) + 5*self.khz 
            # print(dT_i.shape) -> (2, )
            dT.append( dT_i[0] - dT_i[1] )
            
            # compute difference in current between ramp arms 
            dI.append(r1 - r2)

            # difference in area between arms of ramp, with respect to time 
            Q = self.get_int(dI[i], dt, t1)
            # area can't be negative, so if it is, redo after flipping the sign of dI\
            if Q < 0:
                Q = self.get_int(dI[i] * -1, dt, t1)
            dA.append(Q)
            
            # difference in area ... with respect to voltage 
            v = self.protocol.iloc[:,i].dropna().values 
            dv = v[1] - v[0] 
            W = self.get_int( dI[i], dv, v[:len(dI[i])] ) 
            
            if W < 0:
                W = self.get_int( dI[i] * -1, dv, v[:len(dI[i])] ) 
                
            vA.append(W)
            
            if plot:
                clr = cmap((i+1)/self.ntraces) 
                
                # convert to seconds 
                t1 *= 1e-3 
                t2 *= 1e-3 
                
                ax[0,0].plot(t1, r1, c=clr, lw=1)
                ax[0,0].plot(t2, r2, c=clr, lw=1, alpha=0.5)
                ax[1,0].plot(t1, dI[i], c=clr, lw=1)
                
                # mV/s 
                rates.append( 1000 * v[self.tmid[i]]/self.tmid[i] )
                
        # midpoints of ramps 
        Tmids_linear = self.find_RampMids(split_outputs=split_outputs)
        if plot:
            # ax3.plot(self.tmid, dI, marker='o', markersize=5, lw=1) 
            # tmid_secs = [t/1000 for t in self.tmid]
            
            # integrated charge
            ax[0,1].plot(rates, 
                    [Q/1e6 for Q in dA], 
                    marker='o', markersize=5, lw=1)

            # difference in delay to peak current 
            ax[1,1].plot(rates, dT, marker='o', markersize=5, lw=1)
            
            # width of ramp midpoints between arms 
            ax[0,2].plot(rates, Tmids_linear, marker='o', markersize=5, lw=1, label="")
            ax[0,2].legend(["Hpol.", "Depol."], loc='upper left', fontsize=10)
            
            # difference between the above between arms 
            Tmids_linear = Tmids_linear[:,0] - Tmids_linear[:,1]
            ax[1,2].plot(rates, Tmids_linear, marker='o',
                        markersize=5, lw=1)
            
            # xtick labels for summary stats w.r.t. time 
            for i in range(2):
                for j in range(1, 3):
                    ax[i,j].locator_params(axis='y', nbins=4)
                    ax[i,j].locator_params(axis='x', nbins=5)
                    
                    # ax[i,j].set_xticks(rates)
                    # ax[i,j].set_xticklabels(["%.1f" % r for r in rates]) 
                
                # show at least 4 labels on the current vs. time plots  
                ax[i, 0].locator_params(axis='x', nbins=5)
                                                                        
            fig.suptitle("Hysteresis Summary")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            if pdf is not None:
                pdf.savefig(Figure=fig) 
            plt.show()
            plt.close()
                        
            # current - voltage plots 
            fig_H, ax_H = plt.subplots(1, 2, figsize=(11, 4))
            ax_H[0].set_xlabel("Voltage (mV)", labelpad=12)
            ax_H[1].set_xlabel("Rate (mV/s)", labelpad=12)
            ax_H[0].set_ylabel("Current (pA)", labelpad=12)
            ax_H[1].set_ylabel("Power (pW)", labelpad=12)
            
            # plot voltage vs current 
            for i in range(self.ntraces):
                clr = cmap((i+1)/self.ntraces) 
                
                # current values for ith trace 
                y = self.df.iloc[:,i].dropna().values  
                
                v = self.protocol.iloc[:,i].dropna().values 
                ax_H[0].plot(v, y, c=clr, lw=1, alpha=0.8, label="%.1f" % rates[i] )
                
            # divide watts by 1e6 = femtoW -> pW 
            ax_H[1].plot(rates, [W/1e3 for W in vA], 
                        marker='o', markersize=5, lw=1, label="")
            ax_H[1].locator_params(axis='y', nbins=5)
            # ax_H[1].locator_params(axis='y', nbins=5)
            
            ax_H[0].legend(loc='lower right', title="Rate (mV/s)")
            ax_H[0].locator_params(axis='y', nbins=5)
            ax_H[0].locator_params(axis='x', nbins=6)
            fig_H.tight_layout()
            
            if pdf is not None:
                pdf.savefig(Figure=fig_H)
                # pdf.close()
            
            plt.show() 
            plt.close()
            # exit()
        
        return rates, dA, vA, dI, dT, Tmids_linear 

    def H_de(self, plot=False, pdf=None, filter=False):
        """
        Hysteresis analysis for `ramp_de` protocol, comparing varying-slope deactivating ramps with the constant-duration and constant-voltage activating prepulses.  
        
        Arguments are the same as in `analyze_ramp_dt.H(...)`
        """
        return None 
    
    def compare_dt_de(self, df_de):
        """
        Compare `ramp_dt` and `ramp_de` recordings from the same cell.  
        `df_de` = dataframe containing `ramp_de` test pulses  
        """
        return None 

class normalize_for_fitting():
    def __init__(self, pname, fname, dfs, khz, volts, 
                pmins=None, GV=None, boltz_params=None, paired=None):
        """
        pname = protocol name 
        fname = filename 
        
        df = list of dataframes containing corresponding prepulses and leak-subtracted test pulses 
        khz = sampling frequency 
        volts = list of test pulse voltages, or (half-)ramp durations for a ramp protocol 
        pmins = dataframe containing voltage index and Pmin column for given file 
        
        GV = dataframe of aggregated steady-state GV data
        boltz_params = dataframe of aggregated boltzmann fit parameters 
        
        paired = name of activation recording from the same cell            
        """
        self.pname = pname 
        self.fname = fname 
        self.df = dfs 
        self.khz = khz 
        self.volts = volts 

        if "ramp_dt" in pname: 
            pass 
        else:        
            # if protocol is not an activation protocol 
            if "de" in pname:
                # tail pmins for deactivation; for ramp protocol, we will rely on GV/boltz/simple normalization, so pmins will not be necessary 
                if pmins is not None:
                    self.pmins = pmins.loc[volts]

                # if a paired protocol is available, then take the corresponding boltzmann parameters/GV 
                if paired is None:
                    self.boltz_params = boltz_params.mean(axis=0).values.tolist()
                    self.GV = GV.mean(axis=1).dropna(how="all")
                    
                # else, take global average of GV and boltzmann parameters 
                else:
                    self.boltz_params = boltz_params.loc[paired,:].values.tolist() 
                    self.GV = GV.loc[:,paired].dropna()  
                
                # index of test pulses in `dfs`
                # self.test_idx = 1 
                
            # if activaiton, try to find GV/boltzmann parameters 
            # otherwise, take the global average of available boltzmann parameters/GV 
            else:
                if paired is None:
                    self.boltz_params = boltz_params.mean(axis=0).values.tolist() 
                
                    if GV is not None: 
                        if fname in GV.columns:
                            self.GV = GV.loc[:,fname].dropna()
                    else:
                        self.GV = GV.mean(axis=1).dropna(how="all")
                else:
                    self.boltz_params = boltz_params.loc[paired,:].values.tolist() 
                
                    
                # index of test pulses in `dfs` 
                # self.test_idx = 0
        
    def boltz(self, v, vh, s, c):
        return ((1-c)/(1 + np.exp((v-vh)/s))) + c
        
    def do_norm(self, prepulse=None, postpulse=None, reduce=0, show=False, pdf_dir=None):
        """
        prepulse = voltage of prepulse, only for deactivation (activation starts from holding, -35mV)
        postpulse = voltage of postpulse, only for activation (same as test voltages for deactivation)
        
        reduce = int; number of time points to save from normalized data 
        show = whether to show normalized plots or not 
        pdf_dir = path to PDF file to which plots will be appended using PyPDF2
        """
        if "ramp_dt" in self.pname:
            print("We can't normalize equal-duration ramp, `ramp_dt,` protocols currently because we have no estimate of Pmax at the ramp's midpoint.")
            return None 
        
        khz = int(self.khz) 
        N = int(self.df[0].shape[1]) 
        
        # f, ax = plt.subplots(1, 2, figsize=(12,7))
        # ax[0].plot(self.df[0])
        # ax[1].plot(self.df[1])
        # plt.show()
        # plt.close()
        
        # get finf values, which determine Pmax for deactivation, and steady-state Po for activation 
        if prepulse is None:         
            # get normalized conductances for each of the test voltages 
            finfs = np.zeros(N)
            
            for (i, v) in enumerate(self.volts):
                # get finf from GV if available 
                if v in self.GV.index:
                    finfs[i] = self.GV.loc[v]
                # else, estimate from boltzmann curve 
                else:
                    finfs[i] = self.boltz(v, *self.boltz_params) 
        else:
            # with specified prepulse, assume test pulses start after reaching steady state at the given voltage 
            try:
                finfs = np.ones(N) * self.GV.loc[prepulse]
            # if prepulse voltage not contained in GV data, compute with Boltzmann function 
            except: 
                finfs = np.ones(N) * self.boltz(prepulse, *self.boltz_params)

        def apply_normalization(data, test=False, p0=0.02, trunc=200, noise_bd=0.025):
            """
                data = raw dataframe for single protocol
                finfs = from above; Pmax values 
                test = whether normalizing test pulses or not 
                p0 = baseline open probability, default is 0.02, e.g. Proenza and Yellen 2006.
                trunc = upper bound in ms for location of the crest of 'hooks' aka delay
                noise_bd = makeshift estimate of upper bound for 'noise'
                    noise estimated by std of absolute first differences of normalized (1/max) data 
                    if a given trace exceeds `noise_bd`, we apply a modest savgol_filter 
            """
            # take absolute value of current traces; abs() makes min/max estimates below invariant to direction of current change
            df1 = data.copy().iloc[:,:N].abs() 
            df1.columns = self.volts 

            # # apply bessel filter to noisy traces             
            for i in range(N):
                smoothed = apply_bessel(df1.iloc[:,i].dropna().values, khz, 
                                    desired_freq = 0.1, show=False)
                df1.iloc[:len(smoothed),i] = smoothed 
            
            # truncate from crest of hook, if possible
            # limit search for crest within first 200ms 
            # squared sum of each 5ms; helps smooth/emphasize small changes in current 
            df1_5sum = df1.iloc[:trunc*khz,:].rolling(5*khz).sum().dropna().values **0.5
            # plt.plot(df1.index[:250*khz], df1.iloc[:250*khz,:], lw=3, alpha=0.2)
            # plt.show()
            # plt.close()
            
            # the crest is defined as the minimum or maximum current b/w 3-trunc ms 
            if np.max(df1_5sum[-1,:]) > np.min(df1_5sum[0,:]):
                idx = np.argmin(df1_5sum[3*khz:,:], axis=0) + 8*khz 
            else:
                idx = np.argmax(df1_5sum[3*khz:,:], axis=0) + 8*khz 

            # apply truncation, if length to truncate is 0 - trunc ms  
            for i in range(N):
                if 0 < idx[i] < trunc*khz:
                    df1.iloc[:,i] = df1.iloc[idx[i]:,i] 

            # shift any NaNs to the top of the dataframe 
            df1 = df1.apply(lambda x: pd.Series(x.dropna().values))
            # ensure index matches sample frequency 
            df1.index *= 1/khz 
            
            # plt.plot(df1.index[:250*khz], df1.iloc[:250*khz,:], ls='--', lw=2)
            # plt.scatter(self.volts, idx/khz)
            # plt.show()

            # rolling average over 5ms 
            df_avg = df1.rolling(10*khz).mean()
            # direction-invariant estimates of current amplitude         
            i0 = df_avg.min(axis=0).values             
            imax = df_avg.max(axis=0).values 
            
            if not isinstance(p0, np.ndarray):
                p0 = np.array( [p0]*N )
            else:
                p0 = np.mean(p0) 
            
            # lower bound for pmin is 0.02 
            for i in range(N):
                if p0[i] < 0.02:
                    p0[i] = 0.02
            
            # print(p0) 
            iter = 0 
            while np.any(np.abs(i0 - p0) > 0.005) or np.any(np.abs(imax - finfs) > 0.005): 
                # np.ndarray; scaling factors for each voltage
                X = (i0 - (p0*imax/finfs)) / (1 - (p0/finfs))
                df1 = finfs * ((df1 - X)/(imax - X))
        
                # i0, imax = get_i0_imax(df1.rolling(2*khz).mean().dropna()) 
                # rolling average over 5ms 
                df_avg = df1.rolling(20*khz).mean()
                
                # direction-invariant estimates of current amplitude         
                i0 = df_avg.min(axis=0).values             
                imax = df_avg.max(axis=0).values 
                
                iter += 1 
                if iter > 100:
                    break 
            
            for i in range(N):
                dfi = df1.iloc[:,i].copy()
                dfa = dfi.rolling(2*khz).mean().dropna()
                i0 = dfa.min(axis=0)
                imax = dfa.max(axis=0) 
                
                if abs(imax - i0) < 0.1 and abs(i0 - p0[i]) > 0.01:
                    iter = 0 
                    while abs(i0 - p0[i]) > 0.005 or abs(imax - finfs[i]) > 0.005:
                        X = (i0 - (p0[i]*imax/finfs[i])) / (1 - (p0[i]/finfs[i]))
                        dfi = finfs[i]*((dfi - X)/(imax - X))
                        
                        dfa = dfi.rolling(500*khz).mean().dropna()
                        i0 = dfa.min()
                        imax = dfa.max() 
                        
                        iter += 1 
                        if iter > 100:
                            break 
            
                    df1.iloc[:,i] = dfi 
            
            return df1 

        def plot_normalized(df_norm, out_pdf=None, show=show):
            """
            if reduce, df_norm = [act, de, act_reduced, de_reduced]
            out_pdf = output PDF to append to `pdf_dir` 
            """            
            f, ax = plt.subplots(1, 2, figsize=(12,7))
            ax[0].set_ylabel("Normalized Open Fraction")
            ax[0].set_xlabel("Time (ms)")
            ax[1].set_xlabel("Time (ms)")

            if prepulse is None:
                ax[0].set_title("Activation")
                
                if postpulse is not None:
                    ax[1].set_title("Deactivation")
                else:
                    ax[1].set_title("Deactivation at %d mV" % postpulse)
            else:
                ax[0].set_title("Activation at %d mV" % prepulse)
            
            # leg_ncol = number of columns for legend 
            if N < 4:
                leg_ncol = N 
            elif N < 10:
                leg_ncol = int(N/2) 
            else:
                leg_ncol = 4 
            # leg_dy = y offset for legend to place it below the subplots' x-axes 
            leg_dy = -0.1*(N/leg_ncol)
            
            if reduce > 0:
                full = df_norm[:2]      # full
                red = df_norm[2:]       # reduced 
                for i in range(2):
                    d1 = full[i]        # act
                    d2 = red[i]         # de 
                    
                    for j in range(0, d2.shape[1], 2):
                        h = int(j/2)
                        clr = cmap((h+1)/N)
                        v = self.volts[h]
                        
                        # full data 
                        ax[i].plot(d1.index.values, d1.iloc[:,h], 
                                lw=2, c=clr, label=v)
                        # reduced 
                        ax[i].plot(d2.iloc[:,j], d2.iloc[:,j+1], 
                                marker='o', markersize=2, ls='none', 
                                c=clr, label=None)
                    
                    ax[i].legend(loc='upper center', bbox_to_anchor=[0.5, leg_dy], ncol=leg_ncol)
            else:
                for i in range(2):
                    d = df_norm[i]
                    
                    for j in range(N):
                        clr = cmap((j+1)/N)
                        v = self.volts[j]
                        
                        ax[i].plot(d.index.values, d.iloc[:,j], 
                                lw=2, c=clr, label=v)
                    
                    ax[i].legend(loc='upper center', bbox_to_anchor=[0.5, leg_dy], ncol=leg_ncol)
            
            f.suptitle(r"%s / %s" % (self.fname, self.pname))
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            if out_pdf is not None:
                out_pdf.savefig(Figure=f, bbox_inches="tight")
                out_pdf.close()
                
            if show:
                plt.show()
                
            plt.close()
            
        def apply_reduction(df_norm):
            df_merge = [] 
            colnames = [] 
            
            # linearly decimate data, keeping every reduced-th 
            for i in range(N):
                y = df_norm.iloc[:,i].dropna()

                # determine linear spacing between timepoints
                k = int(len(y)/reduce)
                if k < 1:
                    pass 
                else:
                    y = y.iloc[::k].dropna()
                    
                df_merge.append(y.index.to_series())
                df_merge.append(y) 
                # print(len(y))
                
                colnames.append( str(self.volts[i]) + "_t" )
                colnames.append( str(self.volts[i]) + "_i" )
            
            # merge columns 
            df_merge = pd.concat(df_merge, axis=1).reset_index(drop=True)
            df_merge.columns = colnames 
            
            # move nan to end 
            # https://stackoverflow.com/a/64332789
            df_merge = df_merge.apply(lambda x: pd.Series(x.dropna().values))
            # print(df_merge.shape)
            
            return df_merge 
        
        def merge_pdfs(pdf_to_add):
            """
            pdf_to_add = path to pdf file to append to `pdf_dir`
            """
            if pdf_dir is None:
                print("Tried merging pdf of normalized plots, but `pdf_dir` was not specified.")
                exit()
                            
            # create pdf merger object
            merger = PdfFileMerger()
            
            # append pdf1 to pdf0 
            merger.append(PdfFileReader(pdf_dir))       # previous pdf 
            merger.append(PdfFileReader(pdf_to_add))    # new pdfs 
            
            # write the merged pdf to the same location as the original pdf 
            merger.write(pdf_dir) 
            # close files 
            merger.close()           
            
            try:
                # delete tmp pdf file 
                os.remove(r"./tmp000.pdf")
            except:
                print(" Tried removing tmp PDF file in home directory, but failed.")
                pass 
        
        if "ramp_de" in self.pname:
            p0 = self.boltz(-35, *self.boltz_params)
            dfa = apply_normalization(self.df[0], test=False, p0 = p0)
            dfd = apply_normalization(self.df[1], test=True, p0 = p0)        
        elif "act" in self.pname:
            print("Tail Pmin", self.boltz(postpulse, *self.boltz_params))
            dfa = apply_normalization(self.df[0], test=True,
                    p0 = self.boltz(-35, *self.boltz_params))
            dfd = apply_normalization(self.df[1], test=False, 
                    p0 = self.boltz(postpulse, *self.boltz_params))

        else:
            dfa = apply_normalization(self.df[0], test=False,
                    p0 = self.boltz(-35, *self.boltz_params))
            dfd = apply_normalization(self.df[1], test=True,
                    p0 = self.pmins.values)

        # separate time and Po columns for each trace 
        if reduce > 0:
            dfa_r = apply_reduction(dfa)
            dfd_r = apply_reduction(dfd)

            if pdf_dir is not None:
                # create a temporary pdf to hold the plots of normalized data 
                pdf_to_add = PdfPages(r"./tmp000.pdf")
                plot_normalized([dfa, dfd, dfa_r, dfd_r], show=show, out_pdf=pdf_to_add)
                
                # mrege plots of normalized data with previously generated PDF of plots 
                merge_pdfs(r"./tmp000.pdf")
                
            else:
                plot_normalized([dfa, dfd, dfa_r, dfd_r], show=show)
                
            return dfa, dfd, dfa_r, dfd_r         
        else:
            if pdf_dir is not None:
                # create a temporary pdf to hold the plots of normalized data 
                pdf_to_add = PdfPages(r"./tmp000.pdf")
                plot_normalized([dfa, dfd], show=show, out_pdf=pdf_to_add)
                
                # mrege plots of normalized data with previously generated PDF of plots 
                merge_pdfs(r"./tmp000.pdf")
                
            elif show:
                plot_normalized([dfa, dfd], show=show)
                
            return dfa, dfd 
        
class process():
    def __init__(self, 
            filter_criteria = {},
            show_protocols = True, 
            files_to_skip = [], 
            dates_to_save = 'None', 
                show_abf_segments = False,
                show_csv_segments = False,
            show_leak_subtraction = False,
            show_Cm_estimation = False, 
            show_MT_estimation = False,
                do_exp_kinetics = False,
                do_activation_curves = False,
                do_ramp_stuff = False,
                do_inst_IV = False,
            save_AggregatedPDF = False,
            do_pubplots = False,
                normalize = False,
                remove_after_normalize = {}
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
            exit()
        else:    
            read_ephys_info = EphysInfoFiltering(filter_criteria)
            filenames, ephys_info = read_ephys_info.filter()            
            paired_files, exp_params = read_ephys_info.ExpParams(ephys_info)
        
        # path to main directory 
        main_dir = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/Github_repos/hcn-gating-kinetics/"
        # path to csv files 
        data_path = main_dir + r"data/current_time_course/Pooled_2020/"
        
        # load csv files containing corresponding data 
        paths = [data_path + "%s.csv" % f for f in filenames]
        data_files = {} 
        
        to_remove = [] # indices of CSV files that weren't found 
        for i, x in enumerate(paths):
            if os.path.exists(x):
                df = pd.read_csv(x, header=None, index_col=0)

                # check if index name is not 0; if so, remove header row 
                if df.index[0] != 0:
                    df = df.iloc[1:, :]
                    df.index = df.index.astype(float)       # change index type to float
                
                data_files.update(
                    {filenames[i]: df})
            else:
                print("CSV file not found. ", filenames[i])
                print(" Removing %s from processing." % filenames[i])
                to_remove.append(i) 

                #remove file not found from dataframes of recording parameters
                ephys_info = ephys_info[ephys_info['Files'] != 
                                        filenames[i]]
                exp_params.drop(filenames[i], axis=0, inplace=True)
        
        # remove from `filenames` if CSV file not found 
        if len(to_remove) > 0:
            print("     Files kept: \n", ephys_info.loc[:,["Files", "Protocol"]])
            
            h = 0 
            for t in to_remove:
                del filenames[t-h]
                h += 1 
        
        # for k in data_files.keys():
        #     print(data_files[k]) 
        #     plt.plot(data_files[k].iloc[:,:5])
        #     ### plt.show()
        # exit()
        
        #path to raw abf files 
        abf_path = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/data_files/delbert/"
        # search subdirectories of entire `abf_path` for the original .abf files
        abf_files = [pyabf.ABF(glob.glob(abf_path + "**/%s*.abf" % f, recursive=True)[0]) for f in filenames] 
        
        # check that abf files were found 
        if len(abf_files) != len(filenames):
            print("Number of filenames is %d, but only found %d abf files" % (
                len(filenames), len(abf_files)))
            exit()
                        
        #create dictionary that holds, for each unique filename, start and end of first leak ramp 
        self.ramp_startend = {}
        # dict to hold start and end of +20mV membrane test steps
        self.mt_startend = {} 
        #dictionary to hold filename : {epoch1:[t0, t1], ...}
        self.epoch_startends = {} 
        # dictionary to hold sampling frequencies in kHz  
        self.dataRates = {} 
        
        # loop over .ABF files found, and find start and end of leak ramps
        for i, a in enumerate(abf_files):
            print("\n Reading...", filenames[i])
            
            #sampling frequency in kHz 
            khz = a.dataRate / 1000
            self.dataRates.update({filenames[i] : khz})
            
            #find first voltage ramp for the first trace, then assume this is constant
            a.setSweep(0)                               
            epochtypes = a.sweepEpochs.types 
            
            # find index of first ramp epoch 
            if "Ramp" in epochtypes[:10]:
                first_ramp_idx = epochtypes[:10].index('Ramp')
            else:
                print("Skipping {f} because no ramp epoch found in the first 10 steps.".format(
                    f=filenames[i]))               
                continue 
                
            #check if membrane test step to +20mV is present within 1-3 epochs after the ramp
            if 20.0 in a.sweepEpochs.levels[first_ramp_idx+1:first_ramp_idx+4]:
                mt_idx = a.sweepEpochs.levels[first_ramp_idx+1:first_ramp_idx+4].index(20.0) 
                dt_ = mt_idx + 3    # relevant pulses start 3 epochs after the membrane test step
                
                # `mt_idx` was relative to `first_ramp_idx` -> get absolute index 
                mt_idx += first_ramp_idx + 1 
                self.mt_startend.update(
                    {
                        filenames[i] :
                        a.sweepEpochs.p1s[mt_idx:mt_idx+3] 
                    }
                )        
                                                
            # else, relevant pulses start after the last -35mV step 
            else:
                # number of epochs b/w end of leak ramp and start of relevant protocols
                dt_ = 1 + next(
                    (i for i, x in enumerate(
                        a.sweepEpochs.levels[first_ramp_idx+1:first_ramp_idx+4]
                        ) if x != -35.0)
                ) 
                print(dt_)
                # dt_ = 3     
            
            #add indices of the start and end of the ramp 
            t0 = a.sweepEpochs.p1s[first_ramp_idx]
            t1 = a.sweepEpochs.p1s[first_ramp_idx+2] 
            
            self.ramp_startend.update(
                {filenames[i] : file_specific_transform(filenames[i], [t0, t1])}
            ) 
            # print(a.sweepEpochs.p1s)
            # print(t0, t1)
            
            #add epoch start and end times for each sweep
            h = 0 #if csv and abf are 1:1, this stays zero; otherwise, increase by 1 so that we keep indexing csv at the jth column while proceeding through the abf file 
            
            # print(a.sweepList)    # 0-indexed list of indices of sweeps, e.g. range(# sweeps)
            N = int(data_files[filenames[i]].shape[1]/2) 
            
            for j in a.sweepList:
                a.setSweep(j)
                
                # don't exceed number of traces in the .csv file 
                if N + j - h >= 2*N:
                    break 
                
                #check if jth sweep is in both abf and csv files by subtracting their sums (voltage commands) 
                abf_sum = sum(a.sweepC) 
                csv_sum = data_files[filenames[i]].iloc[:,N+j-h].sum()
                
                # apply file-specific transform if necessary to enforce equality 
                csv_sum = file_specific_transform(filenames[i], (abf_sum, csv_sum))
                
                # print(j, h, abs((abf_sum - csv_sum)/abf_sum))
                if abs((abf_sum - csv_sum)/abf_sum) < 0.01:  
                    try:
                        self.epoch_startends[filenames[i]].update({
                            j : a.sweepEpochs.p1s[first_ramp_idx+dt_:]
                        })
                    except:
                        self.epoch_startends.update({
                            filenames[i] : {
                                j : a.sweepEpochs.p1s[first_ramp_idx+dt_:]
                            }
                        })
                else:
                    h += 1 
                    
                    # df = data_files[filenames[i]]
                    # N = int(df.shape[1]/2)
                    
                    # for g, h in enumerate(a.sweepList):
                    #     a.setSweep(h) 
                    #     plt.plot(df.iloc[:,N+g])
                    #     plt.plot(df.index, a.sweepC, 
                    #             marker='o', ls='none', 
                    #             markersize=4, markevery=250) 
                        
                    # ### plt.show()
                    # exit()

            # apply transform to `mt_startend` and `epoch_startends` if file requires it 
            self.mt_startend = file_specific_transform(filenames[i], self.mt_startend)
            self.epoch_startends = file_specific_transform(filenames[i], self.epoch_startends)
            
            # print(self.mt_startend)
            # print(self.epoch_startends.keys())
            # exit()
            
            # continue 
            
            # plot segmented epochs from protocol
            if show_abf_segments:
                f, ax = plt.subplots(2,1, figsize=(8,6))
                
                for sweepNumber in a.sweepList:
                    a.setSweep(sweepNumber)                               
                    times = np.array(a.sweepX) * 1000
                    ax[0].plot(times[::2], a.sweepY[::2], c='lightblue', alpha=0.7)
                    ax[1].plot(times[::2], a.sweepC[::2], c='r', alpha=0.7)                
            
                    # for x, p1 in enumerate(a.sweepEpochs.p1s[first_ramp_idx+dt_:]):
                    for x, p1 in enumerate(a.sweepEpochs.p1s):
                        # epochLevel = a.sweepEpochs.levels[x]
                        # epochType = a.sweepEpochs.types[x]
                        # print(f"epoch index {i}: at point {p1} there is a {epochType} to level {epochLevel}")
                        
                        for j in range(2):
                            ax[j].axvline(p1/khz, c='white', ls='--', lw=2, alpha=0.5)

                ax[0].set_title(filenames[i])
                ax[0].set_ylabel("Current")
                ax[1].set_ylabel("Command Voltage")
                ax[1].set_xlabel("Time")
                
                plt.tight_layout()
                plt.show()
                plt.close()

            # plot segmented epochs on .csv file 
            if show_csv_segments:
                df = data_files[filenames[i]] 
                
                #number of traces 
                N = int(df.shape[1]/2) 
                
                f, ax = plt.subplots(2, 1, figsize=(14,5))
                for j, g in enumerate(self.epoch_startends[filenames[i]].keys()):
                    ax[0].plot(df.iloc[::5,j], lw=2)
                    ax[1].plot(df.iloc[::5,N+j], lw=2)
                    
                    try:
                        for k in range(2):
                            for n, u in enumerate(self.epoch_startends[filenames[i]][g]):
                                
                                # if epoch (likely if transformed) exceeds dimensions, remove it
                                if u >= df.shape[0]:
                                    self.epoch_startends[filenames[i]][g] = self.epoch_startends[filenames[i]][g][:n] 
                                    break
                                else:
                                    ax[k].axvline(u/khz, c='white', ls='--', alpha=0.5)
                    except:
                        print(self.epoch_startends)
                        plt.close()
                        plt.plot(df.iloc[:,:N])
                        ### plt.show()
                        raise 
                    
                ax[0].set_title(filenames[i])
                ax[0].set_ylabel("Current")
                ax[1].set_ylabel("Voltage")
                ax[1].set_xlabel("Time (ms)")
                
                plt.tight_layout()
                plt.show()
                plt.close()
        
        #assign self variables for class 
        self.dates_to_save = dates_to_save
        self.main_dir = main_dir 
        self.save_path = main_dir + "output/Processing/Pooled_Analyses/"
        self.ephys_info = ephys_info 
        self.exp_params = exp_params        # experimental recording parameters 
        self.paired_files = paired_files    # recordings from same cell {parent filename : [[subsequent files], [activation files]]}
        self.filenames = filenames 
        self.data_files = data_files
        self.abf_files = abf_files 
        
        # create prefix for output 
        self.output_prefix = read_ephys_info.CreatePrefix()
                
        #for downstream options
        self.show_protocols = show_protocols 
        self.show_abf_segments = show_abf_segments
        self.show_csv_segments = show_csv_segments
        self.do_pubplots = do_pubplots 
        self.show_leak_subtraction = show_leak_subtraction
        self.show_Cm_estimation = show_Cm_estimation 
        self.show_MT_estimation = show_MT_estimation 
        self.do_ramp_stuff = do_ramp_stuff
        self.do_exp_kinetics = do_exp_kinetics
        self.do_activation_curves = do_activation_curves
        self.do_inst_IV = do_inst_IV 
        self.normalize = normalize 
        self.remove_after_normalize = remove_after_normalize 
        self.save_AggregatedPDF = save_AggregatedPDF
        
        #for computations 
        self.exp_fit_params = []    # parameters for exp1-3 fitting 
        self.exp_fit_delay = {}     # delay in exp1 
        self.ac_fit_params = []     # parameters for activation curve 
        self.ac_norm_data = []      # normalized activation curve 
        self.tail_post_pmins = []   # Pmin from post-pulse currents 
        self.ramp_stats = []        # Ramp hysteresis statistics
        self.IV_params = []         # Summary statistics from I-V curves: Erev, P_K, P_Na
        self.IV_currents = []       # instantaneous current and current dnesity vs. voltage
    
    def etc(self, save_leaksub=True, save_extracted=True):
        """
        Runs processing functions.  
        
        `save_leaksub` = whether to save leak-subtracted data  
        `save_extracted` = whether to save leak-subtracted, extracted data  
        """
        
        filenames = self.filenames 
        ephys_info = self.ephys_info
        data_files = self.data_files
        
        # hold leak subtracted dataframes, {filename : leak-subtracted dataframe}
        data_files_Lsub = {}        

        for i, df in enumerate(data_files.values()):
            print("\n Leak subtracting... ", filenames[i])

            khz = int( self.dataRates[filenames[i]] ) 
            N = int(df.shape[1]/2)

            #copy dataframe 
            df1 = df.copy()       
            
            # step intervals for sweeps in current file 
            intervals = self.epoch_startends[filenames[i]]     
            
            # get name of protocol for the current recording 
            try:
                pname = ephys_info.loc[ephys_info['Files'] == filenames[i], "Protocol"].iat[0] 
            except:
                plt.plot(df1.iloc[:,:int(df1.shape[1]/2)])
                ### plt.show()
                plt.close()
                
                print("Couldn't find protocol name for %s" % filenames[i])
                print(filenames[i], ephys_info.loc[:,['Files', 'Protocol']])
                raise 
            
            # create PDF for holding multiple plots 
            if self.save_AggregatedPDF:
                s = r"{savedir}/PDFs/{fname}-{pname}.pdf".format(savedir=self.save_path, fname=filenames[i], pname=pname)
                PDFPlots = PdfPages(s)
                
                print("Created PdfPages object, which will be saved at: \n  {pdfloc}".format(pdfloc=s))
            else:
                PDFPlots = None 
            
            #create leak subtract object, then perform subtraction 
            # print(self.ramp_startend[filenames[i]])
            df_Lsub = leak_subtract(self.ramp_startend[filenames[i]], 
                        khz=khz, epochs=intervals, residual=False)
            # perform leak subtraction 
            df1 = df_Lsub.do_leak_subtraction(df1, method="ohmic", 
                    plot_results = self.show_leak_subtraction, pdfs=PDFPlots)            

            # experimental seal/whole cell parameters 
            # if any lists in experimental parameters, take the mean of each element 
            if np.any([isinstance(x, list) for x in self.exp_params.loc[filenames[i], :]]):
                expp = [np.mean(x) for x in self.exp_params.loc[filenames[i],:]]
            else:
                expp = self.exp_params.loc[filenames[i],:]

            # quality of voltage clamp wrt frequency, shows corner frequency 
            BC_VoltageClampQuality(filenames[i], expp, pdfs=PDFPlots, show=False)
            
            # save leak-subtracted dataframe 
            s = r"{maindir}/output/Processing/Processed_Time_Courses/leaksub/{fname}_leaksub.csv".format(maindir=self.main_dir, fname=filenames[i])
            # df1.to_csv(s)
            
            data_files_Lsub.update({filenames[i]:df1}) 
            
            if self.show_leak_subtraction:
                print(" Showing leak subtraction for %s..." % filenames[i])
                
                f, ax = plt.subplots(2,1,figsize=(12,6))
                N = int(df.shape[1]/2)
                ax[0].plot(df.iloc[::4,:N], lw=1, c='lightblue', alpha=0.4, label="Original")
                ax[0].plot(df1.iloc[::4,:N], lw=2, alpha=1, c='r', label="Subtracted")
                
                # times demarcating leak ramp 
                if len(self.ramp_startend[filenames[i]]) > 2:
                    ts = self.ramp_startend[filenames[i]]
                    
                    for j in range(0, len(ts), 2):
                        t0, t1 = ts[i:i+2]
                        k = int(j/2)
                        
                        dt = int((t1 - t0)/4)
                        if t0 - dt > 0:
                            t1 += dt + 1
                            t0 -= dt 
                        else:
                            t1 += (t0 - 250*khz) + 1
                            t0 = 250*khz 
                            
                        if j == 0:
                            ax[1].plot(df.iloc[t0:t1+1:4, k], lw=2, c='lightblue', 
                                    alpha=0.7,   label="Original") 
                            ax[1].plot(df1.iloc[t0:t1:4, k], lw=2, alpha=0.7, 
                                    c='r', label="Subtracted") 
                        else:
                            ax[1].plot(df.iloc[t0:t1+1:4, k], lw=2, c='lightblue', 
                                    alpha=0.7,   label=None) 
                            ax[1].plot(df1.iloc[t0:t1:4, k], lw=2, alpha=0.7, 
                                    c='r', label=None) 
                    
                else:
                    t0, t1 = self.ramp_startend[filenames[i]]
                    # print(t0, t1, df.shape[0])
                    
                    # show quarter of the total duration before and after the ramp 
                    dt = int((t1-t0)/4)
                    if t0 - dt > 0:
                        t1 += dt + 1
                        t0 -= dt 
                    # if duration before onset of leak ramp is less than half the total ramp duration 
                    else:
                        # start plotting at 200ms and adjust plotting interval to have equal length after end of the ramp
                        t1 += (t0 - 250*khz) + 1
                        t0 = 250*khz 
                        
                    ax[1].plot(df.iloc[t0:t1+1:4, :N], lw=2, c='lightblue', 
                            alpha=0.7, label="Original") 
                    ax[1].plot(df1.iloc[t0:t1:4, :N], lw=2, alpha=0.7, c='r', label="Subtracted") 
                                                    
                for j in range(2):
                    h, l = ax[j].get_legend_handles_labels()
                    
                    if j == 0:
                        ax[j].legend([h[0], h[-1]], [l[0], l[-1]], loc="lower right",
                                bbox_to_anchor=[0.81, 0])
                    else:
                        ax[j].legend([h[0], h[-1]], [l[0], l[-1]], loc="upper right")
                    
                    ax[j].set_ylabel("Current (pA)")
                
                ax[0].set_title("Leak Subtraction using Linear Voltage Ramps")
                ax[1].set_xlabel("Time (ms)")                   
                                    
                s = "%s \n$R_p$ = %.1f $M\\Omega$\n$R_{seal}$ = %.1f $G\\Omega$\n$C_m$ = %.1f pF\n$R_m$ = %.1f $M\\Omega$\n$R_{series}$ = %.1f $M\\Omega$" % (filenames[i], expp[0], expp[1], expp[2], expp[3], expp[4])
                ax[0].text(0.987, 0.315, s, 
                        # color = 'k', 
                        ha='right', va='center', 
                        transform=ax[0].transAxes,  # use axes coords instead of data coordinates
                        bbox=dict(fc='k', alpha=0.5, ec='white', lw=1)
                )
                
                for j in range(2):
                    ax[j].locator_params(axis='x', nbins=5)
                    ax[j].locator_params(axis='y', nbins=4)
                
                plt.tight_layout()
                
                if self.save_AggregatedPDF:
                    PDFPlots.savefig(Figure=f)
                    
                plt.show()
                plt.close()
                # exit()
            
            if self.show_Cm_estimation:       
                print(" Cm estimation for %s..." % filenames[i])    
                
                cm_values = estimate_Cm(self.ramp_startend[filenames[i]], df, khz=khz) 
                plt.plot(range(N), cm_values, marker='o')
                
                plt.title(r"$C_m$ estimation")
                plt.tight_layout()
                
                if self.save_AggregatedPDF:
                    PDFPlots.savefig()
                
                # plt.show()
                plt.close()
                
            if self.show_MT_estimation:
                if filenames[i] in self.mt_startend.keys():
                    print(" Membrane test estimation for %s..." % filenames[i])
                    estimate_MT(self.mt_startend[filenames[i]], df, khz=khz, pdf=PDFPlots)
            
            # get upper index that bounds test pulse in protocol `pname`
            u = get_upper_index(df, pname)
            if u is None:
                continue 
                        
            if self.do_pubplots:
                print(" Making pub plots for %s..." % filenames[i])
                make_figure(df1, intervals, khz=khz, show=True, save_path=None, pdfs=PDFPlots)
        
            # extract test pulses
            print(" Extracting test pulses for %s..." % filenames[i])
            
            if "env" in pname:
                print(intervals)
                v_, env_times, df_to_fit, df_protocol = extract_traces(df, u, intervals, 
                                                            N, khz, env=True, return_voltages=True)
            else:
                # if ramp, return half-durations along with current
                if "ramp_dt" in pname:
                    tmids, df_to_fit, df_protocol = extract_traces(df, u, intervals, 
                                                        N, khz, ramp="dt", return_voltages=True)
                elif "ramp_de" in pname:
                    tmids, df_to_fit, df_protocol = extract_traces(df, u, intervals, 
                                                        N, khz, ramp="de", return_voltages=True)
                # else, return test pulse voltages 
                else:
                    vtest, df_to_fit, df_protocol = extract_traces(df, u, intervals, 
                                                        N, khz, return_voltages=True)
                    
            if save_extracted:
                # path to directory to save files in 
                s = r"{maindir}/output/Processing/Processed_Time_Courses/leaksub/extracted/{fname}_leaksub_extracted.csv".format(maindir=self.main_dir, fname=filenames[i])
                
                # append voltage protocol columnwise 
                pd.concat([df_to_fit, df_protocol], axis=1).to_csv(s)

            if self.do_ramp_stuff:
                if "ramp_dt" in pname:
                    print(" Ramp analysis for %s..." % filenames[i])
                    
                    _, df_, pro = extract_traces(df1, u, intervals, N, khz, 
                                        return_voltages=True, ramp="dt")
                    
                    # initialize class object for hysteresis summaries 
                    ramp_H = analyze_ramp_dt(pro, df_, tmids, N, khz)
                    # run analyses and return statistics
                    ramp_stats = ramp_H.H(plot=True, pdf=PDFPlots)

                    # save ramp statistics
                    df_ramp_stats = pd.DataFrame(data = {# "Filename" : [filenames[i]]*N,
                                    "rates" : ramp_stats[0], "dA" : ramp_stats[1],
                                    "vA" : ramp_stats[2], "dI" : ramp_stats[3],
                                    "dT" : ramp_stats[4], "dTmids" : ramp_stats[5] 
                        })
                    
                    # if df_ramp_stats.shape[0] > 1:
                    df_ramp_stats.index = [filenames[i] + "_%d" % j for j in range(df_ramp_stats.shape[0])]
                    # else:
                    #     df_ramp_stats.index = filenames[i] + "_0"
                    
                    self.ramp_stats.append( df_ramp_stats )

            if self.do_exp_kinetics:
                df_fit = exp_fitting(df_to_fit, khz, volts=vtest)
                
                if self.save_AggregatedPDF:
                    fit_params, exp_delay = df_fit.get_fit(pdf=PDFPlots) 
                else:
                    fit_params, exp_delay = df_fit.get_fit() 
                    
                #convert exp fit params into dataframe, then add 
                fit_params = pd.DataFrame.from_dict(fit_params) 
                fit_params.columns = vtest 
                
                #add filename column 
                fit_params.insert(loc=0, column='FileName', value=[filenames[i]]*3)
                
                self.exp_fit_params.append(fit_params) 
                
                #extract delay for exp1, then add 
                exp_delay = {vtest[j]:x[0] for j, x in enumerate(exp_delay)} 
                self.exp_fit_delay.update({filenames[i]:exp_delay})           

            #compute activation curve and tail pmins from leak-subtracted dataframe `df1`
            if self.do_activation_curves:
                # skip `ramp_dt` (ie hysteresis) protocols, which might slip through
                if "ramp" in pname:
                    pass 
                else:
                    if "_act_" in pname:  
                        print(" Activation curve and/or Pmins for %s..." % filenames[i])
                        
                        # use `u+1`, since GV calculation is concerned with the tail steps, not the test pulses 
                        v_, df_to_ac = extract_traces(df1, u+1, intervals, N, khz, ramp="x") 
                        
                        # compute activation curve 
                        ac = activation_curve(vtest, df_to_ac, khz, 
                                    plot_results=True, show=True, pdf=PDFPlots)
                        
                        # fitting is done in the previous step; here, we just get the parameters and normalized current values 
                        norm_tails, ac_pars = ac.do_fit() 
                        # pmins 
                        tail_pmins = ac.tail_pmins() 
                        
                        norm_tails = pd.DataFrame(data={filenames[i]:norm_tails})
                        
                        if norm_tails.shape[0] < len(vtest):
                            vtest = ac.return_test_voltages()
                        norm_tails.index = vtest 
                        
                        if len(ac_pars) == 3:
                            ac_pars = pd.DataFrame(
                                data={"Vh":[ac_pars[0]], "s":[ac_pars[1]], "c":[ac_pars[2]]}
                            )
                        else:
                            ac_pars = pd.DataFrame(
                                data={"Vh":[ac_pars[0]], "s":[ac_pars[1]]}
                            )
                        ac_pars.index = [filenames[i]]
                        
                        self.ac_norm_data.append(norm_tails) 
                        self.ac_fit_params.append(ac_pars) 
                        
                    elif "_de_" in pname:
                        v_, df_tails = extract_traces(df1, u, intervals, N, khz, ramp="x")
                        v_post, df_post = extract_traces(df1, u+1, intervals, N, khz, ramp="x") 
                        
                        if sum(v_post) == v_post[0]*len(v_post):
                            if v_post[0] in vtest: 
                                
                                ac = activation_curve(vtest, df_tails, khz, post_tails=df_post, 
                                        post_tail_voltage=v_post[0], show_pmin=True, 
                                        show=False, pdf=PDFPlots) 
                                        
                                d = ac.tail_pmins()   
                                if isinstance(d[0], pd.Series):
                                    tail_mins, post_mins = d 
                                    df_pmins = pd.DataFrame(
                                        data={
                                            "Voltages" : vtest,
                                            "FileName" : filenames[i],                                        
                                            "Tail_Pmin": tail_mins,
                                            "PostTail_Pmin": post_mins}
                                    )
                                else:
                                    tail_mins = d 
                                    df_pmins = pd.DataFrame(
                                        data={
                                            "Voltages" : vtest, 
                                            "FileName" : filenames[i],
                                            "Tail_Pmin": tail_mins}
                                    )
                                
                                self.tail_post_pmins.append( df_pmins ) 
                        
            if self.do_inst_IV:
                if "ramp" in pname or "env" in pname:
                    pass 
                else:
                    # do I-V analysis
                    IV_params, Iinst_df = df_Lsub.IV_analysis(df_to_fit, df_protocol,
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
                    print("PDF object was not closed properly. Maybe it was closed during processing above? \n")
                                        
        # convert summarized analytics into dataframes 
        if self.do_activation_curves:
            if len(self.ac_norm_data) > 0:
                
                try:
                    self.ac_norm_data = pd.concat([d.dropna() for d in self.ac_norm_data], axis=1)
                except:
                    print(self.ac_norm_data)
                    exit()
                    
                self.ac_fit_params = pd.concat(self.ac_fit_params, axis=0)
            
            if len(self.tail_post_pmins) > 0:            
                self.tail_post_pmins = pd.concat(self.tail_post_pmins, axis=0)
                self.tail_post_pmins.set_index("Voltages", drop=True, inplace=True)
                
        if self.do_ramp_stuff:
            if len(self.ramp_stats) > 0:
                self.ramp_stats = pd.concat(self.ramp_stats, axis=0)
            
        if self.do_exp_kinetics:
            if len(self.exp_fit_params) > 0:
                self.exp_fit_delay = pd.DataFrame.from_dict(self.exp_fit_delay)
                self.exp_fit_params = pd.concat(self.exp_fit_params, axis=0)
        
        if self.do_inst_IV:
            if len(self.IV_params) > 0:
                self.IV_params = pd.concat(IV_params, axis=1)
            
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
            for i, df in enumerate(data_files.values()):
                try:
                    pname = ephys_info.loc[ephys_info['Files'] == filenames[i], "Protocol"].iat[0] 
                except:
                    print("Couldn't find protocol name for %s" % filenames[i])
                    raise 
                    
                if filenames[i] != "20d10008":
                    continue 
                    
                if "ramp_dt" in pname:
                    print("Skipping normalization of {f}, with protocol {p}, which is currently not normalized.".format(f=filenames[i], p=pname))
                    continue 
                elif "ramp_de" in pname:
                    extract_option = "de"
                else:
                    extract_option = "x" 
                
                if self.save_AggregatedPDF:
                    # path to PDF of previously generated plots 
                    PDFPlots = r"{savedir}/PDFs/{fname}-{pname}.pdf".format(savedir=self.save_path, fname=filenames[i], pname=pname)
                else:
                    PDFPlots = None 
                
                print("Normalizing... %s \n     Protocol = %s " % (filenames[i], pname))
                                                
                khz = int( self.dataRates[filenames[i]] ) 
                N = int(df.shape[1]/2)
                
                # get leak subtracted dataframe 
                df_Lsub = data_files_Lsub[filenames[i]]
                
                # step intervals for sweeps in current file 
                intervals = self.epoch_startends[filenames[i]]
                
                # x1, x2 are either voltages or ramp durations 
                # ind = 2, 3 -> upper bound + 1 for indexing epochs 
                #   i.e. ind=2 implies test interval is bounded by epochs 0 and 1
                #   e.g. intervals[0][0:2]
                #   e.g. ind=3 -> intervals[0][1:3] 
                # activation
                x1, df_act = extract_traces(df_Lsub, 2, intervals, N, khz, ramp=extract_option)
                # deactivation 
                x2, df_de = extract_traces(df_Lsub, 3, intervals, N, khz, ramp=extract_option)
                
                if "de" in pname:
                    # remove traces if specified 
                    df_act.columns = x2 
                    df_de.columns = x2 
                    if filenames[i] in self.remove_after_normalize.keys():
                        df_act, df_de, x2 = remove_traces([df_act, df_de], khz, x2, 
                                            self.remove_after_normalize[filenames[i]], show=False)
                            
                    # prepulse for deactivation 
                    #   intervals[1][0] = epoch (start of hyperpolarization) for 1st trace
                    #   N+1 = voltage command for 2nd trace
                    v = int(df_Lsub.iat[intervals[0][0]+100*khz, N])
                    print("Prepulse voltage for deactivation", v)                    
 
                    # look for filename of activation protocol from the same cell, if available
                    paired = None 
                    for val in self.paired_files.values():
                        if isinstance(val[1], list):
                            paired = val[1][0] 
                            break
                    
                    if "ramp_de" in pname: 
                        nrm = normalize_for_fitting(pname, filenames[i], [df_act, df_de], 
                                                khz, volts=x2, GV=self.ac_norm_data, boltz_params=self.ac_fit_params, paired=paired)
                    
                        dflis = nrm.do_norm(prepulse=v, reduce=250, show=True, pdf_dir=PDFPlots)
                    
                    else:
                        # continue 
                        # get Pmins corresponding to given file
                        Pmins = self.tail_post_pmins.loc[self.tail_post_pmins["FileName"] == filenames[i]].mean(axis=1)
                        # Drop empty rows 
                        Pmins.dropna(how="all", axis=0, inplace=True)                        
                        
                        nrm = normalize_for_fitting(pname, filenames[i], [df_act, df_de], 
                                                    khz, volts=x2, pmins=Pmins, 
                                                    GV=self.ac_norm_data, boltz_params=self.ac_fit_params, paired=paired)
                        
                        if self.save_AggregatedPDF:
                            # prepulse=None, postpulse=None, reduce=0, show=False, pdf=None
                            dflis = nrm.do_norm(prepulse=v, reduce=250, show=False, pdf_dir=PDFPlots)
                        else:
                            dflis = nrm.do_norm(prepulse=v, reduce=250, show=False)

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
                    v = int(df_Lsub.iat[intervals[0][1]+100*khz, N])
                    print("Deactivating voltage for activation = ", v)
                    # continue 
                
                    nrm = normalize_for_fitting(pname, filenames[i], [df_act, df_de], 
                                                khz, volts=x1, GV=self.ac_norm_data, boltz_params=self.ac_fit_params)
                    
                    dflis = nrm.do_norm(postpulse=v, reduce=250, show=True, pdf_dir=PDFPlots)
                
                # save normalized dataframes as CSVs 
                # destination 
                dest = r"{m}/output/Processing/Processed_Time_Courses/{f}".format(m=self.main_dir, f=filenames[i])
                if "ramp_de" in pname:
                    dest += "_rampde_"
                elif "de" in pname:
                    dest += "_de_"
                else:
                    dest += "_act_"
                    
                # dflis[0].to_csv(dest + "act_norm.csv")
                # dflis[1].to_csv(dest + "de_norm.csv")
                # dflis[2].to_csv(dest + "act_norm_reduced.csv")
                # dflis[3].to_csv(dest + "de_norm_reduced.csv")

                # exit()
                
    def go(self, idle=False, save_csv=False, append=False):
        """
        Run processing pipeline
        
        idle = if true, do nothing
        append = if true, try to append to existing .csv, else write new .csv
        """

        if not idle: 
            self.etc() 
            
            if save_csv:
                # path to save to 
                path = self.save_path + "/CSV_output/"
                
                # specify filenames and data to save
                labels = []
                output = [] 
                
                print("`save_csv` is enabled. output will be saved at {p} with file prefix {pf}".format(p=path, pf=self.output_prefix))                
                
                if self.do_activation_curves:
                    labels.extend(["act_norm.csv", "boltz_params.csv", "tail_post_pmins.csv"])
                    output.extend([self.ac_norm_data, self.ac_fit_params, self.tail_post_pmins])
                elif self.do_ramp_stuff:
                    labels.append("ramp_stats.csv")
                    output.append(self.ramp_stats)
                elif self.do_exp_kinetics:
                    labels.extend(["exp_delay.csv", "exp_params.csv"])
                    output.extend([self.exp_fit_delay, self.exp_fit_params])
                elif self.do_inst_IV:
                    labels.extend(["IV_params.csv", "IV_currents.csv"])
                    output.extend([self.IV_params, self.IV_currents])
                                
                print(" Output that will be saved: ", labels)
                
                # add prefix to output filenames to specify the files and protocols used as input to the call to `process`
                if len(labels) + len(output) > 0:
                    labels = ["{prefix}__{s}".format(prefix=self.output_prefix, s=label) for label in labels]
                # print(labels)
                
                #avoid ellipses when saving long arrays (e.g. lists of parameters) in .csv 
                np.set_printoptions(threshold=1e9) 
                
                for (i, o) in enumerate(output):
                    if len(o) < 1:
                        print(" %s is empty. Skipping creation of .csv file." % labels[i])
                        continue 
                    else:
                        if append:
                            try:
                                o.to_csv(path + labels[i], mode='a', header=False)
                                print(" %s appended successfully." % labels[i])
                            except:
                                print(" Could not append %s. Creating new file." % labels[i])
                                o.to_csv(path + labels[i])
                                print(" %s created successfully." % labels[i])
                        else:
                            try:
                                o.to_csv(path + labels[i])
                                print(" %s created successfully." % labels[i])
                            except:
                                print(o)
                                print(" Could not write %s. Skipping." % labels[i])
                                continue 
                                            
    def summarize(self, title=None, output=False):
        out_files = {} 
        path = self.save_path
        
        if self.save_AggregatedPDF:
            s = "-".join(self.dates_to_save) 
            PDFPlots = r"{savedir}/summary_output/summary_{dates}.pdf".format(savedir=self.save_path, dates=s)
        else:
            PDFPlots = None 
        
        # return mean and SEM values for dataframe, averaging over axis `n` 
        def get_mu_sem(df, n=1):
            mu_ = df.mean(axis=n) 
            err_ = [sem(df.iloc[i,:].dropna()) for i in range(df.shape[0])]
            return mu_, err_
        
        # modified boltzmann with adjustable Pmin = c 
        def b(v, pars):
            vh, s, c = pars 
            return ((1-c)/(1 + np.exp((v-vh)/s))) + c
        
        # compute vh from boltzmann function, where b(vh) = 0.5 
        def get_vh(pars):
            vh, s, c = pars 
            return -s*np.log(1-2*c) + vh 
                    
        def extract_exp_params(df, order=2, prop=False):
            """
            Given dataframe containing parameters for exponential fits, extract fast and slow time constants, and A_F/A_S ratios
            """
            N = int(df.shape[1])
            vrange_p = df.columns.values.tolist() 
            
            # for each column (voltage), extract all but Constant term into a list, ignoring recordings that lack any entries
            # params = [[x[:-1] for x in df.iloc[:,i].dropna().values] for i in range(N)] 
            params = [df.iloc[:,i].dropna().values.tolist() for i in range(N)] 
            FastTau = [] 
            SlowTau = [] 
            AmpProp = []  

            a = order # index of first tau parameter 
            b = a + order # index of last tau parameter 
            for i, p in enumerate(params):
                
                fast_idx = [np.argmin(x[a:b]) + a for x in p] 
                slow_idx = [np.argmax(x[a:b]) + a for x in p]

                FastTau.append( [
                    1000/x[f] for x, f in zip(p, fast_idx)
                ])
                SlowTau.append([
                    1000/x[s] for x, s in zip(p, slow_idx) 
                ])
                
                if prop:
                    z = [abs(x[f-a]/sum(x[:a])) for x, f, s in zip(p, fast_idx, slow_idx)]
                    # z = [abs(x[f-a]/x[-1]) for x, f, s in zip(p, fast_idx, slow_idx)]
                else:
                    z = [abs(x[f-a]/x[s-a]) for x, f, s in zip(p, fast_idx, slow_idx)]
                    
                AmpProp.append(z)
                    
            FastTau = pd.DataFrame.from_records(FastTau, index=vrange_p)
            AmpProp = pd.DataFrame.from_records(AmpProp, index=vrange_p)
            SlowTau = pd.DataFrame.from_records(SlowTau, index=vrange_p) 
            
            return FastTau, SlowTau, AmpProp
        
        def saver(name, df=None):
            """
            Save `current` figure and dataframe. 
            Calls plt.show and plt.close afterwards, so figure space is cleared afterwards.
            
            name = goes into the filename of .CSV and .PNG output, but only if `output` = True when `self.summarize` is called. 
                Distinct from `title`, which, if available, appears before `name` in the filenames. 
                If `title` is absent, a generic filename is created by searching the save path for filenames containing `name`, then appending a 3-digit number at the end.             
            """
            out_path = path + "summary_output/"
            if output:
                if isinstance(title, str):
                    plt.savefig(out_path + title + "_" + name + ".png")
                else:
                    print("Passing an input for `title` is recommended \
                        when `output` = True. \n \
                        Saving under default name.")

                    # find files with the given suffix
                    FoundFiles = glob.glob(out_path + "%s*.png" % name)
                    if len(FoundFiles) > 0:
                        fname = os.path.basename(FoundFiles[-1])
                        n = int(fname[-7:-4]) + 1
                    
                    name += str(n) 
                                                            
                    plt.savefig(out_path + name + ".png")
                
                if isinstance(df, list):
                    if len(df) < 3:
                        df["SEM"] = df[1] 
                        df.columns[0] = "Mean" 
                        df.to_csv(out_path + title + "_" + name + ".csv")
                        
                    else:
                        # if N is length of list `df`, then 
                        # first 2/3*N are mu and SEM for corresponding data, 
                        # the last 1/3*N elements are names of elements in order 
                        u = int(len(df)/3) 
                        for i in range(1, 2*u, 2):
                            df[i] = pd.Series(df[i], index=df[0].index) 
                        
                        colnames = df[2*u:]
                        
                        df_merge = pd.concat(df[:2*u], axis=1)
                        cols = [""]*df_merge.shape[1]  
                        
                        for i in range(0, 2*u, 2):
                            cols[i] = colnames[int(i/2)] + "_Mean"
                            cols[i+1] = colnames[int(i/2)] + "_SEM"

                        df_merge.columns = cols 
                        print(df_merge)
                        df_merge.to_csv(out_path + title + "_" + name + ".csv")
            
            if self.save_AggregatedPDF:
                PDFPlots.savefig()
                        
            plt.show()
            plt.close()
                    
        for l in self.outfile_labels:
            if l == self.outfile_labels[-1]:
                continue 
            
            df = pd.read_csv(path + l, index_col=0, header=0).dropna(how='all')

            if "exp_params" in l:
                # convert str(list) into list if cells are not empty 
                for i in range(df.shape[0]):
                    for j in range(1, df.shape[1]):
                        if df.iat[i,j] == np.nan: 
                            continue 
                        elif isinstance(df.iat[i,j], str):
                            s = df.iat[i,j][1:-1].split(" ")
                            s = [float(x) for x in s if len(x) > 0]
                            df.iat[i,j] = s 
            
            print(" Output file: %s" % l)
            print(df)
            out_files.update({l[:-4] : df}) 
        
        #average boltz params and compute activation curve 
        mu_params = out_files["boltz_params"].mean(axis=0).values.tolist()

        # compute boltzmann using averaged fit parameters 
        vrange = range(-175, 5, 5) 
        mu_boltz = [b(v, mu_params) for v in vrange]
        
        # parameters for boltzmann fit 
        fb, axb = plt.subplots(1, 3, figsize=(11, 4))
        df_pars = out_files["boltz_params"]
        
        # compute 'true' vhalf from modified boltzmann 
        vh_ = [get_vh(df_pars.iloc[i,:].values) for i in range(df_pars.shape[0])] 
        
        for i in range(3):
            try:
                j = i+1 
                axb[i].plot(df_pars.iloc[:,0], df_pars.iloc[:,j], 
                            marker='o', markersize=5, ls='none')
                axb[i].plot(vh_, df_pars.iloc[:,j], 
                            marker='o', fillstyle='none',
                            markersize=5, ls='none')
                
                axb[i].axhline(df_pars.iloc[:,j].mean(), 
                            c='white', ls='--', alpha=0.5)
                axb[i].axvline(df_pars.iloc[:,0].mean(), 
                            c='white', ls='--', alpha=0.5)
                axb[i].axvline(np.mean(vh_), c='r', lw=2, alpha=0.5,
                            label="'True' Vh\n%.1fmV" % np.mean(vh_))
                
                axb[i].set_xlabel(r"$V_{1/2}$ (mV)")        
                axb[i].legend(framealpha=0.5)
                        
            except:
                axb[i].plot(df_pars.iloc[:,1], df_pars.iloc[:,2],
                            marker='o', markersize=5, ls='none')
                
                axb[i].axhline(df_pars.iloc[:,2].mean(), 
                            c='white', ls='--', alpha=0.5)
                axb[i].axvline(df_pars.iloc[:,1].mean(), 
                            c='white', ls='--', alpha=0.5)
                
                axb[i].set_xlabel(r"$s$ (mV)")
                axb[i].set_ylabel(r"$c$")
        
        axb[0].set_ylabel(r"$s$ (mV)")
        axb[1].set_ylabel(r"$c$")
        
        if isinstance(title, str):
            axb[1].set_title(title + " Boltz. Fit Parameters")
        else:
            axb[1].set_title("Boltz. Fit Parameters")
            
        plt.tight_layout()
        saver(name="boltz_fit_params")
        
        #activation curve
        f_ac, ax = plt.subplots(figsize=(10,6))

        # plot average with SEM errorbars 
        ac_mu, ac_sem = get_mu_sem(out_files["act_norm"])
        ax.errorbar(ac_mu.index.values, 
                ac_mu, ac_sem, 
                ls='none', alpha=0.5,
                marker='o', markersize=5, 
                c='white', capsize=5, 
                label="Avg.")

        ax.plot(vrange, mu_boltz, 
                lw=1, ls='--', 
                c='r', label="\nVh=%.1f mV\ns=%.1f mV\nc=%.2f" % 
                (mu_params[0], mu_params[1], mu_params[2])
                )
        
        ax.legend(loc='upper right')
        ax.set_xlabel("Voltage (mV)")        
        ax.set_ylabel("Normalized Conductance")
        
        if isinstance(title, str):
            ax.set_title(title + " `Steady-state` activation")
        else:
            ax.set_title("`Steady-state` activation")
        
        plt.tight_layout()
        saver(name="GV", df=[ac_mu, ac_sem, "GV"])
        
        #exp params and delay
        df_xp = out_files["exp_params"]
        # print(df_xp.columns)
        
        f_xp = plt.figure(constrained_layout=True, figsize=(13,8))
        gs = f_xp.add_gridspec(2, 2)
        ax1 = f_xp.add_subplot(gs[0,:])
        ax2 = f_xp.add_subplot(gs[1,0])
        ax3 = f_xp.add_subplot(gs[1,1])
        
        vrange = [int(x) for x in df_xp.columns[1:]] 
        df_exp1 = df.loc[1,:].iloc[:,1:] 
        df_exp2 = df.loc[2,:].iloc[:,1:] 
        df_exp3 = df.loc[3,:].iloc[:,1:].dropna(axis=1, how='all').dropna(axis=0, how='all')

        FastTau_exp2, SlowTau_exp2, AmpProp_exp2 = extract_exp_params(df_exp2, order=2, prop=False)
        FastTau_exp3, SlowTau_exp3, AmpProp_exp3 = extract_exp_params(df_exp3, order=3, prop=False)

        N = df_exp1.shape[1] 
        taus_exp1 = [[1000/float(x[1]) for x in df_exp1.iloc[:,i].dropna().values if float(x[1]) > 0] for i in range(N)]
        taus_exp1 = pd.DataFrame.from_records(taus_exp1)
        taus_exp1.index = vrange 
        
        taus_exp1_mu, taus_exp1_err = get_mu_sem(taus_exp1)
        ax1.errorbar(vrange, taus_exp1_mu, taus_exp1_err, 
                     marker='o', markersize=5, c='white', 
                     capsize=5, label=r"$1^o, \tau$")

        # compute mean and SEM for fast and slow taus, then plot 
        SlowTau_exp2_mu, SlowTau_exp2_err = get_mu_sem(SlowTau_exp2)
        SlowTau_exp3_mu, SlowTau_exp3_err = get_mu_sem(SlowTau_exp3)
        ax1.errorbar(vrange, SlowTau_exp2_mu, SlowTau_exp2_err, 
                    marker='s', markersize=6, 
                    c='lightblue', ls='--', alpha=0.3, 
                    fillstyle='none', capsize=5, 
                    label=r"$2^o, \tau_s$")
        ax1.errorbar(vrange, SlowTau_exp3_mu, SlowTau_exp3_err, 
                    marker='^', markersize=8, 
                    c='r', ls='--', alpha=0.3, 
                    fillstyle='none', capsize=5, 
                    label=r"$3^o, \tau_s$")
        
        # if max tau of exp2 fast tau is > twice that of exp1 tau, then plot on twin axes 
        FastTau_exp2_mu, FastTau_exp2_err = get_mu_sem(FastTau_exp2)
        FastTau_exp3_mu, FastTau_exp3_err = get_mu_sem(FastTau_exp3)
        
        if max(FastTau_exp2_mu) >= 2*max(taus_exp1_mu):
            ax1_twin = ax1.twinx() 
            ax1_twin.errorbar(vrange, FastTau_exp2_mu, 
                            FastTau_exp2_err, 
                            marker='s', markersize=6, 
                            c='lightblue', capsize=5, alpha=0.3, 
                            label=r"$2^o, \tau_f$")
            ax1_twin.errorbar(vrange, FastTau_exp3_mu, 
                            FastTau_exp3_err, 
                            marker='^', markersize=8, 
                            c='r', capsize=5, alpha=0.3, 
                            label=r"$3^o, \tau_f$")
            
            ax1_twin.set_ylabel(r"$2^o$ and $3^o$, 1/$\tau_f$ ($s^{-1}$)")
            
            # get legend and handles for previous data, then add label and handle for twin axes 
            h, l = ax1.get_legend_handles_labels()
            h_twin, l_twin = ax1_twin.get_legend_handles_labels() 
            h = h + h_twin 
            l = l + l_twin 
            
        else:
            ax1.errorbar(vrange, FastTau_exp2_mu, 
                        FastTau_exp2_err, 
                        marker='s', markersize=6, 
                        c='lightblue', capsize=5, alpha=0.3, 
                        label=r"$2^o, \tau_f$")
            ax1.errorbar(vrange, FastTau_exp3_mu, 
                        FastTau_exp3_err, 
                        marker='^', markersize=8, 
                        c='r', capsize=5, alpha=0.3, 
                        label=r"$3^o, \tau_f$")
        
            h, l = ax1.get_legend_handles_labels()
        
        by_label = OrderedDict(zip(l, h))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper center')

        # plot ratio of component amplitudes for exp2 fits 
        AmpProp_exp2_mu, AmpProp_exp2_err = get_mu_sem(AmpProp_exp2)
        # AmpProp_exp3_mu, AmpProp_exp3_err = get_mu_sem(AmpProp_exp3)
        ax2.errorbar(vrange, AmpProp_exp2_mu, AmpProp_exp2_err, 
                    marker='s', markersize=5, lw=1,
                    c='lightblue', capsize=5)
        # ax2.errorbar(vrange, AmpProp_exp3_mu, AmpProp_exp3_err, 
                    # marker='^', markersize=6, 
                    # c='r', alpha=0.5, capsize=5)
        
        # plot delay 
        df_delay = out_files["exp_delay"] 
        Delay_mu, Delay_err = get_mu_sem(df_delay)
        ax3.errorbar(vrange, Delay_mu, Delay_err, 
                    marker='o', markersize=5, lw=1,
                    c='white', capsize=5)
        
        ax1.set_xlabel("Voltage (mV)")
        ax2.set_xlabel("Voltage (mV)")
        ax3.set_xlabel("Voltage (mV)")
        
        ax1.set_ylabel(r"$1/\tau$ ($s^{-1}$)") 
        # ax2.set_ylabel(r"$A_f$ / ($A_f + A_s$)")
        ax2.set_ylabel(r"$A_f$ / $A_s$")
        ax3.set_ylabel("Delay (ms)")      

        if isinstance(title, str):
            ax1.set_title(title + " Exponential kinetics")
        else:
            ax1.set_title("Exponential kinetics")
        
        saver(name="exp", df=[taus_exp1_mu, taus_exp1_err,
                            SlowTau_exp2_mu, SlowTau_exp2_err,
                            FastTau_exp2_mu, FastTau_exp2_err,
                            AmpProp_exp2_mu, AmpProp_exp2_err,
                            Delay_mu, Delay_err,
                            "Tau_exp1", "SlowTau_exp2", "FastTau_exp2", "AmpProp_exp2", "Delay_exp1"])
        