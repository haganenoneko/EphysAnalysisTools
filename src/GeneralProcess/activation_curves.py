# fitting activation curves

import numpy as np
import pandas as pd
import math
from GeneralProcess.EphysInfoFilter import EphysInfoFiltering, FindPairedFiles

import lmfit
from scipy.signal import savgol_filter
from scipy.stats import spearmanr, pearsonr, sem

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages

# cmap = plt.cm.get_cmap()
cmap = plt.cm.get_cmap("Set1")
plt.style.use("dark_background")
# plt.style.use("seaborn-colorblind")


def apply_savgol(df, k=3, w=100, show=True):
    """Apply savgol filter to `df` with degree `k` and window `w`"""
    df_filt = df.copy()

    print("Savgol filter with w = %d" % w)

    # window for fitting polynomials in savgol filter
    w = math.floor(df_filt.shape[0]/w)

    # ensure window is an odd integer
    w = w if (w % 2 > 0) else w + 1

    for i in range(df.shape[1]):
        df_filt.iloc[:, i] = savgol_filter(df_filt.iloc[:, i], w, k)

    if show:
        plt.plot(df, alpha=0.2, c="w")
        plt.plot(df_filt, lw=2, c="r")
        plt.title("Order = %d, Window = %d" % (k, w))
        plt.show()

    return df_filt


def apply_savgol_spearmanr(df, frac=0.40, k=2, w=8, show=False):
    """
    Apply savgol filter to last fraction `frac` of `df`
    Savgol parameters: window `w` and polynomial order `k`
    `show` = plot filtered data on top of original data, coloured by spearman r 

    Returns Spearman coefficients for each trace in a N-sized array 
    """
    # select last `frac`-th of `df` for filtering
    ind = int(frac*df.shape[0])
    df_filt = df.iloc[-ind:, :].copy()

    # window for fitting polynomials in savgol filter
    w = math.floor(df_filt.shape[0]/w)

    # ensure window is an odd integer
    w = w if (w % 2 > 0) else w + 1

    # apply savgol fitler
    if show:
        # plt.style.use("default")
        plt.plot(df_filt, alpha=0.2)
        cmap = plt.cm.get_cmap("seismic")

    for i in range(df.shape[1]):
        df_filt.iloc[:, i] = savgol_filter(df_filt.iloc[:, i], w, k)
        r, _ = spearmanr(df_filt.index, df_filt.iloc[:, i])

        if show:
            plt.plot(df_filt.iloc[:, i], c=cmap((r+1)/2), lw=2)

        df_filt.iat[0, i] = r

    if show:
        plt.show()
    plt.close()

    return df_filt.iloc[0, :].abs()


def Set_RC_Defaults(pub=False, dark=False):
    """
    Set rcParams 

    `pub` = if True, sets defaults for publication standards; else, whatever is convenient 
    """
    if dark:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    rcParams['axes.labelweight'] = 'bold'
    rcParams['axes.titleweight'] = 'bold'

    rcParams['axes.labelsize'] = 16
    rcParams['axes.titlesize'] = 18
    rcParams['legend.fontsize'] = 16

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.weight'] = 'normal'

    rcParams["axes.formatter.limits"] = (-3, 3)
    rcParams["ytick.minor.size"] = 6

    if pub:
        rcParams['font.size'] = 14
        rcParams['axes.linewidth'] = 2
        rcParams['font.sans-serif'] = 'Arial'
        rcParams['svg.fonttype'] = 'none'
        rcParams['pdf.use14corefonts'] = True

    else:
        rcParams['font.size'] = 12
        rcParams['axes.linewidth'] = 2
        rcParams['font.sans-serif'] = 'Verdana'


def SetXAxisSpacing(ax, dx=30, xmin=-145, xmax=60):
    xlo, xhi = ax.get_xlim()

    dx2 = int(dx/2)
    xlo = math.floor(xlo/dx2) * dx2
    xlo = xlo if (xlo > xmin) else xmin

    xhi = math.ceil(xhi/dx2) * dx2
    xhi = xhi if (xhi < xmax) else xmax

    ax.set_xticks(range(xlo, xhi+dx2, dx))


def SetAxProps(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    try:
        ax.locator_params(axis="y", nbins=6)
    except:
        pass

    ax.tick_params(axis='both', length=8, width=2, labelsize=12)


class BoltzmannFunctions():
    def __init__(self, func_name="boltz_a"):
        """
        Select Boltzmann function corresponding to `func_name`
        `func_name` options:
            'boltz_a' = standard Boltzmann 
            'boltz_c' = variable maximum 
            'boltz_d' = variable minimum (maximum = 1)
            'boltz_cd' = variable maximum and minimum
        """
        self.func_name = func_name
        self.descr = ""

    def boltz_a(self, v, vh, s):
        return 1/(1 + np.exp((v-vh)/s))

    # variable Pmax
    def boltz_c(self, v, vh, s, c):
        return (c / (1 + np.exp((v - vh)/s)))

    # variable Pmin
    def boltz_d(self, v, vh, s, d):
        return ((1-d)/(1 + np.exp((v-vh)/s))) + d

    # variable Pmin and Pmax
    def boltz_cd(self, v, vh, s, c, d):
        return ((c-d) / (1 + np.exp((v - vh)/s))) + d

    def select(self):
        ind = self.func_name.split("_")[1]

        if ind == "a":
            self.descr = "Standard Boltzmann (max = 1, min = 0)"
            return self.boltz_a
        elif ind == "c":
            self.descr = "Boltzmann with varying max (max = c > 0.5, min = 0)"
            return self.boltz_c
        elif ind == "d":
            self.descr = "Boltzmann with varying min (max = 1, min = d < 0.5)"
            return self.boltz_d
        elif ind == "cd":
            self.descr = "Boltzmann with varying min and max (max = c > 0.5, min = d < 0.5)"
            return self.boltz_cd


def get_vhalf(vh, s, c=1, d=0):
    return vh + s*math.log((c-0.5)/(0.5-d))


def get_gv_leg_lab(vh, s, c=1, d=0):

    if c < 0.5 or d > 0.5:
        print(c, d)
        raise Exception("c < 0.5 or d > 0.5")
    else:
        Vhalf = get_vhalf(vh, s, c=c, d=d)

    if c > 0 and d > 0:
        return "Fit \n $V_{1/2}$ = %.1f mV \n $s$=%.1f mV \n c=%.4f \n d=%.2e" % (Vhalf, s, c, d)

    elif c > 0:
        return "Fit \n $V_{1/2}$ = %.1f mV \n $s$=%.1f mV \n c=%.4f" % (Vhalf, s, c)

    elif d > 0:
        return "Fit \n $V_{1/2}$ = %.1f mV \n $s$=%.1f mV \n d=%.2e" % (Vhalf, s, c)

    else:
        return "Fit \n $V_{1/2}$ = %.1f mV \n $s$=%.1f mV" % (Vhalf, s)


def multi_sort(zipped):
    """
    Sort multiple lists in `zipped` according to first element 
    Returns sorted lists 
    """
    return map(list, zip(*sorted(zipped)))


class NoisyGV():
    def __init__(self, df, khz):
        self.df = df
        self.khz = khz

    def current_histogram(self, dt=500):
        N = int(self.df.shape[1]/2)
        dt = int(dt * self.khz)

        f = plt.figure(constrained_layout=True)
        gs = f.add_gridspec(N, 2)

        # currents
        axI = f.add_subplot(gs[:, 0])

        for i in range(N):
            clr = cmap((i+1)/N)
            axI.plot(self.df.iloc[:, i], c=clr)

            ax_i = f.add_subplot(gs[i, 1])
            ax_i.hist(self.df.iloc[:dt, i],
                      histtype="stepfilled", bins="auto", color=clr)

            ax_i.set_yticks([])
            ax_i.set_yticklabels([])

        plt.show()


class lmfit_boltzmann():
    def __init__(self, func_name="boltz_a"):
        """
        `v` = test voltages 
        `g` = normalized conductances 
        """

        # declare Boltzmann parameters
        B_pars = lmfit.Parameters()
        B_pars.add("Vh", value=-100, min=-200, max=0)
        B_pars.add("s", value=10, min=3, max=50)
        self.pars = B_pars
        self.popt = None

        # variant of boltzmann function for fitting
        b = BoltzmannFunctions(func_name=func_name)
        print(b.descr)
        self.boltz = b.select()

    def do_fit(self, v, g, func_name="boltz_c", c_max=1.1):

        # add extra parameters if needed
        if "c" in func_name:
            # self.pars.add("c", value=1, min=0.51, max=c_max)
            self.pars.add("c", value=1, min=1., max=c_max)

        if "d" in func_name:
            self.pars.add("d", value=0, min=-0, max=0.49)

        # select boltzmann function
        b = BoltzmannFunctions(func_name)
        self.boltz = b.select()
        print("Fitting with: \n %s\
            Note that `lmfit` is used for fitting. \
            \n Parameters are unpacked in the residual by flattening dict values to a list, which may cause loss of order. \
            \n If errors are suspected, check if this is the case." % b.descr)

        # define residual
        def residual(pars, volts=v, data=g):
            # unpack parameters: extract .value attribute for each parameter
            parvals = list(pars.valuesdict().values())
            return self.boltz(volts, *parvals) - data

        res = lmfit.minimize(residual, self.pars, method='nelder')

        res.params.pretty_print()
        print(lmfit.fit_report(res))

        # prameters in dict format
        self.popt = res.params.valuesdict()

        try:
            pcov = res.covar
            perr = np.sqrt(np.diag(pcov))
        except:
            perr = [np.nan] * len(self.popt.keys())

        return self.popt, perr

    def get_fit(self, vrange, p0=[-100, 5]):
        popt = list(self.popt.values())

        if len(popt) > 0:
            return [self.boltz(v, *popt) for v in vrange]
        else:
            raise Exception("Fitting failed")


def subtract_tail_baseline(df, khz, window=50, filter_ma=10,
                           show=False, method='min'):
    """
    Subtract average of last `window` ms from each trace. 
    This is justified when tail currents are expected to reach steady-state at the tail voltage. 

    `df` = dataframe containing extracted tail currents
    `khz` = sampling rate 
    `window` = time window, beginning from the end of the pulse, where currents will be averaged and subtracted 
    `filter_ma` = if > 0, uses this as timerange to compute moving average 
    `method` = 'all' or 'min' 
        * if 'all', subtracts average current in `window` for each tail current; 
        * if 'min', subtracts average current in `window` for tail current with least standard deviation from all tail currents 
        * if 'auto', begins with 'all', but switches to 'min' if any Spearman R^2 has absolute value > 0.6 for last 40% of tail currents. 

    Returns
    `df_sub` = subtracted dataframe 
    """

    if window > df.shape[0]/khz:
        raise Exception("`window` cannot be larger than tail duration.")

    if method not in ['all', 'min', 'auto']:
        raise Exception(
            "`method` for `subtract_tail_baseline` must be 'min', 'auto', or 'all'")

    if filter_ma > 0:
        print("Appling < %d ms > rolling average..." % filter_ma)
        df = df.rolling(filter_ma*khz).mean().iloc[filter_ma*khz:, :]
        # reset time index to start at 0
        df.index -= filter_ma

    # test linearity of last 40% of tail currents by Spearman R
    print("Spearman coeff. (R) for last 40% of tail currents")
    r = apply_savgol_spearmanr(df)
    print("Absolute Spearman r: \n", r)
    print("Average: ", r.mean())

    # convert to time units, then add small offset of 10ms to avoid capacitance
    window = (window + 10)*khz

    # slice `df` by `window` (+ 10ms offset) with 5ms rolling avg
    i_ss = df.iloc[-window:-10*khz, :]

    # skip subtraction if average current in `window` is less than 1pA
    if (i_ss.rolling(5*khz).mean().mean(axis=0).abs() < 1).all():
        print("Mean current in `window` of tail currents: \n", i_ss.mean(axis=0))
        print("All mean currents < 1pA. Skipping baseline subtraction.")
        print(i_ss.mean(axis=0))
        return df

    # check that currents are relatively unchanging in `i_ss` by computing (max - min)
    def max_minus_min(x): return x.max(axis=0) - x.min(axis=0)

    # current drop across `window` and the entire range `whole`
    drops = [max_minus_min(y.rolling(2*khz).mean().abs()) for y in [i_ss, df]]
    drops = 100 * (drops[0] / drops[1].max())
    # range_ss = max_minus_min(i_ss.rolling(2*khz).mean().abs())
    # range_whole = max_minus_min(df.rolling(2*khz).mean().abs()).max()

    # ideally, the change in current over the subtracted region is
    # <= 1% the total current drop
    print("Current drop in `window`as % of total current drop. \n", drops)

    # try to identify noisy traces where current drop in `window` is
    # substantial compared to total current drop throughout tail current
    if (drops >= 3).all():
        print("All traces have >3% current drop in `window`. Applying savgol filter.")
        # degree down to 60
        w = max([50, 400 - 40*int(drops.max())])
        df = apply_savgol(df, w=w, show=True)

    if method == 'min':
        # use standard deviation to find tail current with
        # least change in current amplitude
        # idx = df.iloc[window:-window, :].abs().std(axis=0).argmin()
        # i_ss = i_ss.iloc[:, idx]

        if df.iat[0, 0] > df.iat[0, -1]:
            i_ss = i_ss.iloc[:, -1]
        else:
            i_ss = i_ss.iloc[:, 0]

    # compute baseline
    # if method = 'min', this is scalar
    # else, 1 x N
    i_ss = i_ss.mean(axis=0)

    # test for steady-state using Spearman coefficient
    if (r > 0.55).any():
        print("Spearman coefficients > 0.55 found.")

        if method == "auto":
            print("For traces with `r > 0.55`, replace with mean of other traces")
            inds = r[r > 0.55].index.values
            i_ss.iloc[inds] = i_ss.iloc[~inds].min()

        else:
            print(
                "If baseline subtraction should be modified, consider setting `method = 'auto'`")

    if show:
        fig, axs = plt.subplots(1, 2, figsize=(9, 5), constrained_layout=True)
        axs[0].set_title("Subtraction of baseline")
        axs[1].set_title("Division by maximum")

        axs[0].plot(df.iloc[::10, :], c='gray', lw=1, alpha=0.5)

    # subtract baseline
    df = df - i_ss

    if show:
        # decrement in alpha
        da = 1/df.shape[1]

        # plot subtracted currents and (crudely) normalized currents
        imax = df.max(axis=0).max()
        for i in range(df.shape[1]):
            alpha = 1 - da*i
            axs[0].plot(df.iloc[::10, i], c='r', lw=2, alpha=alpha)

            # plot a crude normalization, i.e. divide by maximum
            axs[1].plot(
                (df.iloc[::10, i] / imax),
                c="purple", lw=2, alpha=alpha
            )

        # demarcate window for subtraction
        axs[0].axvline(df.index[-window], c='r', ls='--', lw=2, alpha=0.5)
        axs[0].axvline(df.index[-10*khz], c='r', ls='--', lw=2, alpha=0.5)

        fig.suptitle("Average range_ss/range_whole = %.2f %%" % drops.mean())

        plt.show()
        plt.close()

    return df


class activation_curve():
    def __init__(self, test_voltages, tails, khz,
                 boltz_func="boltz_cd",
                 subtract_baseline_method="all",
                 base_method="zero_negative_norm",
                 post_tails=None, post_tail_voltage=0,
                 fit_only=False, show_pmin=False, plot_results=False,
                 show=True, pdf=None):
        """
        Compute activation curve/Pmin from leak-subtracted data 

        INPUTS
        `test_voltages` = list of test voltages 
        `tails` = dataframe of tail currents

        `subtract_baseline_method` = method argument for `subtract_baseline_method()`
        `base_method` = how to set values of baseline. Unlike `subtract_baseline_method`, which is applied prior to any processing, this is intended to handle stray values *after* initial normalization. 
            Default is 'leak', which uses the leak-subtracted current. Other options:
            * 'zero_negative_norm' = subtract most negative value to zero, then re-normalize. No changes if all values are greater or equal to 0.
            * 'zero_negative' = set negative values to zero without re-normalization. Again, no changes if all values are greater or equal to 0.
            * 'pmin_threshold=#', where # is a number; all values >= # will be set to 0, without re-normalization.
            * 'pmin_threshold_norm=#', subtract minimum of values >= #, then renormalize.

            See docstring of `apply_base_method` for more details.

        `nparam` = determines number of free parameters in boltzmann fit 
            'boltz_a' for standard boltzmann, 'boltz_c' for floating maximum, 'boltz_d' for floating minimum, or 'boltz_cd' for floating maximum + minimum 

        `post_tails` = dataframe of post-tail currents, if available
        `post_tail_voltage` = voltage used to elicit post_tails 
        `show_pmin` = compute Pmin for tail and/or post_tail currents

        For deactivation, calling the class will simply return Pmin estimates from tail currents, and, when available, post-tail currents

        For activation, calling the class returns normalized Po and Boltzmann parameters
        """

        # initialize class variables as None
        # if they change, this informs downstream methods about the protocol used
        self.norm_tails = None
        self.tail_mins = None
        self.post_peaks = None

        # general
        self.test_voltages = None
        self.popt = None

        self.nparam = boltz_func

        if fit_only:
            self.test_voltages = test_voltages
            self.norm_tails = tails
        else:
            # correct baseline for activation protocols (GV)
            # subtract baseline current from tail currents if average current in last 50ms is greater than 10
            if post_tails is None:
                tails = subtract_tail_baseline(
                    tails, khz, window=200, show=show,
                    method=subtract_baseline_method
                )

            # manually truncate a problematic trace
            # if abs(tails.iloc[:,2].sum() + 66710) < 5:
            #     tails.iloc[:,2] = tails.iloc[:1850*khz,2]
            #     print("Warning: Manual truncation of 3rd sweep's tail current during `activation_curve`.")

            # compute peak amplitude of tail currents
            # peak current from 4ms moving avg over first 100ms,
            # with 5ms offset to avoid residual capacitance
            tail_peaks = tails.iloc[5*khz:105*khz,
                                    :].rolling(5*khz).mean().abs().max().values

            # deactivation, i.e. post_tails isn't None
            if post_tails is not None:

                # for deactivation, compute pmin from tails, as well as post_tail pmin, if available
                norm_tails = pd.concat(
                    [tails.iloc[:, i].abs()/tail_peaks[i]
                     for i in range(tails.shape[1])],
                    axis=1
                )

                # Pmin from tail currents
                # avg of 5ms moving avg of last 50ms, offset by -2ms to help avoid capacitive currents
                tail_mins = norm_tails.dropna(
                ).iloc[-52*khz:-2*khz, :].rolling(5*khz).mean().mean()
                self.tail_mins = tail_mins

                # check if post_tail_voltage is in test_voltages; if not, ignore post_tails
                if post_tail_voltage in test_voltages:
                    # index of sweep with test pulse voltage corresponding to post_tail_voltage
                    j = test_voltages.index(post_tail_voltage)
                    # divide post_tail currents by peak current from (test) tail current at the same voltage
                    norm_post = post_tails / tail_peaks[j]

                    # normalized peak from post_tails
                    # this measures residual activation after deactivating test pulses, hence `Pmin`
                    post_peaks = norm_post.iloc[:100 *
                                                khz, :].rolling(4*khz).mean().max()
                    self.post_peaks = post_peaks

                # sort all data according to voltages
                if self.post_peaks is None:
                    self.test_voltages, self.tail_mins = multi_sort(
                        zip(test_voltages, tail_mins))
                else:
                    self.test_voltages, self.tail_mins, self.post_peaks = multi_sort(
                        zip(test_voltages, tail_mins, post_peaks))

                # `base_method` != 'leak' would normally modify Pmin values for deactivation, but \
                # because a Popen is unavailable at the duration used for deactivating tail currents,
                # this is probably unwise.
                # if base_method != "leak":
                #     self.tail_mins, self.tail_peaks, self.post_peaks = self.apply_base_method(base_method)

                if show_pmin:
                    self.plotter(mode="deactivation", pdf=pdf)

            # activation
            else:
                # ignore tail currents following activating pulses below -150mV or above 100mV
                # indices out of the range [-150, 100]
                V_InRange = [i for (i, v) in enumerate(
                    test_voltages) if abs(v + 25) <= 125]

                # only keep test voltages that are in range
                if len(V_InRange) > 0:
                    test_voltages = [t for i, t in enumerate(
                        test_voltages) if i in V_InRange]
                    tail_peaks = [t for i, t in enumerate(
                        tail_peaks) if i in V_InRange]

                    # dataframe of tail currents
                    tails = tails.iloc[:, V_InRange]

                else:
                    raise Exception(
                        " No voltages were found in range [-150, 100].")
                    exit()

                # normalize all tail currents by maximum of the entire data
                M = np.max(tail_peaks)
                norm_tails = tails / M

                # find minima of normalized tails as avg of 5ms moving avg of last 50ms, offset by 2ms to help avoid capacitance
                tail_mins = norm_tails.iloc[-52*khz:-2 *
                                            khz, :].rolling(5*khz).mean().mean(axis=0)
                # normalize peaks by maximum peak
                tail_peaks = [t/M for t in tail_peaks]

                # sort all data according to voltages
                test_voltages, tail_peaks, tail_mins = multi_sort(
                    zip(test_voltages, tail_peaks, tail_mins))

                self.test_voltages = test_voltages
                self.norm_tails = tail_peaks
                self.tail_mins = tail_mins

                # apply baseline method
                # variables that are None will remain as such, eg self.post_peaks for activation
                if base_method != "leak":
                    self.tail_mins, self.norm_tails, self.post_peaks = self.apply_base_method(
                        base_method)

                # vrange, y, popt = sim
                sim = self.fit_boltz(return_sim=True)
                self.popt = sim[2]

                if show:
                    self.plotter(mode="activation", sim=sim, pdf=pdf)

    def apply_base_method(self, base_method):
        """
        `base_method` = how to set values of baseline. Default is 'leak', which simply uses the leak-subtracted current. Other options:
            * 'zero_negative' = set negative values to zero without re-normalization
            * 'zero_negative_norm' = subtract minimum of negative values, then re-normalize
            * 'pmin_threshold=#', where # is a number; all values < # will be set to 0, without re-normalization.
            * 'pmin_threshold_norm=#', subtract minimum of values < #, then renormalize.
            * 'v_threshold=#', where # is a voltage; values for voltages > # will be set to 0
        """
        # list of relevant quantities for steady-state activation
        P = [self.tail_mins, self.norm_tails, self.post_peaks]

        if all(p is None for p in P):
            raise Exception(
                "   All of `tail_mins`, `tail_peaks` and `post_peaks` are None.")
        else:
            print(" Treating baseline with method < %s >" % base_method)
            for i, p in enumerate(P):

                # skip if None, e.g. post_peaks in activation
                if p is None:
                    continue

                elif base_method in ["zero_negative", "zero_negative_norm", "pmin_threshold", "pmin_threshold_norm", "leak"]:
                    if base_method == "leak":
                        continue

                    elif "zero_negative" in base_method:
                        if any(x < 0 for x in p):
                            pmin = np.nanmin(p)
                            P[i] = [x - pmin for x in p]

                    elif "pmin_threshold" in base_method:
                        # threshold as float
                        pmin_t = float(base_method.split("=")[1])

                        # find values < pmin_t
                        if any(x < pmin_t for x in p):
                            for j in range(len(p)):
                                if p[j] < pmin_t:
                                    P[i][j] = 0

                    elif 'v_threshold' in base_method:
                        # threshold voltage as float
                        v_t = float(base_method.split("=")[1])

                        # check if any voltages above v_t
                        if any(x > v_t for x in self.test_voltages):

                            # assume data are sorted according to voltage (negative to positive)
                            # index of first element in `v` that satisfies v[i] > v_t
                            idx = next(i for i, x in enumerate(
                                self.test_voltages) if x > v_t)

                            for j in range(len(p) - idx):
                                P[i][idx+j] = 0

                    # re-normalize
                    if "norm" in base_method:
                        p = P[i][:]

                        # select `pmax` from `self.norm_tails` if available
                        # i.e. to avoid normalizing Pmin to Pmin and getting a Pmin of 1
                        if self.norm_tails is None:
                            print(
                                " `self.norm_tails` is None, but `base_method` with `norm` implies re-normalization. \n Re-normalization will therefore produce a [0, 1] scale for the given dataset.")
                            pmax = np.nanmax(p)

                        else:
                            pmax = np.nanmax(self.norm_tails)

                        P[i] = [x/pmax for x in p]

                else:
                    raise Exception(
                        "   `base_method` %s not understood." % base_method)

            return P

    def plotter(self, mode="activation", sim=None, show=True, pdf=None):
        """
        Plot maximum and minimum values from normalized tail currents (`mode=activation`) or normalized minimum values from tail currents and normalized maximum values from post-tail currents (`mode=deactivation`)

        `sim` = output from `self.fit_boltz(return_sim=True)` 
        """
        volts = self.test_voltages

        if mode == "deactivation":
            f, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
            ax.plot(volts, self.tail_mins, marker='o', lw=0.5, label="Tail")

            if self.post_peaks is not None:
                ax.plot(volts, self.post_peaks, marker='s',
                        lw=0.5, label="Post-tail")

            ax.legend()

            ax.set_xlabel("Voltage (mV)")
            ax.set_ylabel(r"$P_{\mathregular{min}}$")
            ax.set_title(
                r"Average $P_{\mathregular{min}}$ = %.3f" % np.mean(self.tail_mins))

        elif mode == "activation":

            if sim is None:
                raise Exception(
                    "   No information from Boltzmann fitting provided.")
            else:
                vrange, y, popt = sim
                popt = popt.values()

            f, ax = plt.subplots(1, 2, figsize=(12, 5))

            ax[0].plot(volts, self.norm_tails,
                       marker='o', lw=0.5, label="Data")

            # create legend label with fit parameters
            n = len(popt)
            # set defaults
            c, d = [0, 0]

            # unpack fit parameters
            if n == 4:
                vh, s, c, d = popt
            elif n == 3:
                if "d" in self.nparam:
                    vh, s, d = popt
                else:
                    vh, s, c = popt
            else:
                vh, s = popt

            # create legend label
            lab = get_gv_leg_lab(vh, s, c=c, d=d)

            ax[0].plot(vrange, y, ls='--', label=lab)
            ax[1].plot(volts, self.tail_mins, marker='o', lw=0.5)

            # location of legend label
            if y[0] > 0.5:
                ax[0].legend(loc='lower left', fontsize=10,
                             framealpha=0.5, edgecolor="none")
            else:
                ax[0].legend(loc='upper right', fontsize=10,
                             framealpha=0.5, edgecolor="none")

            ax[0].set_title("Normalized Conductance")
            ax[1].set_title(r"$P_{min}$")

            ax[0].set_ylabel(r"$I/I_{max}$")
            for i in range(2):
                ax[i].set_xlabel("Voltage (mV)")

            # yticks
            ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

        if pdf is not None:
            pdf.savefig()

        if show:
            plt.show()

        plt.close()

    def fit_boltz(
        self, return_sim=False, vrange=list(range(-200, 10, 5)), fit_kw={}
    ):
        """
        Perform fit with Boltzmann function (standard if `self.nparam = 2`, modified if `self.nparam = 3`).

        `c_max` = upper bound for `c` in Boltzmann fit 
        `vrange` = voltages to evaluate Boltzmann at, only used if `return_sim=True`

        If `return_sim = True`, return 
            voltages `vrange` = list 
            Boltzmann output `y` = list 
            Boltzmann fit parameters `popt` = OrderedDict
        Else, return Boltzmann fit parameters `popt`.
        """
        # sort voltages and normalized tail peaks
        try:
            v, g = zip(*sorted(zip(self.test_voltages, self.norm_tails)))
        except:
            v, g = self.test_voltages, self.norm_tails

        # fit boltzmann
        LM = lmfit_boltzmann()

        popt, _ = LM.do_fit(v, g, func_name=self.nparam, **fit_kw)

        self.popt = popt

        if return_sim:
            y = LM.get_fit(vrange)
            return vrange, y, popt
        else:
            return popt

    def do_fit(self):
        """
        Return normalized peak tail amplitudes and fit parameters 
        """
        if self.popt is None:
            raise Exception(
                "\n No fit parameters available to return. Fit wasn't done first, or fit failed.")
        else:
            return self.norm_tails, self.popt

    def tail_pmins(self):
        """
        Return normalized minimum tail currents and, if available, normalized post tail peaks (for deactivation protocol). 
        """
        if self.post_peaks is None:
            return self.tail_mins
        else:
            return self.tail_mins, self.post_peaks

    def return_test_voltages(self):
        """
        Return test voltages used in recording protocol (not necessarily the same as voltages used for simulating Boltzmann output).
        """
        return self.test_voltages


class Summarize_GV():
    def __init__(self, fname, dv=3.178, paired_dic={},
                 individual=False, vsim=range(-200, 10, 5),
                 path=r"./output/Processing/Pooled_Analyses/CSV_output/"):
        """
        `fname` is the filename of a .csv file in the `CSV_output` folder containing voltages and normalized conductances. `fname` can also be a list of Strings. If so, all dataframes will be concatenated. 
        `dv` = liquid junction potential, subtracted from test_voltages. Calculated using LJPCalc
            https://swharden.com/software/LJPcalc/theory/#ljpcalc-calculation-method
        `paired_dic` = dictionary of {parent : child} filenames. Otherwise, will retrieve automatically.
        `individual` = whether to treat List of Strings in `fname` as individual dataframes; if False, then the data are concatenated and analyzed together
        `vsim` = voltages to simulate Boltzmann 

        Computes statistics and make figure.
        """

        plt.style.use("dark_background")
        Set_RC_Defaults()

        # initialize self.df and self.df_pars, which will hold normalized conductances and boltzmann fit parmaeters, respesctively
        self.df = None
        self.df_pars = None

        # read dataframe of normalized conductances vs voltage;
        # paths may specify single String or List of Strings
        if isinstance(fname, list):
            # concatenate all dataframes column-wise
            df_list = [pd.read_csv(path + f, header=0, index_col=0)
                       for f in fname]

            # find parameters csv, which should have same filename except for suffix "boltz_params"
            pre = [f.split("__")[:-1] for f in fname]
            for i, f in enumerate(pre):
                f.append("boltz_params.csv")
                pre[i] = "__".join(f)
            df_pars_list = [pd.read_csv(
                path + f, header=0, index_col=0) for f in pre]

            # if 'c' is missing from columns, add it and set it to all 0
            for i, d in enumerate(df_pars_list):
                if d.shape[1] == 3:
                    df_pars_list[i].columns = ["Vh", "s", "c"]
                elif d.shape[1] == 2:
                    d["c"] = np.zeros(d.shape[0])
                    d.columns = ["Vh", "s", "c"]
                    df_pars_list[i] = d

            if individual:
                self.df = df_list
                self.df_pars = df_pars_list
            else:
                # concatenate dataframes
                df = pd.concat(df_list, axis=1)
                # remove duplicate columns (filenames)
                self.df = df.loc[:, ~df.columns.duplicated()]

                # concatenate along rows
                df_pars = pd.concat(df_pars_list, axis=0)
                # remove duplicate rows (filenames)
                self.df_pars = df_pars.loc[~df_pars.index.duplicated(), :]
        else:
            self.df = pd.read_csv(path + fname, header=0, index_col=0)

        if (self.df is None) or (self.df_pars is None):
            raise Exception(
                "Failed to parse data as`df` and `df_pars` from `fname`.")

        self.fname = fname
        self.dv = dv
        self.paired_dic = paired_dic
        self.vsim = vsim
        self.outpath = path
        self.refit_params = None

    def do_stats(self, fname, df):
        """
        Returns mean and error of G-V data in a single dataframe `df` with fname `fname`

        First finds paired recordings to identify technical replicates. 
        Technical replicates are averaged together, then averaged with remaining biological replicates. 
        Errors are given as stdev if technical replicates are present. Else, uses sem. 
        If technical replicates are present, variances between these and biological replicates are averaged, then sqrt to find stdev. 
        """
        # find paired files
        F = []
        if len(self.paired_dic.keys()) < 1:
            try:
                F = FindPairedFiles(fname).Find()

                # flatten dictionary, since we don't need the parent/child distinction
                F = list(F.values())

            except:
                print(
                    " Finding paired files failed. \n Continuing, assuming all replicates are biological replicates.")

        else:
            # check if parent filename is in dictionary value; if so, we can just flatten the dictionary
            k = self.paired_dic.keys()[0]
            v = self.paired_dic[k]
            if k in v:
                F = list(self.paired_dic.values())

            # otherwise, add parent filename to values, then flatten dictionary afterwards
            else:
                for k, v in self.paired_dic.items():
                    if k in v:
                        continue
                    else:
                        v.append(k)
                F = list(self.paired_dic.values())

        # if technical replicates are present, then `F` is not empty
        paired_mu = []
        paired_err = []

        if len(F) > 0:
            for f in F:
                # columns of dataframe that correspond to cell `f`, one biological replicate
                df_f = df.loc[:, df.columns.isin(f)].dropna(how='all')

                if df_f.shape[0] < 2:
                    continue
                else:
                    # compute mean and std over columns
                    paired_mu.append(df_f.mean(axis=1))
                    paired_err.append(df_f.std(axis=1))

                    # drop the paired files from original dataframe
                    df.drop(df_f.columns, axis=1, inplace=True)

        # mean and sem over biological replicates (if no technical replicates, then this is performed over entirety of `df`)
        mu = df.mean(axis=1)
        # if no technical replicates, use sem instead of std
        if len(paired_err) < 1:
            err = df.sem(axis=1)
        else:
            err = df.std(axis=1)

        # if technical presents are present, average with mean and variances of unpaired biological replicates
        if len(paired_mu) > 0:
            # concatenate statistics of paired files
            paired_mu = pd.concat(paired_mu, axis=1, ignore_index=False)
            paired_err = pd.concat(paired_err, axis=1, ignore_index=False)

            # compute total averages and sem
            mu = pd.concat([mu, paired_mu], axis=1,
                           ignore_index=False).mean(axis=1)
            # std**2 -> variance -> average variance -> sqrt(variance) = std
            err = pd.concat([err, paired_err], axis=1, ignore_index=False).pow(
                2).mean(axis=1).pow(0.5)

        return mu, err

    def boltzfit(self, df, mu, LJPcorrection=True):
        """
        Use lmfit to fit Boltzmann function to G-V data, with voltages in index of `df` and conductances in `mu` \\
        `df` = dataframe of voltage x normalized conductance 
        `mu` = average of conductances in `df` 
        `LJPcorrection` = whether to apply offset due to liquid junction potential \\

        Returns `yfit`, `popt`, `perr`
        """
        # adjust voltage, V_cell = V_measured - V_LJP
        if LJPcorrection:
            volts = df.index - self.dv
        else:
            volts = df.index.values

        # fit boltzmann with lmfit
        LM = lmfit_boltzmann()
        popt, perr = LM.do_fit(volts, mu, func_name="boltz_cd")
        yfit = LM.get_fit(self.vsim)

        return yfit, volts, popt, perr

    def MeanBoltzFit(self, df, LJPcorrection=True, median=False):
        """
        Average fit parameters in `df`, then compute a Boltzmann curve  
        """
        df_pars = df.copy()

        if (LJPcorrection == True) and (self.dv > 0):
            df_pars.loc[:, "Vh"] -= self.dv

        # compute SEM
        err = df_pars.sem(axis=0)

        # compute median or mean
        if median:
            mu = df_pars.median(axis=0)
        else:
            mu = df_pars.mean(axis=0)

        if len(mu.shape) > 1:
            mu = pd.Series(mu)

        if mu.shape[0] < 3:
            func_name = "boltz_a"
        elif mu.shape[0] > 3:
            func_name = "boltz_cd"
        else:
            if "c" in mu.index:
                func_name = "boltz_c"
            else:
                func_name = "boltz_d"

        # instantiate LM object, specify Boltzmann function to use
        LM = lmfit_boltzmann(func_name=func_name)

        # use columns of `mu` as parameter names to create dictionary of parameters
        LM.popt = mu.to_dict()
        yfit = LM.get_fit(self.vsim)
        return yfit, mu, err

    def create_leg_labels(self, popt: dict, perr=[], f=""):
        if len(f) > 0:
            lab = [f]
        else:
            lab = []

        if len(perr) > 0:
            for i, tup in enumerate(popt.items()):
                k, v = tup
                if k == "Vh":
                    u = r"$V_h$ = %.1f $\pm$ %.1f mV" % (popt[k], perr[i])
                else:
                    if k == "c":
                        u = r"$P_{max}$ = "
                        u += "%.3f $\pm$ %.3f" % (popt[k], perr[i])

                    elif k == "d":
                        u = r"$P_{min}$ = %.3f $\pm$ %.3f" % (popt[k], perr[i])

                    else:
                        u = r"%s = %.2f $\pm$ %.2f mV" % (k, popt[k], perr[i])

                lab.append(u)
        else:
            for i, tup in enumerate(popt.items()):
                k, v = tup
                if k == "Vh":
                    u = r"$V_h$ = %.1f mV" % popt[k]
                else:
                    if k == "c":
                        u = r"$P_{max}$ - 1 = %.3f" % (popt[k]-1)
                    elif k == "d":
                        u = r"$P_{min}$ = %.3f" % (popt[k])
                    else:
                        u = r"%s = %.2f mV" % (k, popt[k])

                lab.append(u)

        # insert linebreak between each entry
        lab = "\n".join(lab)
        return lab

    def refit(self, func_name="boltz_cd", save_csv=False):
        """
        Refit each '__act_norm' dataset with `func_name` 
        """
        LM = lmfit_boltzmann()

        # each column of `df` is a different recording
        params = {}
        for i, df in enumerate(self.df):
            params_i = {}

            for j in range(df.shape[1]):
                y = df.iloc[:, j].dropna()
                popt, _ = LM.do_fit(
                    y.index.values, y.values, func_name=func_name)
                params_i.update({j: popt})

            fname = self.fname[i].split("__")[0]
            y = pd.DataFrame.from_dict(params_i).T
            params.update({fname: y})

            if save_csv:
                # path to save csv file
                p = self.outpath + "%s__refit_%s.csv" % (
                    fname, func_name.split("_")[1]
                )

                y.to_csv(p)

        if save_csv:
            print("Successfully saved refit parameters to \n < %s >" % p)

        self.refit_params = params
        print("Refit each dataset successfully. Access refit params at `self.refit_params`")

    def go(
        self, refit=False, fname_as_label=100, clrs={},
        MeanDataBoltz=False, save_csv=False, save_fig=False
    ):
        """
        Run stats, fit Boltzmann, and return figure

        `fname_as_label` = when multiple dataframes are being summarized, `fname_as_label` specifies the section of respective filenames to use as legend labels 
        e.g. if `fname_as_label` < 100 -> uses `fname_as_label` to index the output of `fname.split('__')` as legend label

        `clrs` = dictionary with file names as keys and colours as values 
            any file names encountered not contained in `clrs` are coloured by iterating over a cmap 

        `MeanDataBoltz` = whether to show a Boltzmann fit computed on averaged data, if 
        `self.df` is a list 

        `save_csv` = whether to fit mean/sem and fit output as csv 
        `save_fig` = whether to save figure
        """
        Set_RC_Defaults()

        fig, ax = plt.subplots(figsize=(7, 5))

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        if self.dv > 0:
            ax.set_xlabel("Voltage - LJP (mV)", fontsize=14)
        else:
            ax.set_xlabel("Voltage (mV)", fontsize=14)

        ax.set_ylabel("Normalized Conductance", fontsize=14)

        if isinstance(self.df, list):
            df_list = []
            df_colnames = []
            pars_list = {}

            for i, d in enumerate(self.df):

                mu, err = self.do_stats(self.fname[i], d)
                df_list.append(mu)
                df_list.append(err)
                df_colnames.append(self.fname[i])
                df_colnames.append(self.fname[i] + "__err")

                # set index of average/error GV to LJP-corrected voltages
                volts = d.index.values - self.dv
                mu.index = volts
                err.index = volts

                # legend label with Boltzmann parameters and data name
                lab = []

                if fname_as_label < 100:
                    # skip the last element, which describes type of output, e.g. 'act_norm.csv'
                    f = self.fname[i].split("__")[:-1]
                    if fname_as_label <= len(f):
                        lab.append(f[fname_as_label])
                    f = f[0]

                if isinstance(clrs, dict) and (f in clrs.keys()):
                    clr = clrs[f]
                else:
                    clr = cmap((i+1)/len(self.df))

                if MeanDataBoltz:
                    yfit, volts, popt, perr = self.boltzfit(d, mu)
                elif isinstance(self.refit_params, dict):
                    yfit, popt, perr = self.MeanBoltzFit(self.refit_params[f])
                else:
                    yfit, popt, perr = self.MeanBoltzFit(self.df_pars[i])

                pars_list.update({self.fname[i]: popt})
                pars_list.update({self.fname[i] + "__err": perr})
                lab = self.create_leg_labels(popt, perr=perr, f=f)

                # plot boltzmann fit
                ax.plot(self.vsim, yfit, ls='--', c=clr, lw=2, alpha=0.5)

                # plot markers with error bars
                ax.errorbar(
                    volts, mu, marker='o', yerr=err,
                    mfc=clr, mec='k', ms=8, ls='none',
                    ecolor=clr, elinewidth=1.5, capsize=4, label=lab
                )

            out = pd.concat(df_list, axis=1, ignore_index=False)
            out.columns = df_colnames

            out_pars = pd.DataFrame.from_dict(pars_list)
            if out_pars.shape[0] == 2:
                out_pars.index = ["Vh", "s"]
            else:
                out_pars.index = ["Vh", "s", "c"]

        else:
            mu, err = self.do_stats(self.fname, self.df)
            yfit, volts, popt, perr = self.boltzfit(self.df, mu)

            lab = self.create_leg_labels(popt, perr)
            ax.errorbar(volts, mu, marker='o', mfc="k", mec='k', ms=5, ls='none',
                        yerr=err, ecolor="k", elinewidth=1.5, capsize=4, label=None)

            if MeanDataBoltz:
                ax.plot(self.vsim, yfit, ls='--', c="k", lw=2, label=lab)

            out = pd.concat([mu, err], axis=1, ignore_index=False)
            out.columns = [self.fname + s for s in ["_mu", "_sem"]]
            out.index = volts

            out_pars = pd.DataFrame(
                data={
                    self.fname: popt,
                    self.fname + "__err": perr
                }
            )
            out_pars.index = list(popt.keys())

        if save_csv:
            out.to_csv(self.outpath + "SummaryGV.csv")
            out_pars.to_csv(self.outpath + "SummaryGV_FitParams.csv")
            print(
                " Average GV with errors and Boltzmann fit parameters saved to: \n",
                self.outpath,
                "\n    GV = ...SummaryGV.csv, \n   FitParams = ...SummaryGV_FitParams.csv"
            )

        ax.legend(fontsize=12, labelspacing=0.7)
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(axis='both', labelsize=12)
        ax.locator_params(axis="x", nbins=5)

        fig.tight_layout()

        if save_fig:
            plt.savefig(self.outpath + "%s_SummaryGV.png" % save_fig, dpi=300)
            plt.savefig(self.outpath + "%s_SummaryGV.svg" % save_fig, dpi=300)

        plt.show()
