import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import bessel, sosfiltfilt

cmap = plt.cm.get_cmap("gist_rainbow")
plt.style.use("dark_background")


def apply_bessel(A, khz, desired_freq=0.1, show=False):
    """
    Create and apply 4th order Bessel filter. Sampling parameters can be viewed in pClamp. 

    A = array of data  
    khz = sample frequency -> 'angular frequency'  
    desired_freq = target frequency after filtering, in khz 

    Returns filtered `df` as `np.ndarray`
    """
    # normalize target frequency
    # desired_freq = desired_freq / (khz/2)
    desired_freq *= 2/khz

    # norm = mag -> normalize so that gain magnitude is -3dB at khz (angular frequency)
    # sos = second order sections representation
    # return digital filter
    # fs = sample frequency of data
    sos = bessel(4, desired_freq, btype="lowpass", analog=False, output="sos")

    # apply filter along rows of `A`
    output = sosfiltfilt(sos, A, axis=0)

    if show:
        x = np.arange(df.shape[0]) / khz

        plt.plot(x, A, alpha=0.5, label="Data")
        plt.plot(x, output, lw=2, label="Filtered")

        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    return output


def find_cap_dt(df, N, khz=2, window=15, dT=500):
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
    # take absolute value to remove dependence on polarity
    dfa = df.iloc[:, :N].abs().values
    khz = int(khz)

    if dfa.shape[0] < window*khz:
        print(dfa.shape, " < ", window*khz)
        raise Exception("Not enough data to estimate capacitance.")

    # find rate of change between `window` and `window + dT ms`
    # if length of pulse is less than 500ms, iteratively reduce by 25ms
    while dfa.shape[0] < (dT + window)*khz:
        dT -= 25

    if dT < 100:
        print(dT)
        raise Exception("Window for capacitance estimation less than 100 ms")

    df_dt = (dfa[window*khz:(dT+window)*khz, :] - dfa[:dT*khz, :]) / window*khz
    mu = np.mean(np.abs(df_dt), axis=0)

    # find index of pd Series where the rate of current change first falls below the mean `mu`
    caps = []
    for i in range(N):
        for j in range(df_dt.shape[0]):
            # J = df_dt[j, i]

            if abs(df_dt[j, i]) > mu[i]:
                continue
            else:
                caps.append(j+1)
                break

        # plt.plot(df.iloc[caps[i]:, i])

    if len(caps) > 0:
        if max(caps) - min(caps) > 5*khz:
            return np.median(caps)
        else:
            return max(caps)
    else:
        return 0


class ExtractTraces():
    def __init__(self, df, ind, intervals, ntraces, khz, fname, show=False,
                 return_voltages=False, ramp="x", env=False,
                 manual_cap_offset={}):
        """
        Extract pulses from a dataframe given capacitance duration and bounding interval.
        For `ramp`, `de`, and `act` protocols, returns test voltages (or ramp half-durations for ramps), extracted test pulses with current and voltage columns` 

        `ind` = upper index that bounds test pulse, i.e. [u-2, u] 
        `intervals` = dict {sweeps : {epochs : durations}} 
        `ntraces` = number of traces 
        `khz` = sampling frequency in khz 
        `fname` = filename 
        `return_voltages` = True -> returns corresponding voltage protocols  
        `manual_cap_offset` = dictionary of filenames as keys and additional capacitance offset in time units as values 

        ### `ramp` = "x", "dt", or "de" 
        * "x" -> Nothing  
        * "dt" -> Equal-duration ramps. Returns half-durations instead of voltages. `ind` is the end of second ramp. Extracts both first and second ramps.  
        * "de" -> Deactivating ramps (fixed prepulse followed by varying-duration depolarizing ramp.) `ind` is end of ramp. Returns duration of and Extracts depolarizing ramps.   

        ### `env` = bool; if `True`, treats file as an envelope of tails protocol:  
        For `env` protocols, returns `[act_volt, tail_volt], test_times, df_to_fit` when return_voltages=False.  
        * `act_volt` and `tail_volt` are voltages for activation and deactivation, respectively, and taken from the first sweep.  
        * `test_times` contains four epochs between the start of the first activation and end of the second activation. `df_to_fit` takes current data from this entire range.  
        """

        self.df = df
        self.ind = ind
        self.intervals = intervals
        self.N = ntraces

        self.khz = khz
        self.fname = fname

        self.return_voltages = return_voltages
        self.ramp = ramp
        self.env = env

        self.show = show
        self.manual_cap_offset = manual_cap_offset

    def ExtractBoundedEpoch(self, test_times=[], test_voltages=[]):

        df_to_fit = []
        protocol = []

        intervals = self.intervals
        ntraces = self.N
        df = self.df
        ind = self.ind

        # extract traces bounded by (`ind`-2)th and (`ind`-1)th epoch
        for j, k in enumerate(intervals.keys()):
            if j >= ntraces:
                break

            if self.env:
                # 1st activation, start of tail, start of 2nd activation, end of 2nd activation
                ts = intervals[k][ind-4:ind]
                test_times.append(ts)

                if len(ts) < 2:
                    print("Could not extract any epochs! \n Assuming 1st activation starts at the first available epoch. If this is wrong, consider adjusting `ind` or something..")
                    print("All epochs: ", intervals[k])
                    print("Index: %d" % ind)
                    ts = intervals[k][:4]

                if len(test_voltages) < 1:
                    # act volt
                    test_voltages.append(df.iat[ts[0]+50, ntraces+j])
                    # tail volt
                    test_voltages.append(df.iat[ts[1]+50, ntraces+j])
                elif test_voltages[1] == test_voltages[0]:
                    test_voltages[0] = df.iat[ts[0]+50, ntraces+j]
                    test_voltages[1] = df.iat[ts[1]+50, ntraces+j]

                # isolate current and voltage protocol from test pulse
                # extract each segment (activation, tail, envelope pulses)
                for i in range(3):
                    df_to_fit.append(
                        df.iloc[ts[i]:ts[i+1]+1,
                                j].dropna().reset_index(drop=True)
                    )

                    protocol.append(
                        df.iloc[ts[0]:ts[-1]+1, ntraces +
                                j].dropna().reset_index(drop=True)
                    )

            else:
                # start and end of jth test pulse
                if self.ramp == "dt":
                    # start of 1st ramp, middle of ramp, end of 2nd ramp
                    t0, tmid, t1 = intervals[k][ind-3:ind]
                    test_times.append(tmid-t0)
                elif self.ramp == "de":
                    t0, t1 = intervals[k][ind-2:ind]
                    test_times.append(t1-t0)
                else:
                    try:
                        t0, t1 = intervals[k][ind-2:ind]
                    except:
                        print(intervals[k])
                        print(ind)
                        print("\n Failed to extract test pulses.")
                        exit()

                    test_voltages.append(int(df.iat[t1-100, ntraces+j]))

                # isolate current and voltage protocol from test pulse
                trace = df.iloc[t0:t1+1, [j, ntraces+j]
                                ].dropna().reset_index(drop=True)
                df_to_fit.append(trace.iloc[:, 0])
                protocol.append(trace.iloc[:, 1])

        if self.env:
            if len(test_times) > self.N:
                test_times = test_times[:self.N]

            return test_voltages, df_to_fit, protocol, test_times
        else:
            if self.ramp == "x":
                return df_to_fit, protocol, test_voltages
            else:
                return df_to_fit, protocol, test_times

    def FineTune(self, df_dt, dT=1000, start=True):
        # central differences of `df_dt` between 2ms and dT ms with time spacing of 2ms

        khz = self.khz

        if start:
            to_grad = df_dt[2*khz:(2+dT)*khz:2*khz, :]
        else:
            to_grad = df_dt[-(2+dT)*khz:-2*khz:2*khz, :]

        try:
            d_base = np.abs(np.gradient(to_grad, (2*khz), edge_order=2))
            mu = np.median(np.median(d_base, axis=1), axis=0)
        except:
            return None

        if start:
            def get_median_grad(x): return np.median(
                np.median(d_base[(x*khz):(10+x)*khz, :], axis=1), axis=0)
        else:
            def get_median_grad(x): return np.median(
                np.median(d_base[-(10+x)*khz:-(x*khz), :], axis=1), axis=0)

        w = 0
        while np.any((abs((get_median_grad(w) - mu)/mu)) > 0.1):
            w += 2
            if w > dT:
                break

        return w

    def GradientEstimate(self, df_to_fit, freq=0.1, start=True, window=200):
        """
        Eestimate capacitance using gradient 
        """
        khz = self.khz

        if start:
            for_grad = df_to_fit.values[:window*khz:5]
        else:
            for_grad = df_to_fit.values[-window*khz::5]

        try:
            # gradient using central differences for every 5th sample
            dI_dt = apply_bessel(
                np.gradient(for_grad, axis=0) / (khz/5), khz/5, desired_freq=freq
            )
            # rolling average over 5ms
            dI_dt = pd.DataFrame(dI_dt).rolling(
                5*khz).mean().dropna().reset_index(drop=True)

            # reset index
            dI_dt.index = (dI_dt.index + 5*khz) / (khz/5)

            # index of first entry in each processed column of `dI_dt` that is >= 0
            dc = [
                next(i for i, v in enumerate(dI_dt.iloc[:, j]) if v >= 0) for j in range(dI_dt.shape[1]) if any(dI_dt.iloc[:, j] >= 0)
            ]

            # take median of indices if less than 100ms
            if np.median(dc) < 100*khz:
                dc = int(np.median(dc))
            # if median is larger than 100ms, take the minimum index if it's less than 100ms
            elif min(dc) < 100*khz:
                dc = int(min(dc))
            # else, we set the offset to 100ms
            else:
                dc = 100

        except:
            print("Failed using gradient to estimate capacitance duration.")
            return 0

        if start:
            return dc
        else:
            return window*khz - dc

    def extract(self, return_voltages=False):

        khz = self.khz

        # envelope
        if self.env:
            # test_times = list of 4 epochs
            # 1st hpol, start of tail, start and end of 2nd hpol
            v_, df_lis, protocol, test_times = self.ExtractBoundedEpoch()
            act_volt, tail_volt = v_

            # df_lis is a list of N x 3 dataframes
            # each df_lis[i:i+3] contains dataframes of 1st hpol, 1st, depol, 2nd hpol
            m = int(len(df_lis)/self.N)     # number of pulses saved per sweep
            if m < 1:
                print(test_times)
                print(df_lis[0].head)
                print(self.N)
                raise

            # we perform capacitance estimation on the first pulse in `df_lis`,
            # which is the activation pulse; the same values will then be
            # used to truncate each element of `df_lis`
            df_to_fit = [df_lis[i] for i in range(0, len(df_lis), m)]

        else:
            # non-ramp -> 2-step protocols where steps vary in voltage
            if self.ramp == "x":
                df_to_fit, protocol, test_voltages = self.ExtractBoundedEpoch(
                    test_voltages=[])

            # ramp -> 2-step protocols with varying duration; track as half-duration in `tmids`
            else:
                df_to_fit, protocol, tmids = self.ExtractBoundedEpoch(
                    test_times=[])

        if self.show:
            plt.close()

            f = plt.figure(figsize=(10, 5), constrained_layout=True, dpi=100)
            ax = f.add_subplot(111)

            ax.set_title("ExtractTraces: %s" % self.fname)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Current (pA)")

            for i, df_i in enumerate(df_to_fit):
                clr = cmap(i/len(df_to_fit))
                ax.plot(df_i.index/khz, df_i, c=clr, alpha=0.5, lw=2)

            plt.show()

        # print(self.fname, self.manual_cap_offset)

        # varying capacitance = time of minimum for absolute value of data
        if self.fname in self.manual_cap_offset.keys() and \
                self.manual_cap_offset[self.fname] == "top":

            print("`manual_cap_offset = top`. \n Traces will be truncated until the peak current within the first 1s. Gradient and other capacitance estimation methods will not be performed.")

            for i in range(len(df_to_fit)):
                # minimum of 5ms rolling avg over first 3s
                try:
                    idx = df_to_fit[i].iloc[10*khz:1000*khz]
                    idx = idx.abs().rolling(5*khz).mean().argmin()
                except Exception as e:
                    print(e)
                    continue

                # correct for offsets due to rolling window (5ms) and initial offset (10ms)
                idx += 15*khz

                # limit truncation at 100ms
                idx = min([100*khz, idx])

                # apply truncation of `idx` samples from the beginning and 2ms from the end
                if self.env:
                    for j in range(m):
                        df_lis[i*m + j] = df_lis[i*m + j].iloc[
                            idx:-2*khz
                        ].dropna().reset_index(drop=True)
                else:
                    df_to_fit[i] = df_to_fit[i].iloc[idx:-
                                                     2*khz].dropna().reset_index(drop=True)

                protocol[i] = protocol[i].iloc[idx:-2 *
                                               khz].dropna().reset_index(drop=True)

            df_to_fit = pd.concat(df_to_fit, axis=1)
            protocol = pd.concat(protocol, axis=1)
            df_to_fit.index *= 1/khz
            protocol.index *= 1/khz

        else:
            # concatenate traces and zero index
            df_to_fit = pd.concat(df_to_fit, axis=1)
            protocol = pd.concat(protocol, axis=1)
            df_to_fit.index *= 1/khz
            protocol.index *= 1/khz

            # track capacitance
            total_cap = np.array([0., 0.])

            # find duration of capacitive spikes
            print(" Size of extracted dataframe \n", df_to_fit.shape)

            c = int(find_cap_dt(df_to_fit, self.N, khz))
            print(" Estimated capacitance: %.1f ms" % c)

            if c > 0:
                c = min([10, c])

                total_cap[0] += int(c)
                total_cap[1] += 5

            # get dataframe as array
            df_dt = np.abs(df_to_fit.iloc[:, :self.N].values)

            # time window (in ms) for computing average rate of current change
            dT = 2000
            # iteratively reduce time window for rate of current change if exceeds length of prepulse
            while df_dt.shape[0] < (2+dT)*khz:
                dT -= 25

            w_start = self.FineTune(df_dt, dT=dT, start=True)
            w_end = self.FineTune(df_dt, dT=dT, start=False)

            if (w_start is not None) and (w_end is not None):
                if (0 < w_start < 1000) and (0 < w_end < 1000):
                    total_cap[0] += w_start
                    total_cap[1] += w_end

            if df_to_fit.shape[0] < 1:
                raise Exception("No rows of data were kept.")

            dc = self.GradientEstimate(df_to_fit, start=True, window=500)
            dc2 = self.GradientEstimate(df_to_fit, start=False, window=500)
            dc2 = dc2 if 100 > dc2 > (dc + 2) else 2

            if dc > 0 or dc2 > 0:
                print(
                    " Capacitance after considering smoothed time gradient: %d ms" % dc)

                if self.fname in self.manual_cap_offset.keys() and \
                        not isinstance(self.manual_cap_offset[self.fname], str):
                    print(" Adding %d ms to capacitance as specified in `manual_cap_offset`." %
                          self.manual_cap_offset[self.fname])
                    dc += self.manual_cap_offset[self.fname]
                else:
                    dc = min([10, dc])
                    dc2 = min([10, dc2])

                total_cap[0] += (c + dc)
                total_cap[1] += dc2

            if sum(total_cap) > 0:
                c1, c2 = (total_cap * khz).astype(int)

                print(" Total estimated capacitance: \
                        \n Beginning: {c} ms, \n End: {w} ms".format(
                    c=c1/khz, w=c2/khz))

                if self.env:
                    # collect j-th pulse for k-th sweep, where
                    # j \in m, k \in N
                    df_m = [
                        [df_lis[j + k*m].iloc[c1:-c2].dropna().reset_index(drop=True)
                         for k in range(self.N)] for j in range(m)
                    ]

                    # concatenate each vector of j-th pulses
                    df_lis = [pd.concat(d, axis=1) for d in df_m]

                    for j in range(len(df_lis)):
                        df_lis[j].index *= 1/khz

                else:
                    df_ = []
                    pro_ = []
                    for j in range(self.N):
                        df_.append(
                            df_to_fit.iloc[c1:-c2,
                                           j].dropna().reset_index(drop=True)
                        )
                        pro_.append(
                            protocol.iloc[c1:-c2,
                                          j].dropna().reset_index(drop=True)
                        )

                    df_to_fit = pd.concat(df_, axis=1)
                    protocol = pd.concat(pro_, axis=1)

                    df_to_fit.index *= 1/khz
                    protocol.index *= 1/khz

        if self.show:
            if self.env:
                for i in range(2, len(df_lis), 3):
                    plt.plot(df_lis[i])
                    plt.title("Final")
                    plt.show()
            else:
                plt.plot(df_to_fit, lw=1)
                plt.show()
            plt.close()

        if self.env:
            test_times = [
                (t[self.ind-2] - t[self.ind-3])/khz for t in test_times
            ]
            print("Envelope durations: \n", test_times)

            # df_lis = [1st hpol, start of tail, start of 2nd hpol, end of 2nd hpol]
            for i in range(len(df_lis)):
                df_lis[i].columns = test_times

            if return_voltages:
                return [act_volt, tail_volt], test_times, df_lis, protocol
            else:
                return [act_volt, tail_volt], test_times, df_lis
        else:
            if self.ramp == "x":
                df_to_fit.columns = test_voltages
                protocol.columns = test_voltages

                if return_voltages:
                    return test_voltages, df_to_fit, protocol
                else:
                    return test_voltages, df_to_fit
            else:
                df_to_fit.columns = tmids
                protocol.columns = tmids

                if return_voltages:
                    return tmids, df_to_fit, protocol
                else:
                    return tmids, df_to_fit
