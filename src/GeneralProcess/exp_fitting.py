#fitting expoenntial functions to current traces for time constants 

import numpy as np 
import pandas as pd 
import lmfit 
import math 
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt 


def sort_lists(reference, x):
    """
    Sorts elements of lists `x` by sorting `reference`
    
    Returns sorted zip of lists 
    """
    # sort zip of lists
    # specify key for sorting the ith element of `reference` and `x` as the first element of the tuple of the sorted zip object, e.g. pair[0] = (reference[i], x[i])[0]     
    if isinstance(x[0], list):
        s = sorted(zip(reference, *x), key=lambda pair: pair[0])
    else:
        s = sorted(zip(reference, x), key=lambda pair: pair[0])
    
    return zip(*s) 
    
def printer(res):
    """
    Neatly print out fit parameters from a `lmfit` results object
    """
    for i, t in enumerate(res.params.valuesdict().items()):
        if i == 0:
            print("\n {k} = {v}".format(k=t[0], v=t[1]))
        else:
            print("\t {k} = {v}".format(k=t[0], v=t[1]))
            
def convert_indices(ind, offset):
    """
    Convert indices between lists with different number of elements.
    """
    return int(ind*2 + offset)
            
def exp_label(order, delay, residual):
    return "Exp%d = %d (%.1e)" % (order, delay, residual)

def EnvExp1(t, A, tau):
    return A*np.exp(-t/tau)

class exp_fitting():
    def __init__(self, df, fname, khz, sigma=None, volts=None):
        """
        `df` = dataframe containing extracted test pulses 
        `fname` = filename 
        `khz` = sampling frequency 
        `sigma` = standard deviation, used for regression; 1D array of standard deviations of errors in ydata 
        """
        
        # skip manual check if not a dataframe 
        if type(df) == list:
            self.df = df 
            self.N = int(df[0].shape[1])
            self.time = []
        else:
            self.df = self.examine_traces(df)
            self.N = int(self.df.shape[1])      # number of traces 
            self.time = df.index.values         # times 
            
            # zero time axis 
            if self.time[0]:
                self.time -= df.index[0]
                    
        self.fname = fname 
        self.khz = khz 
        self.tau = 500 
        self.std = sigma 
        
        self.D_params = None 
        self.delays = None 
        
        if volts is not None:
            if isinstance(volts, list):
                self.volts = volts 
            else:
                self.volts = range(0, int(7.5*df.shape[1]), 15)
            
        # empty parameters for exp fitting with lmfit 
        self.pars1 = dict(A1=np.nan, tau1=np.nan, C=np.nan, Delay=np.nan)
        self.pars2 = dict(A1=np.nan, tau1=np.nan, A2=np.nan, tau2=np.nan, C=np.nan, Delay=np.nan)
        self.pars1 = dict(A1=np.nan, tau1=np.nan, A2=np.nan, tau2=np.nan, 
                        A3=np.nan, tau3=np.nan, C=np.nan)
            
        #create upper and lower bounds for optimization
        # single exponential
        self.bds1 = ([-1e6, 5, -1e6], [+1e6, 5e4, 1e3])
        # double exponential
        self.bds2 = ([-1e6,5,  -1e6, 5, -1e6], 
                    [1e6, 5e4, 1e6, 5e4, 1e3])
        # triple exponential 
        self.bds3 = ([-1e6, 5, -1e6, 5, -1e6, 5, -1e6], 
                    [1e6, 5e4, 1e6, 5e4, 1e6, 5e4, 1e3])
    
    def examine_traces(self, df):
        """
        Examine input dataframes using manually pre-defined criteria
        """
        # truncate 20d1009 -25mV tail due to abrupt change in holding level 
        if abs(df.iloc[:,2].sum() + 66710) < 5:
            df.iloc[:,2] = df.iloc[:1850*self.khz,2] 
        return df 
        
    def exp1(self, t, A, tau, C):        
        return A*np.exp(-t/tau) + C 
    def exp2(self, t, A1, tau1, A2, tau2, C):
        return A1*np.exp(-t/tau1) + A2*np.exp(-t/tau2) + C 
    def exp3(self, t, A1, tau1, A2, tau2, A3, tau3, C):
        return A1*np.exp(-t/tau1) + A2*np.exp(-t/tau2) + A3*np.exp(-t/tau3) + C 
    
    def find_inflxn(self, y, show=False):
        """
        Find infection point in data `y` by maximum of first time derivative 
        Returns time of inflection point 
        """
        
        # window for savgol filter 
        w = 501*self.khz
        if self.khz % 2 == 0:
            w = 501*self.khz + 1
        
        # smoothen data with savgol filter and compute first time derivative
        y_2 = np.gradient(savgol_filter(y, w, 3, delta=1/self.khz), 1/self.khz)
        
        # find maximum of [absolute = polarity invariant] first derivative 
        t0 = np.argmax(np.abs(y_2))
        
        if show:
            f, ax = plt.subplots()
            
            ax.plot(y_2, c='yellow', alpha=0.5)
            ax.axvline(t0, c='yellow', lw=3, ls='--')
            
            axb = ax.twinx()
            axb.plot(y, c='white', alpha=0.75)
                        
            plt.show()
            plt.close()
            # exit()
            
        return int(t0 / self.khz)  
    
    def get_sim(self, popt, func, time):
        """
        Simulate current using exponential function
        
        `func` = one of `exp1`, `exp2`, or `exp3` \\
        `popt` = fit parameters \\
        `x` = time 
        """
        if popt is None:
            return None, None  
        
        popt = popt.valuesdict()

        if "Delay" in popt.keys():
            delay = popt["Delay"]
            
            if 0 < delay < time[-1]:
                # index corresponding to delay 
                # dt = next(i for i, t in enumerate(time) if t >= delay)
                dt = int(math.floor(delay*self.khz))
                time = time[dt:]                
            else:
                dt = 0
                
            popt = list(popt.values())[:-1]
            ysim = func(time, *popt)
            return dt, ysim 
        
        else:
            popt = list(popt.values())
            ysim = func(time, *popt)
            return 0, ysim 
            
    def weighted_cost(self, x, dt, T=251, w=0.3):
        """
        Apply weighting on array residual `x` by increasing weight of `x(t < T)` by `w`, and vice versa for `x(t >= T)`, which is normalized.
        
        `x` = array of residuals, i.e. true - simulation
        `dt` = delay, in samples 
        `T` = cutoff for applying weighting 
        `w` = weighting 
        """
        n_a = int(self.khz*T) - dt
        
        if n_a < 1:
            # apply weighting 
            x[:n_a] *= (1 + w/n_a)      
            x[n_a:] *= (1 - w/(1 - n_a))
        
        return x 
            
    def cost(self, popt, func, time, current, array=True):
        """
        Cost function (RRMSE)

        If searching for delay, `time` and `current` data are assumed to be already truncated, e.g. time = time[delay:]
        
        `array` = return array of residuals; else, return scalar given by mean squared error 
        """
        # simulate exponential 
        dt, ysim = self.get_sim(popt, func, time)
        
        # residual 
        if array: 
            E = np.zeros(len(current))
            if time[dt] < 250:
                E[dt:] = np.square(self.weighted_cost(current[dt:] - ysim, dt))
            else:
                E[dt:] = np.square(current[dt:] - ysim)
        else:
            if time[dt] < 250:
                E = np.sum(np.square(self.weighted_cost(current[dt:] - ysim, dt)))/len(ysim)
            else:
                E = np.sum(np.square(current[dt:] - ysim))/len(ysim)
                
        # divide by standard deviations 
        if self.std is not None:
            E *= 1/self.std 
            
        # sum of squares with penalty on large delay
        if dt > 0:
            E *= (1 + (dt*self.khz)/len(current))
        
        return E 
        
    def LossWithDelay(self, result):
        """
        Account for fitting of delay and return a mean squared error
        `result` = MinimizerResult from lmfit 
        """
        return result.chisqr / (result.ndata - result.nvarys - result.params["Delay"]*self.khz)
        
    def do_fit(self, func, time, current, p0,  
            method=["ampgo", "nelder", "powell", "lfbgsb", "leastsq"],
            try_all_methods="chained", show=False):
        """
        Returns best-fit parameters as `lmfit.Parameters` class object 
        
        func: exp1, exp2, or exp3 
        trace: trace to fit 
        p0: initial params 
        bds: bounds 
        delay: current offset in time 
        method = fit method for lmfit.minimize; if ith method doesn't work, uses (i+1)th, etc.
        try_all_methods = 
            if True, fits with all methods in `method`, then selects the result with lowest residual
                if all methods fail or if `try_all_methods=False`, the default method is `nelder`
            if False, fits with default, `nelder`
            if 'chained', first computes the result with `leastsq`, then uses the resulting parameters as initial points for methods in `method`, ultimately keeping the most optimal solution
        
        Returns parameters and residual of lmfit MinimizerResult object 
        """    
        # minimize function `func` 
        args = (func, time, current)        
        
        res = None 
        
        # try all optimization algorithms, keeping the one that gives the best result
        if try_all_methods == True:         
            res = [] 
            for i, m in enumerate(method):
                try:
                    res.append(lmfit.minimize(self.cost, p0, method=m, args=args))
                except:
                    continue 
            
            # only keep the result with lowest cost function 
            if len(res) > 1:
                idx_min = np.argmin([self.LossWithDelay(r) for r in res])            
                res = res[idx_min] 
            else:
                # default
                res = lmfit.minimize(self.cost, p0, method="nelder", args=args)
                
        elif try_all_methods == "chained":
            res = lmfit.minimize(self.cost, p0, method="leastsq", args=args)
            loss1 = self.LossWithDelay(res)
            
            for m in method:
                res_m = lmfit.minimize(self.cost, res.params, method=m, args=args)
                if self.LossWithDelay(res_m) < loss1:
                    res = res_m
                    break 
                else:
                    continue 
            
        else:
            res = lmfit.minimize(self.cost, p0, method="nelder", args=args)
                
        if res is None:
            raise Exception("   Fitting failed.")
            
        # print fit results 
        # printer(res) 
        print(lmfit.fit_report(res))
        
        if show:
            dt, ysim = self.get_sim(res.params, func, time)
            plt.plot(time, current, lw=1, c='white', alpha=0.3)
            plt.plot(time[dt:], ysim, lw=2, ls='--', c='y')
            
            if dt > 0:
                plt.axvline(time[dt], lw=2, c='y', label=time[dt])
            
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()
            exit()
            
        # return parameters and residual
        return res.params, res.residual
        
    def get_p0(self, y, dt, show=False):
        """
        Initial guesses for 1-3rd order exponentials
        `y` = data 
        `dt` = amount of 'delay' by which the data has been shifted 
        `show` = whether to show initial guesses 
        
        ## Single-exponential:
            A1, tau1, C 
            A1 + C = y[0], C = y[-1], therefore A1 = y[0] - y[-1]
            tau1 = duration/3 + dt 
        
        ## Double-exponential:
            A1, A2, tau1, tau2, C 
            C = y[-1], A1 + A2 + C = y[0] 
            
            We assume: A1/(A1 + A2) = 0.8, so A1 + A2 + C = A1 (1 + 1/4) + C = 5/4 A1 + C = y[0]
            So, A2 = A1/4, A1 = (y[0] - C) * (4/5) = 0.8 * (y[0] - y[-1])
            
            We assume: tau2 = tau1*2 
        
        ## Triple-exponential:
            A1, A2, A3, tau1, tau2, tau3, C 
            C = y[-1], A1 + A2 + A3 + C = y[0]
            
            We assume: A1/(A1 + A2 + A3) = 0.7, and A2/(A1 + A2 + A3) = 0.2
            Then, A2 = (2/7)*A1, and A3 = (1/7)*A1 
            A1 + A2 + A3 + C = A1 * (1 + 3/7) + C = A1*(10/7) + C = y[0], so A1 = (7/10)*(y[0] - y[-1])
            
            We assume: tau2 = tau1*2, tau3 = tau1*3
        
        Returns list of lists, each containing initial parameter estimates for exp1, exp2, and exp3, respectively.
        """
        dy = (y - y[0]).abs()
        
        # exp1 A1 = initial current - final current 
        y0 = y.iat[0] - y.iat[-1] 
        
        # tau is simply estimated as 1/3rd total duration 
        tau = dt + dy.where(dy > (dy.iat[-1]/2.718)).dropna().index[0] 
        self.tau = tau
        
        # visualize tau to confirm 
        if show:
            plt.close()
            
            plt.plot(y)
            plt.axvline(tau, c='white', ls='--')
            plt.show()
            
            exit()
                
        return [
            [y0, tau, y.iat[-1]], 
            [0.7*y0, tau, 0.3*y0, tau*2, y.iat[-1]],
            [0.7*y0, tau, (2/7)*y0, 2*tau, (1/7)*y0, 3*tau, y.iat[-1]]
            ]
        
    def get_fit(self, func, params, x, y, khz, with_delay=True, show=False):
        """
        iteratively perform fit by increasing delay until optimum is reached \\
        every 2ms, record progress to get delay vs. rmse 
        
        `func` = single, double, or triple exp \\
        `params` = initial parameters for func \\        
        `x` = time \\
        `y` = current \\ 
        `with_delay` = whether to also try fitting delay; else, simply fits entire trace \\
        `show` = whether to plot fit result
        """
        
        # create parameters 
        p = lmfit.Parameters()
                
        # amplitudes and taus 
        for i in range(0, len(params) - 1, 2):
            j = int((i+1)/2) + 1
            p.add("A%d" % j, value=params[i], min=-5e3, max=1e4)
            p.add("tau%d" % j, value=params[i+1], min=5., max=2e4)
        
        # constant 
        p.add("C", value=params[-1], min=-3e4, max=1e4)
        
        # add delay as a parameter to fit 
        if with_delay:
            # estimate delay as maximum of first time derivative of current 
            delay = max([50, self.find_inflxn(y) * 0.5])
                        
            if delay > 1000:
                p.add("Delay", value=500, min=50, max=1000)
            elif 200 < delay < 1000:
                p.add("Delay", value=delay, min=10, max=delay + 100)
            else:
                p.add("Delay", value=delay, min=0, max=delay + 50*khz)
        else:
            # add a fixed delay parameter to enable self.LossWithDelay
            p.add("Delay", value=0, vary=False)
        
        # fit (with delay if `with_delay = True`)        
        # r1 = self.rrmse(p1, x, y, p1, func=func) 
        return self.do_fit(func, x, y, p, show=show)
        
    def do_fitting(self, with_delay=True, third=False, plot_every=False):
        """
        Fit traces in `self.df`
        
        Output:
        `D_params` = dictionary of parameters, see below for structure 
        `D_res` = dictionary of fit residuals, same structure as `D_params`
        `with_delay` = whether delay is used 
        `third` = whether to fit triple exponential 
        """
        x = self.time 
        N = self.N 
                
        #dictionary to hold parameters for each sweep, {i : {1 : [...], 2 : [...]}}
        # e.g. D_params[i][1] gives parameters for single-exponential fit for ith sweep
        D_params = {} 
        # dictionary to hold residuals for each sweep {i : {[res1, ...]} }
        D_res = {}
        
        #perform fitting 
        for i in range(N): 
            print("Fitting sweep %d of %d" % (i+1, N))
            
            # data to fit 
            y = self.df.iloc[:,i].dropna() 
            
            #initial parameter guesses 
            i1, i2, i3 = self.get_p0(y, 0)
            
            y = y.values 
            t = x[:len(y)]
            
            # fit parameters 
            params = {}
            # residuals 
            residuals = []
            
            # single exponential 
            pars1, res1 = self.get_fit(self.exp1, i1, t, y, self.khz, 
                                    with_delay=with_delay, show=plot_every)
            params.update({1 : pars1 })
            residuals.append(np.sum(res1)/len(res1))

            # try sum of two (if `third`, three) exponentials 
            try:
                pars2, res2 = self.get_fit(self.exp2, i2, t, y, self.khz, 
                                    with_delay=with_delay, show=False) 
                
                params.update({2 : pars2 })
                residuals.append(np.sum(res2)/len(res2))
            except:
                print("Fitting with exp2 failed for %d-th sweep. Returning all NaNs" % i)
                params.update({2 : None})
                residuals.append(np.nan)
                                
            if third:
                try:
                    pars3, res3 = self.get_fit(self.exp3, i3, t, y, self.khz, 
                                    with_delay=False, show=plot_every)
                    params.update({3 : pars3})
                    residuals.append(res3)
                except:
                    params.update({2 : None})
                    residuals.append(np.nan)
            
            if len(params.keys()) > 0:
                D_res.update({i : [r for r in residuals if r != np.nan]})            
                D_params.update({i : params})
                
        return D_params, D_res 
        
    def extract_fit_params(self, D_params):
        """
        Extract fast and slow time constants, and proportion of fast component from fits
        
        `D_params` = dictionary of fit parameters
        
        Returns: 
        `tau_f`, `tau_s`, and `amp_f`, which are lists of lists of fast taus, slow taus, and proportion of fast components, respsectively. 
        
        `amp_f` is calculated by absolute values, e.g. |A_f| / |A_f| + |A_s|
        
        Each list has the structure [[2, 3], [2, 3], [..]], where 2 and 3 indicate given parameter values for 2- and 3-exponential fits, the latter being NaN if only double exponential fits were done.
        """
        # time constants and amplitudes for fast components in exp2, exp3 
        tau_f = [] 
        tau_s = [] 
        amp_f = [] 
        
        for val in D_params.values():
            # val = dictionary of {order : dict} for each trace 
            
            # convert each parameter set in `val` to a list if not None 
            val = {k:list(v.valuesdict().values()) for k, v in val.items() if v is not None}
            # number of parameter sets (accounting for possible None types)
            i = len(val.keys())
            
            # only one component for single-exponential, so we can skip 
            if i < 2:
                continue 
            
            elif i == 3:
                #find indices of fastest components 
                j2 = convert_indices(np.argmin(val[2][1:5:2]), 1)
                j3 = convert_indices(np.argmin(val[3][1:7:2]), 1)
                tau_f.append([val[2][j2], val[3][j3]])
            
                #sum amplitudes, then append proportion of fast component 
                s2 = np.sum(np.abs(val[2][:2]))
                s3 = np.sum(np.abs(val[3][:3]))
                amp_f.append([abs(val[2][j2-2])/s2, abs(val[3][j3-3])/s3]) 
                
                #index of slow component in exp3 
                j2 = abs(j2 - 4)
                j3 = convert_indices(np.argmax(val[3][1:7:2]), 1)
                tau_s.append([val[2][j2], val[3][j3]])
            
            else:                
                #find indices of fastest components 
                j2 = convert_indices(np.argmin(val[2][1:5:2]), 1)
                tau_f.append([val[2][j2], np.nan]) 
                
                #sum amplitudes, then append proportion of fast component 
                s2 = np.sum(np.abs(val[2][0:4:2]))
                amp_f.append([abs(val[2][j2-1])/s2, np.nan]) 
                
                #slow component 
                j2 = abs(j2 - 4)
                tau_s.append([val[2][j2], np.nan])
                
        return tau_f, tau_s, amp_f 
    
    def fit_traces(self, with_delay=True, third=False, plot_every=False, 
                plot_results=True, show_plots=True, return_plots=False, 
                save_path=None, pdf=None):
        """
        For each trace in `self.df`, fit single, double, and triple exponential functions, and return parameters for each. 
        
        bool arguments:
        `with_delay` = whether to use delay in fitting (for 1- and 2-components)
        `third` = whether to fit 3 components
        `plot_every` = whether to show every individual fit/trace
        `plot_results` = whether to plot results 
        `show_plots` = whether to show plots (otherwise, figures remain active, and can be modified)
        `return_plots` = updates class variable `self.plots` with the current figures, which can then be returned by calling `self.return_plots()`
        
        `save_path` = if not None, figures will be saved here 
        `pdf` = if not None, figures will be appended to the existing PDF 
        
        Returns:
        `D_params` = dictionary of fit parameters 
        
        If `with_delay=True`, returns `D_params, delays`, where `delays` dictionary of fitted delay with same structure as `D_params`. Here, `D_params` loses delay parameters. 
        """    
        
        if not return_plots:
            self.canvas = None 
        
        D_params, D_res = self.do_fitting(with_delay=with_delay, third=third, plot_every=plot_every)
        tau_f, tau_s, amp_f = self.extract_fit_params(D_params)
        
        if with_delay:
            self.D_params, self.delays = self.return_fit_results(D_params, with_delay=with_delay,
                                                                third=third)
        else:
            self.D_params = self.return_fit_results(D_params, with_delay=with_delay, third=third)

        if plot_results: 
            canvas = self.create_figure(both=True)
            self.plot_traces(D_params, D_res, canvas=canvas[:2])
            self.plot_params(D_params, tau_f, tau_s, amp_f, self.delays, canvas=canvas[2:],
                            third=third, with_delay=with_delay)
            
            if return_plots:
                self.canvas = canvas 
            
            if pdf is not None:
                for i in range(0, 4, 2):
                    pdf.savefig(canvas[i])
            
            if save_path is not None:
                canvas[0].savefig(save_path + self.fname + "_traces.png", dpi=300, bbox_inches='tight')
                canvas[2].savefig(save_path + self.fname + "_params.png", dpi=300, bbox_inches='tight')
                print("Figures successfully saved at < %s >" % (save_path + self.fname + "..."))
            
            if show_plots:
                plt.show()
                plt.close()

        return self.D_params, self.delays 
        
    def create_figure(self, both=True, traces=False, params=False):
        """
        Create figure for plotting fit results.
        `both` = figures and axes for both individual traces and fit parameters
        If `both` is False, 
            `traces` = figure for only individual traces 
            `fit` = figure for only fit parameters 
        """
        if both or traces:
            # number of rows and columns for plotting individual traces  
            N = self.N 
            if 2 < N < 5:
                d = (2, 2) 
            elif N > 4:
                d = int(N**0.5) 
                if d**2 < N:
                    d = (d, d+1)
                else:
                    d = (d, d)
            else:
                d = (1, 2) 
            
            fe, axe = plt.subplots(d[0], d[1], squeeze=False, figsize=(14,6), constrained_layout=True)
        
        if both or params:
            #plots for time constants, parameters, and delay             
            fr = plt.figure(figsize=(10,6), constrained_layout=True)
            gs = fr.add_gridspec(nrows=7, ncols=2)
            axr = [fr.add_subplot(gs[:4,:]), fr.add_subplot(gs[4:,0]), fr.add_subplot(gs[4:,1])]
                                    
            axr[0].set_title(r"Rates, $\tau^{-1}$ (1/s)")
            axr[1].set_title(r"$\frac{A_f}{A_f + A_s}$ for $2^o$")
            axr[2].set_title("Delay (ms)")
            
            axr[0].set_ylabel(r"$\tau_{1}^{-1}$" + "\n " + r"$\tau_{f}^{-1}$", 
                            labelpad=15, fontsize=12, rotation=0)
            
            axr_slow = axr[0].twinx() 
            axr_slow.set_ylabel(r"$\tau_{s}^{-1}$", labelpad=15, fontsize=12, rotation=0)
            
            for a in axr:
                a.set_xlabel("Voltage (mV)")
        
        if both:
            return fe, axe, fr, axr, axr_slow
        elif traces:
            return fe, axe 
        elif params:
            return fr, axr, axr_slow 
    
    def plot_traces(self, D_params, D_res, canvas=None):
        """
        Plot individual traces overlayed with exponential fits 
        
        `D_params` = dictionary of fit parameters, {i : {1 : [..], 2 : [..], 3: [..]} }
        e.g. D_params[i][1] indexes the monoexponential fit of the ith sweep 
        
        `D_res` = dictionary of fit residuals, follows the same structure as `D_params`
        
        If `canvas` is None, then new figures are made using `self.create_figure()`
        Else, `canvas` contains `[fig, ax, fig, ax]` which are the figure and axes of individual traces and fit parameters, respectively.
        """
        
        if canvas is None:
            fe, axe = self.create_figure(both=False, traces=True, params=False)
        else:
            if len(canvas) == 2:
                fe, axe = canvas 
            else:
                raise Exception("`canvas` must be of length 2, holding [figure, ax]")
        
        # dimensions of axis 
        d = axe.shape 
                
        h = 0 
        for i in range(d[0]):
            for j in range(d[1]): 
                
                # clear unused plots 
                if h not in D_params.keys():
                    axe[i,j].axis('off')
                    h += 1 
                    continue 
                
                # plot data 
                y = self.df.iloc[:,h].dropna()
                # time for simulation 
                x = y.index.values  
                # time for plotting 
                ts = y.index.values * 1e-3 
                # plot data 
                axe[i,j].plot(ts, y, c='white', lw=3, alpha=0.5)
                
                # number of parameter sets fit for ith sweep 
                npar = len(D_params[h].keys()) 
                
                # simulate and plot exp1 
                dt, e1 = self.get_sim(D_params[h][1], self.exp1, x)
                
                # indicate delay with fitting exp1 
                lab = exp_label(1, dt/self.khz, D_res[h][0])
                # lab = "Exp1 = %d (%.1e)" % (dt, D_res[h][0])
                if dt > 0:
                    axe[i,j].plot(ts[dt:], e1, c='r', lw=2, label=lab)
                    axe[i,j].axvline(ts[dt], c='r', lw=2, ls='--')
                else:
                    axe[i,j].plot(ts, e1, c='r', lw=2, label=lab)
                
                # if 2 or more parameter sets, then there are higher order fits 
                if npar >= 2:
                    dt, e2 = self.get_sim(D_params[h][2], self.exp2, x)
                    
                    if dt is None:
                        h += 1 
                        continue 
                    
                    lab = exp_label(2, dt/self.khz, D_res[h][1])
                    # "Exp2 = %d (%.1e)" % (dt, D_res[h][1])
                    if dt > 0:
                        axe[i,j].plot(ts[dt:], e2, c='lightblue', lw=2, label=lab)
                        axe[i,j].axvline(ts[dt], c='lightblue', lw=2, ls='--') 
                    else:
                        axe[i,j].plot(ts, e2, c='lightblue', lw=2, label=lab)
                        
                    if npar == 3:
                        # no delay for triple exponential fits, so ignore `dt` 
                        dt, e3 = self.get_sim(D_params[h][3], self.exp3, x)
                        
                        if dt is None:
                            h += 1 
                            continue 
                        
                        lab = exp_delay(3, 0, D_res[h][2])
                        # "Exp3 (%.1e)" % D_res[h][2]
                        axe[i,j].plot(ts, e3, c='gray', lw=2, label=lab)
                
                # title each subplot with test voltage 
                axe[i,j].set_title(self.volts[h]) 
                
                # ylabel in first column of plots 
                if j == 0:
                    axe[i,j].set_ylabel("Current (pA)")

                # xlabel in bottom row of plots 
                if i == (d[0] - 1):
                    axe[i,j].set_xlabel("Time (s)")

                # legend 
                axe[i,j].legend(loc='center right', fontsize=10)
                
                h += 1 
                
    def plot_params(self, D_params, tau_f, tau_s, amp_f, delays, 
                    with_delay=True, third=False, canvas=None):
        """
        Plot parameters from exponential fitting
        
        `D_params` = dictionary of fit parameters, see docstring of `self.plot_traces` for structure 
        
        The following are lists of [[2, 3], [2, 3], ...], where [2, 3] represent given parameters for 2nd and 3rd order exponentials, respectively 
        `tau_f` = fast taus 
        `tau_s` = slow taus 
        `amp_f` = fast amplitude / sum of amplitudes 
        `delays` = delays, structured as delays for each order of fit, for each sweep
                    e.g. [[delay1, delay2, ...] [...]]
        `with_delay` = whether delay is used 
        If `canvas` is None, new figure is made using `self.create_figure(both=False, params=True)`
        """
        
        if canvas is None:
            fr, axr, axr_slow = self.create_figure(both=False, traces=False, params=True)
        else:
            if len(canvas) == 3:
                fr, axr, axr_slow = canvas 
            else:
                raise Exception("`canvas` must be of length 3, holding [figure, axs, axs_slow]")
        
        # elements of `tau_f`, `tau_s`, and `amp_f` are lists for all parameter sets of given trace        
        
        # taus of exp2 
        v, tau_f2, tau_s2 = sort_lists(
            self.volts[:len(tau_f)], 
            [[1000/a[0] for a in tau_f], 
            [1000/a[0] for a in tau_s]]
        )
        # fast tau 
        axr[0].plot(v, tau_f2, marker='s', lw=0.5, label=r"$2^o$, $\tau_f$")
        # slow tau 
        axr_slow.plot(v, tau_s2, marker='s', fillstyle='none', lw=0.5, label=r"$2^o$, $\tau_s$")
        
        # taus of exp3 
        if third:
            v, tau_f3, tau_s3 = sort_lists(
                self.volts[:len(tau_f)],
                [[1000/a[1] for a in tau_f],
                [1000/a[1] for a in tau_s]]
            )
            
            axr[0].plot(v, tau_f3, marker='o', lw=0.5, label=r"$3^o$, $\tau_f$")
            axr_slow.plot(v, tau_s3, marker='o', fillstyle='none', lw=0.5, label=r"$3^o$, $\tau_s$")
        
            # fast amplitude ratio for exp3 
            # axr[1].plot(self.volts[:len(tau_f)], [a[1] for a in amp_f],
            #             marker='o', label=r"Exp3")
            
        # exp1 tau 
        v, tau_1 = sort_lists(
            self.volts[:len(D_params.keys())], [1000/v[1]["tau1"] for v in D_params.values()]
        )
        axr[0].plot(v, tau_1, marker='x', lw=0.5, label=r"$1^o$, $\tau$")
        
        # fast amplitude ratio for exp2 
        v, amp_f = sort_lists(
            self.volts[:len(tau_f)], [a[0] for a in amp_f]
        )
        axr[1].plot(v, amp_f, marker='s', label=r"$2^o$")
        
        # delay for exp1 and exp2 
        if with_delay:
            for j in range(2):
                # select j-th order delay from `delays`
                dt = [x[j] for x in delays]
                
                # sort delays with test voltages 
                v, dt = sort_lists( self.volts[:self.N], dt)
                
                # marker for 2- vs 1-exp delay
                m = 'x' if (j == 1) else 's'                     
                axr[2].plot(v, dt, marker=m, markersize=8, label="%d$^o$" % j)
        
        #get handles from both plots, then add legend to axr[0] 
        h_f, l_f = axr[0].get_legend_handles_labels()
        h_s, l_s = axr_slow.get_legend_handles_labels() 
        axr[0].legend(h_f + h_s, l_f + l_s, loc='upper center', ncol=3, framealpha=0.5)
        
    def return_plots(self):
        if self.canvas is None:
            raise Exception("`return_plots()` called, but `self.canvas = None`")
        else:
            return self.canvas 

    def return_fit_results(self, D_params, with_delay=True, third=False):
        """
        Convert values in `D_params` into list 
        """
        
        # convert lmfit-style dictionary of parameters to normal dictionary 
        # k1 = sweep #, k2 = order of exponential fit, v2 = lmfit parameters object 
        D_params = {k1 : {
            k2 : v2.valuesdict() for k2, v2 in D_params[k1].items() if v2 is not None 
        } for k1 in D_params.keys()}
        
        if with_delay:
            # extract delay into list of lists
            # list of delay for each voltage = [[exp1 delay, exp2 delay], [...]]
            if third:
                delays = [[D_params[i][j]["Delay"] for j in D_params[i].keys()] for i in range(self.N)]
            else:
                delays = [[D_params[i][j]["Delay"] for j in D_params[i].keys()] for i in range(self.N)]
            
            # check that all fits are accounted for 
            # all_fits = (len(x) != len(delays[0]) for x in delays[1:])
            # if any(all_fits):
            #     idx = np.where(x)
            #     print("Fits of sweeps ", idx, " do not have expected number of fits. \
            #         Replacing with nans.")
                
            #     # index of sweep not in `idx`
            #     expected = [i for i in range(self.N) if i not in idx][0]
            #     expected = len(delays[expected])
                
            #     for i in idx:
            #         if len(expected) - len(delays[idx]) == 1:
            #             if len(delays[idx][0])
            
            # remove delay from `D_params`
            # k1 = sweep #, k2 = order of exponential fit, v2 = dictionary of parameters 
            # D_params = {k1 : {k2 : list(v2.values())[:-1] for k2, v2 in D_params[k1].items()} for k1 in D_params.keys()}
            for k1, v1 in D_params.items():
                for k2, v2 in v1.items():           
                    if v2 is None:
                        continue 
                    else:
                        D_params[k1][k2] = [v3 for k3, v3 in v2.items() if k3 != "Delay"]
            
            return D_params, delays 
                    
        else:
            # convert dictionary of parameters
            D_params = {k1 : {k2 : list(v2.valuesdict().values()) for k2, v2 in D_params[k1].items() if v2 is not None} for k1 in D_params.keys()}
        
            return D_params 
        
    def return_fit_params(self):
        return self.D_params, self.delays 
    
    def FitEnvelope(self, env_times, show=False, subtract_baseline=False):
        """
        Fit extracted envelope pulses with a single exponential.
        
        `env_times` = list of envelope times 
        If `subtract_baseline = True`, the exponential constant is constrained (-10, +10) due to subtraction of baseline current at the activation potential.
        
        Returns fit parameters and amplitude of subtracted current
        """
        khz = self.khz 
        
        # the initial current level of first hyperpolarization 
        # this is the expected level of the 2nd hyperpolarization after full deactivation
        I_0 = self.df[0].iloc[:100*khz, :].rolling(2*khz).mean().dropna()
        # maximum current at end of first hyperpolarization
        I_1 = self.df[0].iloc[-100*khz:, :].rolling(2*khz).mean().dropna()
        # the maximal current level at onset of each envelope pulse (2nd activation)
        I_2 = self.df[2].iloc[:100*khz, :].rolling(2*khz).mean().dropna()

        # check polarity
        if (self.df[0].iloc[0, :] > self.df[0].iloc[-1,:]).all():
            I_0 = I_0.max(axis=0).values
            I_1 = I_1.max(axis=0).values
            I_2 = I_2.max(axis=0).values
        else:
            I_0 = I_0.min(axis=0).values
            I_1 = I_1.min(axis=0).values
            I_2 = I_2.min(axis=0).values
        
        if subtract_baseline:
            # subtract I_1 and I_2 of i-th sweep with respsective I_0
            for i in range(self.N):
                I_1[i] -= I_0[i]
                I_2[i] -= I_0[i]

        # check if there is a 0-duration step    
        # this establishes initial amplitude of exponential (final activation level)
        # otherwise, add as mean of max activation amplitude over all sweeps 
        if env_times[0] > 0:
            env_times.insert(0, 0.0)
            I_2 = np.insert(I_2, 0, I_1.mean())
        
        # time for simulating exponential 
        tsim = np.linspace(0, max([env_times[-1], 1000]), 200)
        if subtract_baseline:
            popt, pcov = curve_fit(EnvExp1, env_times, I_2, 
                                p0=[I_2[0], np.median(env_times)],
                                bounds=( [-1e4, 10.], [1e4, 1e5] ))
            ysim = EnvExp1(tsim, *popt)
            lab = "A = %.1f \ntau = %.1f, \nC = 0" % (popt[0], popt[1])
        else:
            popt, pcov = curve_fit(self.exp1, env_times, I_2,
                        p0=[I_2[0], np.median(env_times), I_0[0]],
                        bounds=(
                            [-1e4, 10., -1e4], [1e4, 1e5, 1e4]
                        ))
            ysim = self.exp1(tsim, *popt)
            lab = "A = %.1f \ntau = %.1f, \nC = %.1f" % (popt[0], popt[1], popt[2])

        if show:
            plt.plot(env_times, I_2, marker='o', ls='none', markersize=6)
            plt.plot(tsim, ysim, ls='--', lw=2, alpha=0.5, label=lab)            
            
            plt.legend(loc='lower right')
            plt.title("Envelope Fit: %s" % self.fname)
            plt.tight_layout()
            plt.show()
        
        if subtract_baseline:
            return popt, I_0 
        else:
            return popt 
        
        