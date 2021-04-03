#fitting expoenntial functions to current traces for time constants 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize 
import lmfit 
class exp_fitting():
    def __init__(self, df, khz, volts=None):
        
        self.df = self.examine_traces(df)
        self.khz = khz 
        self.time = df.index.values.tolist() 
        self.tau = 500 
        
        if isinstance(volts, list):
            self.volts = volts 
        else:
            self.volts = range(0, int(7.5*df.shape[1]), 15)
            
        #create upper and lower bounds for optimization
        # single exponential
        self.bds1 = ([-1e6, 5, -1e6], [+1e6, 1e5, 1e3])
        # double exponential
        self.bds2 = ([-1e6, -1e6, 5, 5, -1e6], 
                    [+1e6, 1e6, 1e5, 1e5, 1e3])
        # triple exponential 
        self.bds3 = ([-1e6, -1e6, -1e6, 5, 5, 5, -1e6], 
                    [+1e6, 1e6, 1e6, 1e5, 1e5, 1e5, 1e3])
    
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
    def exp2(self, t, A1, A2, tau1, tau2, C):
        return A1*np.exp(-t/tau1) + A2*np.exp(-t/tau2) + C 
    def exp3(self, t, A1, A2, A3, tau1, tau2, tau3, C):
        return A1*np.exp(-t/tau1) + A2*np.exp(-t/tau2) + A3*np.exp(-t/tau3) + C 
    
    def get_sim(self, func, popt, dt=0, x):
        """
        Simulate current using exponential function
        
        `func` = one of `exp1`, `exp2`, or `exp3` \\
        `popt` = fit parameters \\
        `dt` = delay \\
        `x` = time 
        """
        if dt > 0:
            return func(x[dt:], *popt)
        else:
            return func(x, *popt)
    
    def rrmse(self, popt, time, current, dt, func=self.exp1):
        """
        Cost function (RRMSE)
        Same as self.rmse, except 
            1. only returns RRMSE, without option for RMSE. 
            2. order of parameters compatible with `minimize` 
        """
        popt = list(popt.valuesdict().values())
        
        # ysim = [func(t, *popt) for t in time]
        ysim = func(t, *popt)
        
        sse = np.sum(np.square(current - ysim))
        
        rrmse = (sse/np.sum(np.square(current)))**0.5 
        return rrmse*(1 + 0.5*(dt/(self.tau*self.khz)))
        
    def do_fit(self, func, time, current, p0, bds, delay=0, method="leastsq"):
        """
        func: exp1, exp2, or exp3 
        trace: trace to fit 
        p0: initial params 
        bds: bounds 
        delay: current offset in time 
        method = fit method for lmfit.minimize 
        """    
        
        # bounds are tuples of (min, max) for each parameter 
        bnds = [(a, b) for a, b in zip(bds[0], bds[1])] 
        
        # minimize cost function 
        args = (time, current, delay)
        kws = [func] 
        res = lmfit.minimize(self.rrmse, p0, method=method, 
                            args=(time, current, delay), kws={"func":func})
        
        return res 
        
    def get_p0(self, y, dt):
        """
        Initial guesses for 1-3rd order exponentials
        y = data 
        dt = amount of 'delay' by which the data has been shifted 
        show = whether to show initial guesses 
        
        Returns list of lists, each containing initial parameter estimates for exp1, exp2, and exp3, respectively.
        """
        # tau is simply estimated as 1/3rd total duration 
        self.tau = int(len(dy)/3) + dt 
        
        # visualize to confirm 
        if show:
            plt.close()
            
            plt.plot(dy)
            plt.axhline(0.33*dy.iat[-1], c='r', ls='-')
            plt.axvline(tau, c='white', ls='--')
            plt.show()
            
            exit()
        
        # initial current = A - C for exp1 
        y0 = y.iat[0] - y.iat[-1] 
        
        return [
            [y0, tau, y.iat[-1]], 
            [y0, -y0/2, tau*2, tau, y.iat[-1]],
            [y0, 40*y0, -40*y0, 10*tau, tau, tau, y.iat[-1]]
            ]
        
    def fit_iter(self, func, params, dt, x, y, khz, bounds, with_delay=True, third=False):
        """
        iteratively perform fit by increasing delay until optimum is reached \\
        every 2ms, record progress to get delay vs. rmse 
        
        `func` = single, double, or triple exp \\
        `params` = initial parameters for func \\
        `dt` = initial delay \\
        `x` = time \\
        `y` = current \\ 
        `with_delay` = whether to also try fitting delay; else, simply fits entire trace \\
        `third` = whether to fit triple exponential 
        """
        
        # create parameters 
        p = lmfit.Parameters()
        for i in range(0, len(params) - 1, 2):
            p.add("A%d" % i+1, value=params[i], min=-1e4, max=1e4)
            p.add("t%d" % i+1, value=params[i+1], min=5., max=4e4)
            
            if i == len(params) - 2:
                p.add("C", value=params[-1], min=-1e-4, max=1e4)
        
        if with_delay:
            p.add("Delay", value=0, min=0, max=5e3)
        
        #initial fit without delay 
        p1 = self.do_fit(func, x, y, params, bounds, 0)
        r1 = self.rmse(func, x, y, p1, 0) 
        
        if with_delay:
            self.r1 = r1 
            params = p1 
            dt = 0 
            progress = [r1] 
            
            for t in range(1, int(self.tau*self.khz)):
                try:
                    p1 = self.do_fit(func, x[t:], y[t:], params, bounds, t)
                    r1 = self.rmse(func, x[t:], y[t:], p1, t) 
                except: 
                    return progress, params, dt 
                
                progress.append(self.r1) 
                
                if r1 < self.r1: 
                    self.r1 = r1 
                    params = p1 
                    dt = t 
                else:
                    break 
                
            return progress, params, dt 
        else:
            return [], p1, 0
        
    def get_fit(self, plot_results=True, return_delay=True, pdf=None):
        """
        For a single trace, fit single, double, and triple exponential functions, and return parameters for each. 
        To fit, start from t=0, then add t+=1 until error is minimized between fit and sweep. 
        """    
        x = self.time 
        N = int(self.df.shape[1]) # number of traces 

        #list to hold progress for each sweep
        A_prog = [] 
        #dictionary to hold parameters for each sweep {i: {1:, 2:}}
        D_params = {} 
        #list to hold delay for each sweep 
        A_delay = [] 
        
        #perform fitting 
        for i in range(N): 
            print("Fitting sweep %d of %d" % (i+1, N))
            
            y = self.df.iloc[:,i].dropna() 
            
            #initial parameter guesses 
            i1, i2, i3 = self.get_p0(y, 0)
            
            y = y.values 
            t = x[:len(y)]
            
            prog1, pars1, d1 = self.fit_iter(self.exp1, i1, 0, 
                                        t, y, 
                                        self.khz, self.bds1)
            try:
                prog2, pars2, d2 = self.fit_iter(self.exp2, i2, 0, 
                                            t, y, 
                                            self.khz, self.bds2) 
                
                # try:
                pars3 = self.do_fit(self.exp3, t, y, i3, self.bds3, 0)
                # except:
                #     pars3 = self.do_fit(self.exp3, t, y, i3, 
                #                         self.bds3, 0, method="min")
                
                # print(i3)
                # print(pars3[3:])
                # exit()
                
                A_prog.append([prog1, prog2])
                A_delay.append([d1, d2])
                D_params.update({i:{1:pars1, 2:pars2, 3:pars3}})
            
            except:
                try:
                    prog2, pars2, d2 = self.fit_iter(self.exp2, i2, 0, 
                                                t, y, 
                                                self.khz, self.bds2)
                    
                    A_prog.append([prog1, prog2])
                    A_delay.append([d1, d2])
                    D_params.update({i:{1:pars1, 2:pars2}})
                except:
                    A_prog.append([prog1])
                    A_delay.append([d1])
                    D_params.update({i:{1:pars1}} )
                    continue                 
                
        # time constants and amplitudes for fast components in exp2, exp3 
        tau_f = [] 
        tau_s = [] 
        amp_f = [] 
        for val in D_params.values():
            
            npar = len(val.keys())
            if npar < 2:
                continue  
            elif npar > 2:
                #find indices of fastest components 
                j2 = np.argmin(val[2][2:4]) + 2
                j3 = np.argmin(val[3][3:6]) + 3
                tau_f.append([val[2][j2], val[3][j3]])
            
                #sum amplitudes, then append proportion of fast component 
                s2 = np.sum(np.abs(val[2][:2]))
                s3 = np.sum(np.abs(val[3][:3]))
                amp_f.append([abs(val[2][j2-2])/s2, abs(val[3][j3-3])/s3]) 
                
                #index of slow component in exp3 
                j2 = abs(j2 - 5)
                j3 = np.argmax(val[3][3:6]) + 3
                tau_s.append([val[2][j2], val[3][j3]])

                # print(j2, val[2], val[2][j2])
                
            else:
                #find indices of fastest components 
                j2 = np.argmin(val[2][2:4]) + 2 
                tau_f.append([val[2][j2], np.nan]) 
                
                #sum amplitudes, then append proportion of fast component 
                s2 = np.sum(np.abs(val[2][:2]))
                amp_f.append([abs(val[2][j2-2])/s2, np.nan]) 
                
                #slow component 
                j2 = abs(j2 - 5)
                tau_s.append([val[2][j2], np.nan])
                
                # print(j2, val[2], val[2][j2])
                
        if plot_results: 
            #figure dimensions 
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

            #plot progress 
            fp, axp = plt.subplots(d[0], d[1], squeeze=False, 
                                figsize=(12,5))
            #plot exponential fits 
            fe, axe = plt.subplots(d[0], d[1], squeeze=False, 
                                figsize=(14,6))
            #plots for time constants, parameters, and delay 
            fr, axr = plt.subplots(1, 3, figsize=(15,4))
            
            axr[2].set_title("Delay (ms)")
            axr[1].set_title(r"$\frac{A_f}{A_f + A_s}$ for $2^o$")
            axr[0].set_title(r"1/$\tau$ (1/s)")
            axr_slow = axr[0].twinx() 
            axr[0].set_ylabel(r"1/$\tau_f$")
            axr_slow.set_ylabel(r"1/$\tau_s$")
            
            axr[0].plot(self.volts[:len(tau_f)], 
                        [1000/a[0] for a in tau_f], 
                        marker='s', lw=0.5, label=r"$2^o$, $\tau_f$")
            axr[0].plot(self.volts[:len(tau_f)], 
                        [1000/a[1] for a in tau_f],
                        marker='o', lw=0.5, label=r"$3^o$, $\tau_f$")
            
            axr_slow.plot(self.volts[:len(tau_s)], 
                        [1000/a[0] for a in tau_s],
                        marker='s', fillstyle='none', 
                        lw=0.5, label=r"$2^o$, $\tau_s$")
            axr_slow.plot(self.volts[:len(tau_s)], 
                        [1000/a[1] for a in tau_s],
                        marker='o', fillstyle='none', 
                        lw=0.5, label=r"$3^o$, $\tau_s$")
            
            axr[0].plot(self.volts[:len(D_params.keys())], 
                        [1000/v[1][1] for v in D_params.values()],
                        marker='x', lw=0.5, label=r"$1^o$, $\tau$")
            
            axr[1].plot(self.volts[:len(tau_f)], [a[0] for a in amp_f],
                        marker='s', label=r"$2^o$")
            # axr[1].plot(self.volts[:len(tau_f)], [a[1] for a in amp_f],
            #             marker='o', label=r"Exp3")
            
            axr[2].plot(self.volts[:len(A_delay)], 
                        [a[0]/self.khz for a in A_delay],
                        marker='x', markersize=10, label=r"$1^o$")
            
            #get handles from both plots, then add legend to axr[0] 
            h_f, l_f = axr[0].get_legend_handles_labels()
            h_s, l_s = axr_slow.get_legend_handles_labels() 
            axr[0].legend(h_f + h_s, l_f + l_s, loc='best')
            
            h = 0 
            for i in range(d[0]):
                for j in range(d[1]): 
                    
                    if h not in D_params.keys():
                        break 
                    
                    y = self.df.iloc[:,h].dropna()
                    x = y.index.values 
                    
                    axe[i,j].plot(x, y, c='white', lw=3, alpha=0.5)
                    
                    npar = len(D_params[h].keys()) 
                    
                    d_1 = A_delay[h][0] 
                    axp[i,j].plot(A_prog[h][0], c='r', lw=2)
                    e1 = self.get_sim(self.exp1, 
                                    D_params[h][1], d_1, x)
                    axe[i,j].plot(x[d_1:], e1, 
                                ls='--', c='r', lw=2, label="Exp1 = %d" % d_1)
                    axe[i,j].axvline(x[d_1], c='r', lw=2, alpha=0.5)
                    
                    if npar == 2:
                        d_2 = A_delay[h][1] 
                    
                        axp[i,j].plot(A_prog[h][1], c='g', lw=2)
                        e2 = self.get_sim(self.exp2,
                                        D_params[h][2], d_2, x)
                        axe[i,j].plot(x[d_2:], e2, 
                                    ls='--', c='g', 
                                    lw=2, label="Exp2 = %d" % d_2)
                        axe[i,j].axvline(x[d_2], c='g', lw=2, alpha=0.5) 
                    
                    elif npar == 3:
                        d_2 = A_delay[h][1] 
                    
                        axp[i,j].plot(A_prog[h][1], c='g', lw=2)
                        e2 = self.get_sim(self.exp2,
                                        D_params[h][2], d_2, x)
                        axe[i,j].plot(x[d_2:], e2, 
                                    ls='--', c='g', 
                                    lw=2, label="Exp2 = %d" % d_2)
                        axe[i,j].axvline(x[d_2], c='g', lw=2, alpha=0.5) 
                        
                        e3 = self.get_sim(self.exp3,
                                        D_params[h][3], 0, x)
                        axe[i,j].plot(x, e3,
                                    ls='--', c='lightblue', lw=2, label="Exp3 = 0")
                    
                    axp[i,j].set_title(self.volts[h]) 
                    axe[i,j].set_title(self.volts[h]) 
                    if j == 0:
                        axp[i,j].set_ylabel("RRMSE") 
                        axe[i,j].set_ylabel("Current (pA)")
                    if i == d[-1]:
                        axp[i,j].set_xlabel("Delay (1/%d ms)" % self.khz)
                        axe[i,j].set_xlabel("Time (ms)")
                    if i + j == sum(d):
                        axp[i,j].legend(["Exp1", "Exp2"], loc='upper right')
                        axe[i,j].legend(["Exp1", "Exp2, Exp3"], 
                                        loc='upper right')
                    
                    h += 1 
                    
            fp.tight_layout()
            fe.tight_layout()
            fr.tight_layout()
            
            if pdf is not None:
                pdf.savefig(Figure=fp) 
                pdf.savefig(Figure=fe)
                pdf.savefig(Figure=fr)
            
            ### plt.show()
            plt.close()        
            # exit() 
        
        if return_delay:
            return D_params, A_delay 
        else:
            return D_params 
