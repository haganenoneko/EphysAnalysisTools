import glob
import math
import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
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

from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit, minimize, least_squares
from scipy.stats import pearsonr, sem

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
        # self.epochs = epochs 
        self.RT_F = (8.31446261815324*298)/96.485
        # self.ramp_df = None 
        
        # black border to plots of fitted leak current
        self.line_border = [pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]
        
        self.ion_set = {"Na" : [14, 110], "K" : [135, 35], "Cl" : [141, 144.6]} 
        if isinstance(ion_set, dict):
            # assume only Na, K, and Cl permeate the membrane at rest 
            ions = ["Na", "K", "Cl"]
            
            # test that Na, K, and Cl are provided 
            if all(x in ion_set.keys() for x in ions):
                if all(len(ion_set[x]) == 2 for x in ions):
                    self.ion_set = ion_set
                else:
                    print("     Provided `ion_set` does not have 2 values (inside, outside concentrations) for each ion. Resorting to default. \n")
                    print(ion_set)
            else:
                print("     Provided `ion_set` does not have one or more of Na, K, or Cl. Ensure the keys match exactly and that all are present. Resorting to default.")
                print(ion_set)

    def ohmic_leak(self, V_cmd, g_leak, E_leak, 
                g_res=None, Vhold=None):
        """
        Fit ohmic leak equation: I_leak = g_leak * (V_cmd - E_leak)
            where, 
                g_leak = conductance of leak current (pA/mV)
                V_cmd = command voltage (mV)
                E_leak = reversal potential of leak current (mV)
        
            If residual=True, return residual leak current, I_res:
                I_res = I_leak + g_res*(V_cmd - Vhold)
                I_leak_total = I_res + I_leak
            Where, 
                g_res = conductance of residual leak current 
                V_cmd[0] = reversal potential of residual leak, set to holding potential, which is initial voltage of voltage ramp     
        """
        if self.res:
            return g_res*(V_cmd-Vhold) + g_leak*(V_cmd-E_leak)
        else:
            return g_leak * (V_cmd - E_leak)
        
    def ghk(self, v, x):
        """
        GHK current equation  
        voltage `v`   
        list of ion concentrations `x` = [internal, external]  
        """
        e = v/self.RT_F 
        return e*(x[0] - x[1]*math.exp(-e))/(1 - math.exp(-e))
    
    def ghk_leak(self, V, P_K):
        """
        Compute GHK leak equation.  
        Besides P_K, free parameters are permeabilities for each ion w.r.t to that of K.  
        
        References for GHK equations:  
        1. Johnston and Wu, p. 58  
        2. Hille 2001, p. 473  
        3. https://en.wikipedia.org/wiki/Goldman%E2%80%93Hodgkin%E2%80%93Katz_flux_equation  
        """       
        Na = self.ion_set["Na"]
        K = self.ion_set["K"]
        
        # assume only K and Na permeate at rest and find P_Na / P_K 
        # P_Na  
        P_Na = P_K*(self.E*K[0] - K[1])/(Na[1] - self.E*Na[0])
        
        if isinstance(V, np.ndarray):
            out = np.zeros((len(V)))
            for i in range(len(V)):
                if V[i] == 0:
                    out[i] = 96.485*(
                        P_K*(K[0] - K[1]) + P_Na*(Na[0] - Na[1])
                    )
                else:
                    out[i] = 96.485*(P_K*self.ghk(V[i], K) + P_Na*self.ghk(V[i], Na))
            return out 
        else:
            if V == 0:
                return 96.485*(
                    P_K*(K[0] - K[1]) + P_Na*(Na[0] - Na[1])
                )
            else:
                return 96.485*(
                    P_K*self.ghk(V, K) + P_Na*self.ghk(V, Na)
                )
    
    def ghk_leak_triple(self, V, P_K, P_Cl):
        """
        GHK equations for K+, Na+, and Cl-.  
        P_K = absolute permeability for K+  
        P_Cl = relative permeability p_Cl / P_K  
        """
        Na = self.ion_set["Na"]
        K = self.ion_set["K"]
        Cl = self.ion_set["Cl"]
        
        # relative permeability P_Na / P_K
        P_Na = (1/(Na[1] - self.E*Na[0])) * ((self.E*K[0] - K[1]) + P_Cl*(self.E*Cl[1] - Cl[0]))
        
        if isinstance(V, np.ndarray):
            out = np.zeros((len(V)))
            for i in range(len(V)):
                if V[i] == 0:
                    out[i] = 96.485*(
                        P_K*((K[0] - K[1]) + P_Na*(Na[0] - Na[1]) + P_Cl*(Cl[0] - Cl[1]))
                    )
                else:
                    out[i] = 96.485*(
                        P_K*(self.ghk(V[i], K) + P_Na*self.ghk(V[i], Na) + P_Cl*self.ghk(V[i], Cl))
                    )
            return out 
        else:
            if V == 0:
                return 96.485*(
                    P_K*((K[0] - K[1]) + P_Na*(Na[0] - Na[1]) + P_Cl*(Cl[0] - Cl[1]))
                )                
            else:
                return 96.485*(
                    P_K*(self.ghk(V, K) + P_Na*self.ghk(V, Na) + P_Cl*self.ghk(V, Cl))
                )
    
    def fit_ghk_leak(self, i, v, mode="double"):
        """
        Fit GHK leak equation.  
        `i` = current values, target  
        `v` = voltage, input  
        """
        if mode == "double":
            self.popt, c = curve_fit(self.ghk_leak, v, i, 
                    p0 = [1e-5],
                    bounds=([1e-8], [10])
            )
        elif mode == "triple":
            self.popt, c = curve_fit(self.ghk_leak_triple, v, i, 
                                p0 = [1e-5, 0.1],
                                bounds=([1e-8, 1e-5], [1, 1])
            )
        else:
            raise Exception("   In call to `fit_ghk_leak`, `mode` argument was %s, which was not understood. Currently supported are `mode='double'` and `mode='triple'` for Na/K and Na/K/Cl being the only permeable ions at rest, respectively." % mode)
    
    def ghk_get_permeability(self, x):
        """
        Return implicit permeabilities from fitting GHK equations to leak ramps.  
        `x` = fit parameter(s). List of fit parameters for each trace.  
            * For `mode='double'`, only Na+ and K+ are permeable. P_K is the only fit parameter, so each x[i] is length 1.  
            * For `mode='triple'`, Na+, K+, and Cl- are permeable. P_K and P_Cl/P_K are the fit parameters, so each x[i] is length 2. P_Na is inferred from GHK voltage equation, and P_Cl is converted to absolute permeability via P_K * P_Cl.  
        `ions` = dictionary of {ion name : [internal, external] concentrations} for each ion. Number of keys must match length of x[i].  
        self.E = RMP*96.485/8.3145/298, where RMP is the resting membrane potential.  
        
        Returns  
            * `out` = list of relevant absolute permeabilities, either [P_Na] for `mode = double` or [P_Na, P_Cl], for each trace in `self.ramp_df`  
        """
        
        E = self.E 
        K = self.ion_set["K"]
        Na = self.ion_set["Na"]
        
        if isinstance(x, float):
            return x*((E*K[0] - K[1])/(Na[1] - E*Na[0]))
        
        out = [] 
        
        if len(self.ion_set.keys()) < len(x[0]):
            raise Exception("   In call to `ghk_get_permeability`, dictionary `self.ion_set` does not have the same number of keys as the fit parameters for the first trace. \n e.g. if only Na+ and K+ are permeable (`mode = 'double'`), then there is only one fit parameter, and `ions` should at least have keys 'Na' and 'K'.")
        
        if len(x[0]) == 1:
            for i, P_K in enumerate(x):
                out.append(P_K*(E*K[0] - K[1])/(Na[1] - E*Na[0]))
        else:
            Cl = self.ion_set["Cl"]
            for i in range(len(x)):
                P_K, P_Cl = x[i] 
                
                P_Na = (P_K/(Na[1] - E*Na[0])) * ((E*K[0] - K[1]) + P_Cl*(E*Cl[1] - Cl[0]))
                out.append([P_K*P_Cl, P_Na])
        
        return out 
        
    def fit_ohmic_leak(self, i, v):
        """
        Fit ohmic leak equation to current and voltage
            i = ramp current
            v = ramp voltage 
        """
        
        if self.res:
            self.popt, c = curve_fit(self.ohmic_leak,
                            v, i,
                            p0 = [-1, -10, -1, -1],
                            bounds = ([-15, -20, -15, -20], 
                                    [15, 20, 15, 20])
            )
        else:
            self.popt, c = curve_fit(self.ohmic_leak,
                            v, i,
                            p0 = [-1, -10],
                            bounds = ([-15, -80], [15, 20])
            )
    
    def get_leak_current(self, traceV, params, mode="ohmic"):
        """
        Calculate leak current for a trace given its voltage protocol.
        traceV = command voltage  
        """
        if mode == "ohmic":
            if self.res:            
                return [self.ohmic_leak(v, *params) for v in traceV]
            else:
                return [self.ohmic_leak(v, *params) for v in traceV]
        elif mode == "double":
            return [self.ghk_leak(v, *params) for v in traceV]
        elif mode == "triple":
            return [self.ghk_leak_triple(v, *params) for v in traceV]
        else:
            raise Exception("   `mode` argument not understood in call to `get_leak_current`. \n    %s was provided, but only 'ohmic', 'double', and `triple` are currently supported." % mode)
            
    def find_ramp(self, df, return_ramp=False, show_ramp=False):
        """
        df = dataframe or array containing current in columns [0...N], and voltage in columns [N+1...2*N]
        return_params: if True, will return 
            [fit_parameters, leak-subtracted df]
        """
        #convert numpy array to pandas dataframe 
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        elif isinstance(df, pd.DataFrame):
            pass 
        else:
            print("`df` must be pd.DataFrame or np.ndarray")
            raise TypeError
        
        #number of traces 
        N = int(df.shape[1]/2)
        
        # get start and end times for both arms of the ramp 
        if len(self.ramp_startend) > 2:
            # whether the protocol was 'transformed' or not 
            self.transformed = True 
            
            # file-specific transform for leak ramp epochs returns a list of start and end times
            ramp_epochs = self.ramp_startend
            
            if len(ramp_epochs)/2 != N:
                print("Number of leak ramp epochs != number of traces.")
            
            # find ramp for each trace 
            ramp_df = [] 
            ramp_pro_df = [] 
            for i in range(0, len(ramp_epochs), 2):
                t0, t1 = ramp_epochs[i:i+2]
                # print(ramp_epochs, df.shape[0])
                
                j = int(i/2)
                ramp_df.append(df.iloc[t0:t1+1, j])
                ramp_pro_df.append(df.iloc[t0:t1+1, N+j])
            
            # concatenate 
            ramp_df = pd.concat(ramp_df, axis=1).apply(lambda x: pd.Series(x.dropna().values))
            ramp_pro_df = pd.concat(ramp_pro_df, axis=1).apply(lambda x: pd.Series(x.dropna().values))
            ramp_df = pd.concat([ramp_df, ramp_pro_df], axis=1)
        else:
            self.transformed = False 
            
            t0, t1 = self.ramp_startend
        
            # get dataframe of just the voltage ramps
            # ramp_df = df.iloc[self.ramp_startend[0]:self.ramp_startend[1]+1, :]
            ramp_df = df.iloc[t0:t1+1, :]
                
        if show_ramp:
            plt.plot(df.iloc[self.ramp_startend[0]-100:self.ramp_startend[1]+101, :])
            plt.show()
            plt.close()
            # exit()
        
        if return_ramp:
            return ramp_df 
        else:
            self.ramp_df = ramp_df 

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
        N = int(df.shape[1]/2)
        tmp = df.copy() 
        
        if plot_results:
            f = plt.figure(figsize=(12,6))
            gs = f.add_gridspec(2, 3)
            
            ax0 = f.add_subplot(gs[0,0])
            ax0.set_title(r"$\gamma$")
            ax0.set_ylabel("pS", rotation=0, labelpad=18)
            
            ax1 = f.add_subplot(gs[0,1])
            ax1.set_title(r"$E_{\mathregular{leak}}$")
            ax1.set_ylabel("mV", rotation=0, labelpad=18)
            
            ax2 = f.add_subplot(gs[0,2])
            ax2.set_title(r"$r^2$")
            
            ax3 = f.add_subplot(gs[1, :])
            ax3.set_ylabel(r"Current (pA)")
                        
            times = self.ramp_df.index
            times -= times[0] 
            
            ax3.plot(times, self.ramp_df.iloc[:,:N], lw=2, alpha=0.7, 
                    c='lightblue', label="Original")
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
                
            ax2.plot(range(N), r_values, marker='o', ls='none', label="Data")
        
        leak_params = []    
        for i in range(N):            
            #fit leak ramp with ohmic leak equation 
            if self.transformed:
                y = self.ramp_df.iloc[:,[i, N+i]].dropna().values 
                self.fit_ohmic_leak(y[:,0], y[:,1])
            else:
                self.fit_ohmic_leak(ramp_arr[:,i], ramp_arr[:,N+i])  

            # subtract from the entire trace 
            leak_i = self.get_leak_current(df.iloc[:,N+i].values.tolist(), self.popt)
            tmp.iloc[:,i] -= leak_i 
            
            leak_params.append(self.popt) 
        
        if plot_results:
            ax0.plot(range(N), [p[0] for p in leak_params], marker='o', ls='none', markersize=6)
            ax1.plot(range(N), [p[1] for p in leak_params], marker='o', ls='none', markersize=6)
                        
            # subtract from just the ramp, and plot the fitted leak current; we do this separately from subtracting the entire recording for efficiency when we don't require visualization 
            # compute current from fitted parameters for the voltage ramp
            fitted_ramps = [] 
            # compute pearson coefficient for each fitted ramp 
            r_values = [] 
            
            # subtracted ramps 
            subtracted = self.ramp_df.copy() 
            
            for i in range(N):
                y = self.ramp_df.iloc[:,N+i].dropna().values 
                
                fitted_ramps.append(
                    self.get_leak_current(y, params=leak_params[i])
                )
                r_values.append(pearsonr(y, fitted_ramps[i])[0]) 
                
                # subtract from leak ramps 
                subtracted.iloc[:,i] = self.ramp_df.iloc[:,i] - fitted_ramps[i]                 
                
            # replace ramp current with leak-subtracted values 
            self.ramp_df = subtracted 
                
            try:
                ax3.plot(times, np.transpose(fitted_ramps), lw=2, c='yellow', alpha=0.6,
                    path_effects=self.line_border,
                    label="Fit")
            except:
                for i in range(len(fitted_ramps)):
                    y = fitted_ramps[i] 
                    times =self.ramp_df.iloc[:,i].dropna().index 
                    
                    if i == 0:
                        ax3.plot(times, y, lw=2, c='yellow', alpha=0.6,
                            path_effects=self.line_border,
                            label="Fit")
                    else:
                        ax3.plot(times, y, lw=2, c='yellow', alpha=0.6,
                            path_effects=self.line_border,
                            label=None)
            
            # r_values = [pearsonr(ramp_arr[:,N+i], fitted_ramps[i])[0] for i in range(N)]
            ax2.plot(range(N), r_values, marker='o', fillstyle='none', ls='none', label="Fit")
            
            h, l = ax2.get_legend_handles_labels()
            ax2.legend(h[:2], list(set(l)), loc='upper right', bbox_to_anchor=[1.65, 1])
            
            ax3.plot(times[::5], self.ramp_df.iloc[::5,:N], lw=2, alpha=0.7, c='r', label="Subtracted")
            ax3.set_xlabel("Time (ms)")
                        
            h, l = ax3.get_legend_handles_labels()
            ax3.legend([h[0], h[N], h[-1]], [l[N], l[0], l[-1]], 
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
                
            f.suptitle("Fitting of Linear Voltage Ramps")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            if pdfs is not None:
                pdfs.savefig(Figure=f)
                
            plt.show()
            plt.close() 
                    
        self.leak_params = leak_params
        
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
        
        ramp_arr = self.ramp_df.values    # convert dataframe to np array 
        N = int(df.shape[1]/2)
        tmp = df.copy() 
        
        if plot_results:
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
            ax1.plot(range(N), r_values, marker='o', ls='none', label="Data")
        
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
        
        if plot_results:
            # plot permeability for inferred permeabilities on separate y-axis
            ax0_inf = ax0.twinx()
            # get inferred permeabilities 
            P_inf = self.ghk_get_permeability(leak_params)
            if mode == "double":
                # plot permeability for K
                ax0.plot(range(N), leak_params, marker='o', ls='-', alpha=0.6, c='r',
                    markersize=6, label=r"$K^{+}$")  
                # ax0.axhline(np.mean(leak_params), c='r', lw=2, alpha=0.5, label=None)
            
                ax0_inf.set_ylabel(r"$P_{Na}$", rotation=0, labelpad=14)
            
                # plot permeability for Na on a separate y-axis 
                ax0_inf.plot(range(N), P_inf, marker='o', ls='-', alpha=0.6, 
                        c='lightblue', markersize=6, label=r"$Na^{+}$")
                # ax0.axhline(np.mean(P_inf), c='r', lw=2, alpha=0.5, label=None)

            else:
                P_K = [x[0] for x in leak_params]
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
                
            # compute current from fitted parameters for the voltage ramp
            fitted_ramps = [] 
            # compute pearson coefficient for each fitted ramp 
            r_values = [] 
            
            subtracted = self.ramp_df.copy()
            for i in range(N):
                y = self.ramp_df.iloc[:,N+i].dropna().values 
                
                fitted_ramps.append(
                    self.get_leak_current(y, params=leak_params[i], mode=mode)
                )
                r_values.append(pearsonr(y, fitted_ramps[i])[0]) 
                
                # subtract from leak ramps 
                subtracted.iloc[:,i] = self.ramp_df.iloc[:,i] - fitted_ramps[i]

            # pearson correlation between fit and data 
            ax1.plot(range(N), r_values, marker='o', fillstyle='none', ls='none', label="Fit")
            
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
                
            f.suptitle("Fitting of Linear Voltage Ramps")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            if pdfs is not None:
                pdfs.savefig(Figure=f)
                
            plt.show()
            plt.close() 
            # exit()
                    
        self.leak_params = leak_params
        
        return tmp
    
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
        N = int(df.shape[1]/2)
            
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
    
    def do_leak_subtraction(self, df, method="ohmic", return_params=False, plot_results=False, pdfs=None):
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
        N = df_i.shape[1]
        
        # select current and voltage in desired window given by `w`
        i_inst = df_i.iloc[:w*khz, :].mean(axis=0).values 
        voltages = df_v.iloc[1, :].values
                
        # current density 
        i_inst_cm = i_inst / Cm 
        
        # linear fits of i_inst and i_inst_cm against voltage using Chebyshev 
        LinFit_i = np.polyfit(voltages, i_inst, deg=1)
        LinFit_i_cm = np.polyfit(voltages, i_inst_cm, deg=1)
        
        # reversal potential from linear fits 
        Erev = -LinFit_i[0]/LinFit_i[1] 
        
        # fit Iinst-V with GHK current equation
        self.E = math.exp(Erev/self.RT_F)
        self.fit_ghk_leak(i_inst, voltages)
        
        # absolute and relative permeabilities
        P_K = self.popt[0]
        P_Na = self.ghk_get_permeability([self.popt])[0]
        P_Na_K = P_Na/P_K
        
        if plot_results:
            f, ax = plt.subplots(1, 2, figsize=(12, 5))
            f.suptitle("I-V")
            
            # information to label in each plot
            s = (r"$P_{Na}$ = %.1e" % P_Na) + "\n" + (r"$P_{K}$ = %.1e" % P_K) + "\n" + \
                (r"$P_{Na}/P_{K}$ = %.2f" % P_Na_K) + "\n" + \
                (r"$E_{rev}$ = %.1f mV" % Erev) + "\n" + (r"$C_m$ = %d pF" % Cm)            
            
            for i in range(2):
                # move left and bottom spines to the center
                ax[i].spines['left'].set_position('center')
                ax[i].spines['bottom'].set_position('center')
                
                ax[i].spines['top'].set_visible(False)
                ax[i].spines['right'].set_visible(False)

                # Show ticks in the left and lower axes only
                ax[i].xaxis.set_ticks_position('bottom')
                ax[i].yaxis.set_ticks_position('left')
                
                # align xlabel to the right edge of the axis 
                ax[i].set_xlabel("Voltage (mV)", ha="right", x=0.95)
                
                # add text box with the labels 
                ax[i].text(0.75, 0.2, s, transform=ax[i].transAxes, fontsize=10, va='top')
            
            # align ylabels to the top edge of the axis
            ax[0].set_title("Current (pA)")
            ax[1].set_title("Current\nDensity (pA/pF)")
            
            ax[0].plot(voltages, i_inst, marker='o', ls='none', markersize=6, c='w')
            ax[0].plot(voltages, np.polyval(LinFit_i, voltages), ls='--', c='w', alpha=0.5)
            
            ax[1].plot(voltages, i_inst_cm, marker='o', ls='none', markersize=6, c='w')
            ax[1].plot(voltages, np.polyval(LinFit_i_cm, voltages), ls='--', c='w', alpha=0.5)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            if pdfs is not None:
                pdfs.savefig(bbox_inches='tight')
            
            plt.show()
            plt.close()
        
        if output:
            # create dataframe with columns for instantaneous current and current density 
            Iinst_df = pd.DataFrame(data={"pA" : i_inst, "pA/pF" : i_inst_cm}, index=voltages)
            IV_params = pd.DataFrame(data=[Erev, P_K, P_Na], columns=["Erev", "P_K", "P_Na"])
            
            return IV_params, Iinst_df
        