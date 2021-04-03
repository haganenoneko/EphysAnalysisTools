# fitting activation curves
import numpy as np 
import pandas as pd 
import math 
from EphysInfoFilter import EphysInfoFiltering, FindPairedFiles

import lmfit
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, sem 

import matplotlib.pyplot as plt 
from matplotlib import rcParams 
from matplotlib.backends.backend_pdf import PdfPages 

rcParams['font.size'] = 12 
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Verdana'
rcParams['font.weight'] = 'normal'
rcParams['axes.linewidth'] = 2
rcParams['axes.labelweight'] = 'bold' 

# cmap = plt.cm.get_cmap()
cmap = plt.cm.get_cmap("Set1")
plt.style.use("dark_background")
# plt.style.use("seaborn-colorblind")

class lmfit_boltzmann():
    def __init__(self):
        """
        `v` = test voltages 
        `g` = normalized conductances 
        """
        
        # declare Boltzmann parameters 
        B_pars = lmfit.Parameters()
        B_pars.add("vh", value=-100, min=-200, max=0)
        B_pars.add("s", value=10, min=3, max=50)
        self.pars = B_pars 
        
    def boltz(self, v, vh, s):
        return 1/(1 + np.exp((v-vh)/s))

    def do_fit(self, v, g):        
        def residual(pars, volts=v, data=g):
            # unpack parameters: extract .value attribute for each parameter
            parvals = pars.valuesdict()
            vh = parvals["vh"]
            s = parvals["s"]
        
            model = [self.boltz(v, vh, s) for v in volts]
            return ((np.sum(model - data)**2)/len(volts))**0.5 

        res = lmfit.minimize(residual, self.pars, method='slsqp')
        res.params.pretty_print()
        
        self.popt = list(res.params.valuesdict().values())
        
        try:
            pcov = res.covar 
            perr = np.sqrt(np.diag(pcov))
        except:
            perr = [np.nan] * len(self.popt)
        
        return self.popt, perr 
    
    def get_fit(self, vrange, p0=[-100, 5]):
        try:
            return [self.boltz(v, *self.popt) for v in vrange]
        except:
            print("  Since no fitting was done, default parameters will be used: Vh=-100, s=5.")
            return [self.boltz(v, *p0) for v in vrange]
        
class activation_curve():
    def __init__(self, test_voltages, tails, khz,
                post_tails=None, post_tail_voltage=0,
                nparam=2,
                show_pmin=False,
                plot_results=False,
                show=True,
                pdf = None):
        """
        Compute activation curve/Pmin from leak-subtracted data 
            test_voltages = list of test voltages 
            tails = dataframe of tail currents
            post_tails = dataframe of post-tail currents, if available
            post_tail_voltage = voltage used to elicit post_tails 
            show_pmin = compute Pmin for tail and/or post_tail currents
            
        `nparam` = number of parameters to fit for activation curve. If 3, includes `c`. Else, 2 for standard Boltzmann equation.
            
        For deactivation, calling the class will simply return Pmin estimates from tail currents, and, when available, post-tail currents
        
        For activation, calling the class returns normalized Po and Boltzmann parameters
        """
        if nparam < 2 or nparam > 3:
            raise Exception(" `nparam` can only be 3 (standard Boltzmann) or 4 (Boltzmann with steady-state `c` parameter).")
            exit()
        else:
            self.nparam = nparam 
                
        # manually truncate 
        if abs(tails.iloc[:,2].sum() + 66710) < 5:
            tails.iloc[:,2] = tails.iloc[:1850*khz,2] 
                        
        # compute peak amplitude of tail currents 
        # find the index of the maximum current when tail is averaged over every 4ms 
        tail_peaks = tails.iloc[:100*khz,:].rolling(4*khz).mean().abs().dropna().reset_index(drop=True).idxmax() 
        # add 4ms to account for offset due to rolling average 
        tail_peaks = tail_peaks.values + 4*khz 
        tail_peaks = tail_peaks.tolist()                
        # get actual current values 
        tail_peaks = [(tails.iat[t, i]) for i, t in enumerate(tail_peaks)] 
                        
        # deactivation, i.e. post_tails isn't None 
        if isinstance(post_tails, pd.DataFrame):
            # for deactivation, compute pmin from tails, as well as post_tail pmin, if available
            norm_tails = pd.concat(
                [tails.iloc[:,i]/tail_peaks[i] for i in range(tails.shape[1])],
                axis=1
            )
                                    
            #Pmin from tails 
            tail_mins = norm_tails.dropna().iloc[-50*khz:,:].rolling(5*khz).mean().mean()
            self.tail_mins = tail_mins 
            
            # check if post_tail_voltage is in test_voltages; if not, ignore post_tails
            if post_tail_voltage in test_voltages:
                # index of sweep with test pulse voltage corresponding to post_tail_voltage 
                j = test_voltages.index(post_tail_voltage)
                norm_post = post_tails / tail_peaks[j] 
                                
                #normalized Pmin from post_tails 
                post_peaks = norm_post.iloc[:100*khz,:].rolling(4*khz).mean().max() 
                self.post_peaks = post_peaks  
                
                for i, x in enumerate(tail_mins):
                    if x <= -0.02:
                        # index starts from 1, so use i+1 to set NaN 
                        tail_mins[i+1] = np.nan
                        
                if show_pmin:
                    f, ax = plt.subplots(figsize=(9,6))
                    ax.plot(test_voltages, tail_mins, marker='o', lw=0.5, label="Tail")
                    ax.plot(test_voltages, post_peaks, marker='s', lw=0.5, label="Post-tail")
                    ax.legend()
                    
                    ax.set_xlabel("Voltage (mV)")
                    ax.set_ylabel(r"$P_{\mathregular{min}}$")
                    ax.set_title(r"Average $P_{\mathregular{min}}$ = %.3f" %
                            tail_mins.mean())
                    plt.tight_layout()
                    
                    if show:
                        plt.show()
                    
                    plt.close()
            
            else:
                for i, x in enumerate(tail_mins):
                    if x < 0:
                        tail_mins[i] = tails.iat[-1,i] / tails.iat[0,i]
                        
                if show_pmin:
                    plt.plot(test_voltages, tail_mins, marker='o', lw=0.5)
                    
                    plt.xlabel("Voltage (mV)")
                    plt.ylabel(r"$P_{\mathregular{min}}$")
                    plt.title(r"Average $P_{\mathregular{min}}$ = %.3f" %
                            tail_mins.mean())
                    plt.tight_layout()
                    
                    if show:
                        plt.show() 
                        
                    plt.close()
        
        # activation 
        else:
            # ignore tail currents following activating pulses below -150mV or above 100mV 
            # indices out of the range [-150, 100]
            V_InRange = [i for (i, v) in enumerate(test_voltages) if abs(v + 25) < 125]
            
            # only keep test voltages that are in range 
            if len(V_InRange) > 0:
                test_voltages = [t for i, t in enumerate(test_voltages) if i in V_InRange]
                tail_peaks = [t for i, t in enumerate(tail_peaks) if i in V_InRange]
                
                # dataframe of tail currents 
                tails = tails.iloc[:, V_InRange]                
                
            else:
                raise Exception(" No voltages were found in range [-150, 100].")
                exit()
                                    
            M = np.max(tail_peaks)
            norm_tails = tails / M 
            
            tail_mins = norm_tails.iloc[-50*khz:,:].rolling(5*khz).mean().mean(axis=0)
            tail_peaks = [t/M for t in tail_peaks] 
            
            if show_pmin:
                plt.plot(test_voltages, tail_mins, marker='o', lw=0.5)
                
                plt.xlabel("Voltage (mV)")
                plt.ylabel(r"$P_{\mathregular{min}}$")
                plt.title(r"Average $P_{\mathregular{min}} = %.3f" %
                        tail_mins.mean())
                
                plt.tight_layout()
                
                if pdf is not None:
                    pdf.savefig() 
                    
                if show:
                    plt.show() 
                    
                plt.close()
                
            self.tail_mins = tail_mins 
            self.norm_tails = tail_peaks 
            self.test_voltages = test_voltages 
        
            vrange, y, popt = self.fit_boltz(return_sim=True)
            
            if self.nparam == 3:
                print("Fit parameters: \n Vh=%.1f mV \n s=%.1f mV \n c=%.4f" % 
                    (popt[0], popt[1], popt[2]))
            else:
                print("Fit parameters: \n Vh=%.1f mV \n s=%.1f mV" % (popt[0], popt[1]))
            
            if plot_results:
                f, ax = plt.subplots(1, 2, figsize=(12, 5))
                
                ax[0].plot(test_voltages, tail_peaks, 
                        marker='o', lw=0.5, label="Data")
                
                if self.nparam == 3:
                    ax[0].plot(vrange, y, ls='--', 
                        label="Fit, \n $V_h$=%.1f mV \n $s$=%.1f mV \n c=%.4f" % 
                        (popt[0], popt[1], popt[2]))
                else:
                    ax[0].plot(vrange, y, ls='--', 
                        label="Fit, \n $V_h$=%.1f mV \n $s$=%.1f mV" % (popt[0], popt[1]))
                
                ax[0].set_title("Activation curve")
                ax[0].legend(loc='upper right')
                
                ax[1].set_title("Pmin")
                ax[1].plot(test_voltages, tail_mins, marker='o', lw=0.5) 
                
                for i in range(2):
                    ax[i].set_xlabel("Voltage (mV)")
                    ax[i].set_ylabel("Normalized Po")
                    
                plt.tight_layout()
                
                if pdf is not None:
                    pdf.savefig() 
                    
                if show:
                    plt.show()
                
                plt.close() 
        
            self.popt = popt 
            
    def boltz(self, v, vhalf, s, c=0):
        return ((1-c)/(1 + np.exp((v - vhalf)/s))) + c 
    
    def fit_boltz(self, return_sim=False):
        
        # sort voltages and normalized tail peaks 
        v, g = zip(*sorted(zip(self.test_voltages, self.norm_tails)))
                        
        # initial values and bounds for fitting Boltzmann 
        p0 = [-100, 7, 0.01]
        lb = [-200, 1, 0]
        ub = [0, 100, 0.2]
        
        # depolarization activated 
        if g[0] < g[-1]:
            p0[0] = 0
            lb[0] = -100
            ub[0] = 20
        
        # reduce number of parameters and bounds if fitting standard Boltzmann 
        if self.nparam == 2:
            p0 = p0[:2]
            bds = (lb[:2], ub[:2])
        else:
            bds = (lb, ub)
        
        popt, pcov = curve_fit(self.boltz, v, g, p0=p0, bounds=bds)
        
        if return_sim:
            vrange = range(-200, 10, 5)
            y = [self.boltz(v, *popt) for v in vrange] 
            return vrange, y, popt 
        else:
            return popt 
        
    def do_fit(self):
        return self.norm_tails, self.popt 
    
    def tail_pmins(self):
        try:
            return self.tail_mins, self.post_peaks
        except:
            return self.tail_mins 
        
    def return_test_voltages(self):
        return self.test_voltages

class Summarize_GV():
    def __init__(self, fname, dv=3.178, paired_dic={}, individual=False, vsim=range(-200, 10, 5)):
        """
        `fname` is the filename of a .csv file in the `CSV_output` folder containing voltages and normalized conductances. `fname` can also be a list of Strings. If so, all dataframes will be concatenated. 
        `dv` = liquid junction potential, subtracted from test_voltages. Calculated using LJPCalc
            https://swharden.com/software/LJPcalc/theory/#ljpcalc-calculation-method
        `paired_dic` = dictionary of {parent : child} filenames. Otherwise, will retrieve automatically.
        `individual` = whether to treat List of Strings in `fname` as individual dataframes; if False, then the data are concatenated and analyzed together
        `vsim` = voltages to simulate Boltzmann 
        
        Computes statistics and make figure.
        """
        # path to csv output 
        path = r"./output/Processing/Pooled_Analyses/CSV_output/"
        
        # read dataframe of normalized conductances vs voltage; paths may specify single String or List of Strings
        if isinstance(fname, list):
            # concatenate all dataframes column-wise
            df_list = [pd.read_csv(path + f, header=0, index_col=0) for f in fname]
            
            if individual:
                df = df_list             
            else:
                # concatenate dataframes 
                df = pd.concat(df_list, axis=1)
                # remove duplicate columns (filenames)
                df = df.loc[:, ~df.columns.duplicated()]                
        else:
            df = pd.read_csv(path + fname, header=0, index_col=0)
        
        self.df = df 
        self.fname = fname 
        self.dv = dv 
        self.paired_dic = paired_dic
        self.vsim = vsim 
        self.outpath = path 
        
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
                print(" Finding paired files failed. \n Continuing, assuming all replicates are biological replicates.")
                
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
            mu = pd.concat([mu, paired_mu], axis=1, ignore_index=False).mean(axis=1)
            # std**2 -> variance -> average variance -> sqrt(variance) = std 
            err = pd.concat([err, paired_err], axis=1, ignore_index=False).pow(2).mean(axis=1).pow(0.5)
            
        return mu, err 
    
    def boltzfit(self, df, mu, LJPcorrection=True):
        """
        Use lmfit to fit Boltzmann function to G-V data, with voltages in index of `df` and conductances in `mu` \\
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
        popt, perr = LM.do_fit(volts, mu)
        yfit = LM.get_fit(self.vsim)
        
        return yfit, volts, popt, perr 
    
    def go(self, fname_as_label=100, save_csv=False, save_fig=False):
        """
        Run stats, fit Boltzmann, and return figure
        
        `fname_as_label` = when multiple dataframes are being summarized, `fname_as_label` specifies the section of respective filenames to use as legend labels 
        e.g. if `fname_as_label` < 100 -> uses `fname_as_label` to index the output of `fname.split('__')` as legend label
        
        `save_csv` = whether to fit mean/sem and fit output as csv 
        `save_fig` = whether to save figure
        """
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_xlabel("Voltage - LJP (mV)", fontsize=14)
        ax.set_ylabel("G(V)", fontsize=14)
        
        if isinstance(self.df, list):
            df_list = [] 
            df_colnames = [] 
            pars_list = {} 
            
            for i, d in enumerate(self.df):
                clr = cmap((i+1)/len(self.df))
                
                mu, err = self.do_stats(self.fname[i], d)
                yfit, volts, popt, perr = self.boltzfit(d, mu)
                
                # set index of average/error GV to LJP-corrected voltages 
                mu.index = volts 
                err.index = volts 
                
                # legend label with Boltzmann parameters and data name 
                lab = [] 
                
                if fname_as_label < 100:
                    # skip the last element, which describes type of output, e.g. 'act_norm.csv'
                    f = self.fname[i].split("__")[:-1]
                    if fname_as_label <= len(f):
                        lab.append(f[fname_as_label])
                
                lab.append(r" $V_h$ = %.1f (%.1f) mV" % (popt[0], perr[0]))
                lab.append(r" s = %.1f (%.1f) mV" % (popt[1], perr[1]))
                lab = "\n ".join(lab)
                
                ax.errorbar(volts, mu, marker='o', mfc=clr, mec='k', ms=5, ls='none',
                            yerr=err, ecolor=clr, elinewidth=1.5, capsize=4, label=None)
                ax.plot(self.vsim, yfit, ls='--', c=clr, lw=2, label=lab)
                
                df_list.append(mu)
                df_list.append(err)
                df_colnames.append(self.fname[i])
                df_colnames.append(self.fname[i] + "__err")
                
                pars_list.update({self.fname[i] : popt})
                pars_list.update({self.fname[i] + "__err" : perr})
                
            out = pd.concat(df_list, axis=1, ignore_index=False)
            out.columns = df_colnames             
            
            out_pars = pd.DataFrame.from_dict(pars_list)
            out_pars.index = ["Vh", "s"] 

        else:
            mu, err = self.do_stats(self.fname, self.df)
            yfit, volts, popt, perr = self.boltzfit(self.df, mu)
            
            lab = [] 
            lab.append(r"$V_h$ = %.1f (%.1f) mV" % (popt[0], perr[0]))
            lab.append(r"s = %.1f (%.1f) mV" % (popt[1], perr[1]))
            "\n ".join(lab)
            
            ax.errorbar(volts, mu, marker='o', mfc="k", mec='k', ms=5, ls='none',
                        yerr=err, ecolor="k", elinewidth=1.5, capsize=4, label=None)
            ax.plot(self.vsim, yfit, ls='--', c="k", lw=2, label=lab)

            out = pd.concat([mu, err], axis=1, ignore_index=False)
            out.columns = [self.fname]
            out.index = volts 
            
            out_pars = pd.DataFrame(
                data={
                    self.fname : popt, 
                    self.fname + "__err" : perr
                },
                index = ["Vh", "s"]
            )
        
        if save_csv:
            out.to_csv(self.outpath + "SummaryGV.csv")
            out_pars.to_csv(self.outpath + "SummaryGV_FitParams.csv")
            print(" Average GV with errors and Boltzmann fit parameters saved to: \n", self.outpath, "\n    GV = ...SummaryGV.csv, \n   FitParams = ...SummaryGV_FitParams.csv")
                
        ax.legend(fontsize=12)    
        ax.tick_params(axis='both', labelsize=12)
        
        fig.tight_layout()
        
        if save_fig:
            plt.savefig(self.outpath + "SummaryGV.png", dpi=300)
        
        plt.show()
        