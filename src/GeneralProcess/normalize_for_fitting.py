# normalize data for model fitting 

import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages 
from PyPDF2 import PdfFileReader, PdfFileMerger

from GeneralProcess.ActivationCurves import BoltzmannFunctions 

def merge_pdfs(pdf_to_add, pdf_dir=None):
    """
    pdf_dir = directory to save final PDF output 
    pdf_to_add = path to pdf file to append to `pdf_dir`
    """
    if pdf_dir is None:
        raise Exception("Tried merging pdf of normalized plots, but `pdf_dir` was not specified.")
                    
    # create pdf merger object
    merger = PdfFileMerger()
    
    # append pdf1 to pdf0 
    merger.append(PdfFileReader(pdf_dir))       # previous pdf 
    merger.append(PdfFileReader(pdf_to_add))        # new pdfs 
    
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

class normalize_for_fitting():
    def __init__(self, pname, fname, dfs, khz, volts, 
                prepulse=None, postpulse=None, 
                pmins=None, PminPlot=None, 
                GV=None, boltz_params=None, paired=None
        ):
        """
        pname = protocol name 
        fname = filename 
        
        df = list of dataframes containing corresponding prepulses and leak-subtracted test pulses 
        khz = sampling frequency 
        volts = list of test pulse voltages, or (half-)ramp durations for a ramp protocol 
        
        prepulse = voltage of prepulse, only for deactivation (activation starts from holding, -35mV)
        postpulse = voltage of postpulse, only for activation (same as test voltages for deactivation)
        
        pmins = dataframe containing voltage index and Pmin column for given file 
        PminPlot = matplotlib Axes to plot final Pmins on 
        
        GV = dataframe of aggregated steady-state GV data
        boltz_params = dataframe of aggregated boltzmann fit parameters 
        
        paired = name of activation recording from the same cell    
        
        Keyword arguments that can be passed in call to `process`:
        reduce = factor by which data can be linearly reduced (i.e. keep every reduce-th point)
        trunc = extent by which to truncate beginning of extracted pulses 
        """
        self.pname = pname 
        self.fname = fname 
        self.df = dfs 
        self.N = int(self.df[0].shape[1])
        self.khz = int(khz)
        self.volts = volts 

        self.preV = prepulse 
        self.postV = postpulse 
    
        self.cmap = plt.cm.get_cmap("gist_rainbow")
        
        self.PminPlot = PminPlot 
        
        if not ("ramp_dt" in pname):         
            # if protocol is not an activation protocol 
            if "de" in pname:
                # tail pmins for deactivation; 
                # for ramp protocol, we will rely on GV/boltz/simple normalization, so pmins will not be necessary 
                if pmins is not None:
                    self.pmins = pmins.loc[volts]

                # if a paired protocol is available, then take the corresponding boltzmann parameters/GV 
                # else, take global average of GV and boltzmann parameters 
                if paired is None:
                    print("No paired activation GV is available. A global average will be used instead.")
                    self.boltz_params = boltz_params.mean(axis=0).values.tolist()
                    self.GV = GV.mean(axis=1).dropna(how="all")
                else:
                    self.boltz_params = boltz_params.loc[paired,:].values.tolist() 
                    self.GV = GV.loc[:,paired].dropna()  
                
                # if `c` in boltz_params > 0.001, apply implied scaling (1 + c) to pmins 
                if self.boltz_params is not None:
                    if self.boltz_params[-1] > 1e-3:
                        print("In normalization, scaling `pmins` by < c = %.1e > from Boltzmann fit." % self.boltz_params[-1])
                        self.pmins /= 1 + self.boltz_params[-1]
                
            # if activaiton, try to find GV/boltzmann parameters 
            # otherwise, take the global average of available boltzmann parameters/GV 
            else:
                if boltz_params is not None:
                    if fname in boltz_params.index:
                        self.boltz_params = boltz_params.loc[fname, :].values.tolist()
                    else:
                        self.boltz_params = boltz_params.mean(axis=0).values.tolist()
                    
                if GV is not None:
                    if fname in GV.columns:
                        self.GV = GV.loc[:,fname].dropna()
                    else:
                        self.GV = GV.mean(axis=1).dropna(how="all")
                    
                if paired is None:
                    self.boltz_params = boltz_params.mean(axis=0).values.tolist() 
                
                    if GV is not None: 
                        self.GV = GV.mean(axis=1).dropna(how="all")
                    else:
                        if fname in GV.columns:
                            self.GV = GV.loc[:,fname].dropna()
                        
    def boltz(self, v, vh, s, c=0):
        return (1+c)/(1 + np.exp((v-vh)/s))
        
    def get_finfs(self, prepulse=None):
        """
        Get finf values, which determine Pmax for deactivation, and steady-state Po for activation 
        `prepulse` = voltage of prepulse, only for deactivation (activation starts from holding, -35mV)
        """
        # if no prepulse, then the protocol is activation 
        # get normalized conductances for each of the test voltages 
        if prepulse is None:         
            
            # initialize empty vector
            finfs = np.zeros(self.N)
            
            for (i, v) in enumerate(self.volts):
                # get finf from GV if available 
                if v in self.GV.index:
                    finfs[i] = self.GV.loc[v]
                
                # else, estimate from boltzmann curve using parameters of paired cell fit 
                else:
                    finfs[i] = self.boltz(v, *self.boltz_params) 
        
        # if prepulse is present, then protocol is deactivation 
        # all pulses start after the specified prepulse, which determines the beginning Po 
        # as above, if prepulse voltage not in GV data, compute with Boltzmann function 
        else:
            if prepulse in self.GV.index:
                finfs = np.ones(self.N) * self.GV.loc[prepulse]
            else:
                finfs = np.ones(self.N) * self.boltz(prepulse, *self.boltz_params)
            
        # normalize finfs by `c`, where `c` is self.boltz_params[2]
        if any(f > 1 for f in finfs) or self.boltz_params[2] > 0:
            # fmax = max(finfs) + self.boltz_params[2] 
            return [f/(1+self.boltz_params[2]) for f in finfs]
        else:
            return finfs         
        
    def do_norm(self, show=True, p0=0.02, max_iter=10, avg_window=20,
                reduce=0, trunc=0, pdf_dir=None, vhold=-35
        ):
        """
        
        reduce = int; number of time points to save from normalized data 
        show = whether to show normalized plots or not 
        pdf_dir = path to PDF file to which plots will be appended using PyPDF2
        """
        
        # reset PyPlot style 
        plt.style.use("default")
        
        if "ramp_dt" in self.pname:
            print("We can't normalize equal-duration ramp, `ramp_dt,` protocols currently because we have no estimate of Pmax at the ramp's midpoint.")
            return None 
        
        khz = self.khz 
        N = self.N 
        
        finfs = self.get_finfs(prepulse=self.preV)
        # print(self.preV, finfs)
                
        def apply_truncation(df, dt, khz=khz):
            # truncate from crest of hook, if possible
            # limit search for crest within first 200ms 
            # sqrt sum of each 5ms; helps smooth/emphasize small changes in current 
            df_5sum = df.iloc[:dt*khz,:].rolling(5*khz).sum().dropna().values ** 0.5
            
            # find 'crest' of current from `df_5sum` after 3ms 
            if np.max(df_5sum[-1,:]) > np.min(df_5sum[0,:]):
                idx = np.argmin(df_5sum[3*khz:,:], axis=0) + 8*khz 
            else:
                idx = np.argmax(df_5sum[3*khz:,:], axis=0) + 8*khz 

            # apply truncation, if length to truncate is 0 - dt ms  
            for i in range(N):
                if 0 < idx[i] < dt*khz:
                    df.iloc[:,i] = df.iloc[idx[i]:,i] 

            # shift any NaNs to the top of the dataframe 
            df = df.apply(lambda x: pd.Series(x.dropna().values))
            # ensure index matches sample frequency 
            df.index *= 1/khz 

            return df 
            
        def apply_normalization(data, finfs=finfs, test=False, trunc=trunc, 
                            p0=p0, noise_bd=0.025, khz=khz,
                            max_iter=max_iter, avg_window=avg_window,
                            PminPlot=None):
            """
            Normalize `data` to the range of [`p0`, `finfs`]
            
            data = raw dataframe for single protocol
            finfs = from above; Pmax values 
            test = whether normalizing test pulses or not 
            p0 = baseline open probability, default is 0.02, e.g. Proenza and Yellen 2006.
            trunc = upper bound in ms for location of the crest of 'hooks' aka delay
            noise_bd = makeshift estimate of upper bound for 'noise'
                noise estimated by std of absolute first differences of normalized (1/max) data 
                if a given trace exceeds `noise_bd`, we apply a modest savgol_filter 
            max_iter = maximum number of normalization loops 
            avg_window = window in ms for rolling mean in normalization; 10-20ms is usually best. 
            PminPlot = same as `self.PminPlot` 
            """
            # convert `avg_window` to samples 
            avg_window *= khz 
            
            # take absolute value of current traces; 
            # abs() makes min/max estimates below invariant to direction of current change
            df1 = data.copy().iloc[:,:N].abs() 
            df1.columns = self.volts 
            
            if trunc > 0:
                df1 = apply_truncation(df1, trunc)
            
            # # apply bessel filter to noisy traces             
            # print("`apply_bessel` was previously used at beginning of all normalizations, but currently the function is not defined in this file. If needed, think of something.")
            # for i in range(N):
            #     smoothed = apply_bessel(df1.iloc[:,i].dropna().values, khz, 
            #                         desired_freq = 0.1, show=False)
            #     df1.iloc[:len(smoothed),i] = smoothed 
            
            # rolling average over `avg_window`
            df_avg = df1.rolling(avg_window).mean()
            # direction-invariant estimates of current amplitude         
            i0 = df_avg.min(axis=0).values             
            imax = df_avg.max(axis=0).values 
            
            if isinstance(p0, float):    
                p0 = np.full(self.N, p0)
            # else:
            #     # lower bound for pmin is 2e-5 
            #     for (i, p) in enumerate(p0):
            #         if p < 2e-5: p0[i] = 2e-5
                
            p0 *= finfs 
                        
            if PminPlot is not None:
                PminPlot[1].plot(self.volts, p0, lw=2, alpha=0.7, label="Normalized")
                
                # retrieve indices of unique labels 
                h, l = PminPlot[1].get_legend_handles_labels()
                l = pd.Series(l).drop_duplicates()
                h = [h[i] for i in l.index]
                PminPlot[1].legend(h, l.values.tolist(), loc='upper right', framealpha=1)
                
                if pdf_dir is not None:
                    pdf_to_add = PdfPages(r"./tmp000.pdf", keep_empty=False)    
                    pdf_to_add.savefig(bbox_inches="tight")
                    
                    plt.show()
                    pdf_to_add.close()
                    plt.close()
                    merge_pdfs(r"./tmp000.pdf", pdf_dir=pdf_dir)
                else:
                    plt.show()
                    
                plt.close()
                
            iter = 0 
            while np.any(np.abs(i0 - p0) > 0.005) or np.any(np.abs(imax - finfs) > 0.005): 
                # np.ndarray; scaling factors for each voltage
                # ensures a pmin of `p0` and steady state of `finf`
                X = (i0 - (p0*imax/finfs)) / (1 - (p0/finfs))
                df1 = finfs * ((df1 - X)/(imax - X))
        
                # rolling average over 20ms 
                df_avg = df1.rolling(avg_window).mean()
                
                # direction-invariant estimates of current amplitude         
                i0 = df_avg.min(axis=0).values             
                imax = df_avg.max(axis=0).values 
                
                iter += 1 
                if iter > max_iter:
                    print("normalization iterations exceeded %d" % max_iter)
                    break 
            
            return df1 

        def plot_normalized(df_norm, out_pdf=None, show=show):
            """
            if reduce, df_norm = [act, de, act_reduced, de_reduced]
            out_pdf = output PDF to append to `pdf_dir` 
            """            
            f, ax = plt.subplots(1, 2, figsize=(12,7))
            ax[0].set_ylabel("Normalized Open Fraction")
            ax[0].set_xlabel("Time (s)")
            ax[1].set_xlabel("Time (s)")

            if self.preV is None:
                ax[0].set_title("Activation")
                
                if self.postV is None:
                    ax[1].set_title("Deactivation at %d mV" % self.postV)
                else:
                    ax[1].set_title("Deactivation")
                    
            else:
                ax[0].set_title("Activation at %d mV" % self.preV)
            
            # leg_ncol = number of columns for legend 
            if N < 4:
                leg_ncol = N 
            elif N < 10:
                leg_ncol = int(N/2) 
            else:
                leg_ncol = 4 
            # leg_dy = y offset for legend to place it below the subplots' x-axes 
            leg_dy = -0.08*(N/leg_ncol)
            
            if reduce > 0:
                full = df_norm[:2]      # full
                red = df_norm[2:]       # reduced 
                for i in range(2):
                    d1 = full[i]        # act
                    d2 = red[i]         # de 
                    
                    for j in range(0, d2.shape[1], 2):
                        h = int(j/2)
                        clr = self.cmap((h+1)/N)
                        v = self.volts[h]
                        
                        # full data 
                        ax[i].plot(d1.index.values * 1e-3, d1.iloc[:,h], 
                                lw=2, c=clr, label=v)
                        # reduced 
                        ax[i].plot(d2.iloc[:,j]  * 1e-3, d2.iloc[:,j+1], 
                                marker='o', markersize=2, ls='none', 
                                c=clr, label=None)
                    
                    ax[i].legend(loc='upper center', bbox_to_anchor=[0.5, leg_dy], ncol=leg_ncol)
            else:
                for i in range(2):
                    d = df_norm[i]
                    
                    if len(self.volts) > 5:
                        alpha = 0.7
                    else:
                        alpha = 1
                    
                    for j, v in enumerate(self.volts):
                        clr = self.cmap((j+1)/N)
                        # v = self.volts[j]
                        
                        ax[i].plot(d.index.values  * 1e-3, d.iloc[:,j], 
                                lw=1.5, c=clr, label=v, alpha=alpha)
                    
                    ax[i].legend(loc='upper center', bbox_to_anchor=[0.5, leg_dy], ncol=leg_ncol)
            
            # ensure both plots have the same y-range
            ylims = ax[0].get_ylim()
            ax[1].set_ylim(ylims)
            
            f.suptitle(r"%s / %s" % (self.fname, self.pname))
            plt.tight_layout(rect=[0, 0.02, 1, 0.95])
            
            if out_pdf is not None:
                out_pdf.savefig(bbox_inches="tight")
                out_pdf.close()
                
            if show:
                plt.show()
            
            # exit()
            plt.close()
            
        def apply_reduction(df_norm):
            df_merge = [] 
            colnames = [] 
            
            # linearly decimate data, keeping every `reduced`-th datapoint 
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
        
        if "ramp_de" in self.pname:
            p0 = self.boltz(vhold, *self.boltz_params)
            dfa = apply_normalization(self.df[0], test=False, p0=p0)
            dfd = apply_normalization(self.df[1], test=True, p0=p0)        
            
        elif "act" in self.pname:
            print("Tail Pmin", self.boltz(self.postV, *self.boltz_params))
            dfa = apply_normalization(self.df[0], test=True,
                    p0 = self.boltz(vhold, *self.boltz_params))
            dfd = apply_normalization(self.df[1], test=False, 
                    p0 = self.boltz(self.postV, *self.boltz_params))

        else:
            print(self.pmins)
            dfa = apply_normalization(self.df[0], test=False,
                    p0 = self.boltz(vhold, *self.boltz_params))
            
            dfd = apply_normalization(self.df[1], test=True, 
                    p0 = self.pmins.values, PminPlot=self.PminPlot)

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
                # if not already made, create a temporary pdf to hold the plots of normalized data
                if not os.path.isfile(r"./tmp000.pdf"):
                    pdf_to_add = PdfPages(r"./tmp000.pdf", keep_empty=False)
                
                plot_normalized([dfa, dfd], show=show, out_pdf=pdf_to_add)
                
                # mrege plots of normalized data with previously generated PDF of plots 
                merge_pdfs(r"./tmp000.pdf", pdf_dir=pdf_dir)
                
            elif show:
                plot_normalized([dfa, dfd], show=show)
                
            return dfa, dfd 
   