def get_mu_sem(df, n=1):
        # return mean and SEM values for dataframe, averaging over axis `n` 
        mu_ = df.mean(axis=n) 
        err_ = [sem(df.iloc[i,:].dropna()) for i in range(df.shape[0])]
        return mu_, err_
    
# modified boltzmann with adjustable Pmin = c 
def boltz(v, vh, s, c=None):
    if c is None:
        return 1/(1 + np.exp((v-vh)/s))
    else:
        # return ((1-c)/(1 + np.exp((v-vh)/s))) + c
        return (1 + c)/(1 + np.exp((v-vh)/s))
    
# compute vhalf from boltzmann function, where boltz(vhalf) = 0.5 
def get_vh(vh, s, c=None):
    if c is None:
        return pars[0]
    else:
        # return -s*np.log(1-2*c) + vh 
        return vh + s*np.log(1 + 2*c)
    
def plot_exp_params(ax, x, y, err, order, props):
    """
    Plot x vs y with errorbars `err` on specified axis `ax` for `order` order exponential fit
    if `ax`, `y` and `err` are length 2 lists, then expected to contain [slow, fast] components  
    """
    if order == 1:
        if props is None:
            props = dict(marker='x', markersize=7, lw=0.5, capsize=5, c='white', label=r"$1^o, \tau$")
        ax.errorbar(x, y, err, **props)
    elif order == 2:
        if props is None:
            props = dict(ls='--', marker='s', markersize=7, capsize=5, lw=0.5)
        
        ax[0].errorbar(x, y[0], err[0], label=r"$2^o$, $\tau_f$", **props)
        ax[1].errorbar(x, y[1], err[1], fillstyle='none', label=r"$2^o$, $\tau_s$", **props)
        
    elif order == 3:
        if props is None:
            props = dict(marker='^', markersize=7, capsize=5, lw=0.5, c='r')
        
        ax[0].errorbar(x, y[0], err[0], label=r"$3^o, \tau_s$", **props)
        ax[1].errorbar(x, y[1], err[1], fillstyle='none', label=r"$3^o, \tau_f$", **props)
        
def extract_exp_params(self, df, order=2, prop=True):
    """
    Given dataframe `df` containing parameters for exponential fits, extract fast and slow time constants, and A_F/A_S ratios
    
    Structure of `df` (| = column separator)
        Columns = index, 'FileName', Voltages [...]
        Index | FileName | 2-exponential fit [...] | [...] ...
        (or Index | 3-exponential fit [...] | [...] ... )
        
    `df` = dataframe containing 2- or 3-exponential parameters for multiple voltages 
    `order` = whether extracting 2- or 3-exponential parameters 
    `prop` = whether to compute amplitude proportions; if False, returns ratio instead
    """
    # number of sweeps 
    N = int(df.shape[1])
    # test voltages 
    vrange_p = df.columns.values.tolist() 
    
    # for each column (voltage), extract all but Constant term into a list, ignoring recordings that lack any entries
    # params = [[x[:-1] for x in df.iloc[:,i].dropna().values] for i in range(N)] 
    params = [df.iloc[:,i].dropna().values.tolist() for i in range(N)] 
    FastTau = [] 
    SlowTau = [] 
    AmpProp = []  

    a = order       # index of first tau parameter 
    b = a + order   # index of last tau parameter 
        
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
            
    # FastTau = pd.DataFrame.from_records(FastTau, index=vrange_p)
    # AmpProp = pd.DataFrame.from_records(AmpProp, index=vrange_p)
    # SlowTau = pd.DataFrame.from_records(SlowTau, index=vrange_p) 
    return [pd.DataFrame.from_records(L, index=vrange_p) for L in [FastTau, SlowTau, AmpProp]]

class summarize():
    def __init__(self, ProcessObj, title=None, output=False):
        """
        `ProcessObj` = object of class `process`
        """
        self.out_files = {} 
        self.save_path = ProcessObj.output_prefix
        self.outfile_labels = [
            "act_norm.csv", "boltz_params.csv", "tail_post_pmins.csv",
            "ramp_stats.csv",
            "exp_delay.csv", "exp_params.csv"
        ]
    
        if ProcessObj.save_AggregatedPDF:
            s = "-".join(ProcessObj.) 
            self.pdf = r"{savedir}/summary_output/summary_{dates}.pdf".format(savedir=self.save_path, dates=s)
        else:
            self.pdf = None 
            
        
    def saver(self, df=None, fig=None, name=None, title=None):
        """
        Save `current` figure and dataframe. 
        Calls plt.show and plt.close afterwards, so figure space is cleared afterwards.
        
        `df` = dataframe to save as csv
        `fig` = figure to save as png 
        `name` = goes into the filename of .CSV and .PNG output
            Distinct from `title`, which, if available, appears before `name` in the filenames. 
        If `title` is absent, a generic filename is created by searching the save path for filenames containing `name`, then appending a 3-digit number at the end.             
        """
        out_path = self.save_path + "summary_output/"
        
        if title is not None:
            out_path += title 
        else:
            print("Passing an input for `title` is recommended \n   Saving under default name.")
            if name is None:
                raise Exception("At least one of `title` or `name` must be specified.")

            # find files with the given suffix
            FoundFiles = glob.glob(out_path + "%s*.png" % name)
            name += "%03d" % (len(FoundFiles) + 1)
            
        if title is None:
            out_path += name 
        else:
            out_path += "_" + name 
                
        if isinstance(df, list):
            # [mean, SEM] where elements are pd.Series 
            if len(df) == 2:
                # concatenate along columns
                df = pd.concat(df, axis=1)
                # rename columns
                df.columns = ["Mean", "SEM"]
                df.to_csv(out_path + ".csv")
                
                print("Data saved successfully at \n    < %s.csv >" % out_path)
                
            # if N is length of list `df`, then 
            # first 2/3*N are pd.Series of mu and np.array of SEM for corresponding data, 
            # i.e. mean_i, SEM_i, mean_i+1, SEM_i+1, ...
            # the last 1/3*N elements are names of elements in order 
            elif len(df) > 2:
                u = int(len(df)/3) 
                
                # convert SEM to pd.Series 
                for i in range(1, 2*u, 2):
                    df[i] = pd.Series(df[i], index=df[0].index) 
                
                # names of each data type  
                colnames = df[2*u:]
                
                # merge mean and SEM columnwise 
                df_merge = pd.concat(df[:2*u], axis=1)
                # initialize empty list of strings for column names 
                cols = [""]*df_merge.shape[1]  
                # specify column names 
                for i in range(0, 2*u, 2):
                    j = int(i/2)
                    cols[i] = colnames[j] + "_Mean"
                    cols[i+1] = colnames[j] + "_SEM"
                # assign column names 
                df_merge.columns = cols 
                
                df_merge.to_csv(out_path + ".csv")
                print("Data saved successfully at \n    < %s.csv >" % out_path)
            
            else:
                raise Exception("`df` expected to have 2 or more elements, but has ", len(df))
        else:
            raise Exception("`df` expected to be of type `list`, but received ", type(df))
        
        # save figure   
        if fig:
            fig.savefig(out_path + ".png")
        
            if self.pdf:
                self.pdf.savefig(fig)
                        
    def read_files(self):
        """
        Read each type of output file at path given by `self.save_path`
        
        Returns:
        `out_files` = dictionary with structure `{f : df}`, where `f` is a string from `self.outfile_labels` and `df` is the dataframe of corresponding data 
        """
        out_files = {} 
        for l in self.outfile_labels:
            if os.path.isfile(self.save_path + l):
                df = pd.read_csv(self.save_path + l, index_col=0, header=0).dropna(how='all')
            else:
                print("No file found at path \n     < {p} >".format(p=(path + l))
                continue 
                
            # convert str(list) into list if cells are not empty 
            if "exp_params" in l:
                for i in range(df.shape[0]):
                    for j in range(1, df.shape[1]):
                        s = df.iat[i,j]
                        
                        if s == np.nan: 
                            continue 
                        elif isinstance(s, str):
                            if ("[" in s) and ("]" in s):
                                s = df.iat[i,j][1:-1].split(", ")
                                s = [float(x) for x in s if len(x) > 0]
                                df.iat[i,j] = s 
                                
                            else:
                                try:
                                    df.iat[i,j] = float(s)
                                except:
                                    raise Exception("Failed to convert (%d, %d)-th element of dataframe with value < %s >" % (i, j, s))
                            
            
            print(" Output file: %s" % l)
            out_files.update({l[:-4] : df}) 
    
        self.out_files = out_files 
    
    def get_avg_boltz(self):
        """
        Average Boltzmann fit parameters and simulate Boltzmann
        """
        if self.out_files is None:
            raise Exception("`self.out_files` is None. `self.get_avg_boltz()` failed. Perhaps `self.read_files()` was not called.")
        else:
            if "boltz_params" not in self.out_files.keys():
                raise Exception("`self.get_avg_boltz()` failed because 'boltz_params' not in keys.")
            else:
                #average boltz params and compute activation curve 
                self.mean_gv_params = self.out_files["boltz_params"].mean(axis=0).values.tolist()

                # compute boltzmann using averaged fit parameters 
                vrange = range(-200, 0, 5) 
                self.mean_boltz_fit = [boltz(v, *self.mean_gv_params) for v in vrange]
        
    def gv_params(self, title=None, show=True, save=True):
        """
        Plots for Boltzmann fit parameters (not GV plots)
        """
        # parameters for boltzmann fit 
        fb, axb = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
        df_pars = self.out_files["boltz_params"]
        
        # compute 'true' vhalf from modified boltzmann 
        vh_ = [get_vh(*df_pars.iloc[i,:].values) for i in range(df_pars.shape[0])] 
        
        for i in range(3):
            try:
                j = i+1 
                axb[i].plot(df_pars.iloc[:,0], df_pars.iloc[:,j], 
                            marker='o', markersize=5, ls='none')
                axb[i].plot(vh_, df_pars.iloc[:,j], 
                            marker='o', fillstyle='none',
                            markersize=5, ls='none')
                
                # plot means 
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
        
        if title is None:
            axb[1].set_title(title + " Boltz. Fit Parameters")
        elif isinstance(title, str): 
            axb[1].set_title("Boltz. Fit Parameters")
            
        if save:
            self.saver(name="boltz_fit_params", title=title, fig=fb, df=)
            if show:
                plt.show()
        else:
            if show:
                print("Warning! Showing the plot without saving will wipe the plot.")
                plt.show()        
    
    def gv(self, title=None, show=True, save=True):
        #activation curve
        f_ac, ax = plt.subplots(figsize=(10,6), constrained_layout=True)

        # unpack data 
        ac_mu, ac_sem = get_mu_sem(out_files["act_norm"])
        
        # plot averaged normalized tail peaks with SEM errorbars 
        ax.errorbar(ac_mu.index.values, 
                ac_mu, ac_sem, 
                ls='none', alpha=0.5,
                marker='o', markersize=5, 
                c='white', capsize=5, 
                label="Avg.")

        # simulated boltzmann fit with average parameters 
        ax.plot(vrange, self.mean_boltz_fit, 
                lw=1, ls='--', 
                c='r', label="\nVh=%.1f mV\ns=%.1f mV\nc=%.2f" % 
                (self.mean_gv_params[0], self.mean_gv_params[1], self.mean_gv_params[2])
                )
        
        ax.legend(loc='upper right')
        ax.set_xlabel("Voltage (mV)")        
        ax.set_ylabel("Normalized Conductance")
        
        if isinstance(title, str):
            ax.set_title(title + " `Steady-state` activation")
        else:
            ax.set_title("`Steady-state` activation")
        
        self.saver(name="GV", df=[ac_mu, ac_sem, "GV"], fig=f_ac)
    
    def exp(self, title=None, show=True, save=True):
        #exp params and delay
        df_xp = self.out_files["exp_params"]
        # print(df_xp.columns)
        
        f_xp = plt.figure(constrained_layout=True, figsize=(13,8))
        gs = f_xp.add_gridspec(2, 2)
        ax1 = f_xp.add_subplot(gs[0,:])
        ax2 = f_xp.add_subplot(gs[1,0])
        ax3 = f_xp.add_subplot(gs[1,1])
        
        ax1.set_ylabel(r"$\tau_{1}^{-1}$" + "\n " + r"$\tau_{f}^{-1}$", labelpad=15, fontsize=12)
        ax2.set_title(r"$\frac{A_f}{A_f + A_s}$")
        ax3.set_title("Delay (ms)")
        
        for a in [ax1, ax2, ax3]:
            a.set_xlabel("Voltage (mV)")
        
        # test voltages 
        vrange = [int(x) for x in df_xp.columns[1:]] 
        N = df_exp1.shape[1] 
        
        """ Extract fit parameters and compute statistics """
        # parameters from 1-exponential fit 
        df_exp1 = df.loc[1,:].iloc[:,1:] 
        taus_exp1 = [
            [1000/float(x[1]) for x in df_exp1.iloc[:,i].dropna().values if float(x[1]) > 0] 
            for i in range(N)
        ]
        taus_exp1 = pd.DataFrame.from_records(taus_exp1)
        taus_exp1.index = vrange 
        # mean and SEM 
        taus_exp1_mu, taus_exp1_err = get_mu_sem(taus_exp1)
        
        # extract fast and slow components for 2- and 3-exponential fits 
        df_exp2 = df.loc[2,:].iloc[:,1:] 
        FastTau_exp2, SlowTau_exp2, AmpProp_exp2 = extract_exp_params(df_exp2, order=2, prop=True)
        # compute stats 
        SlowTau_exp2_mu, SlowTau_exp2_err = get_mu_sem(SlowTau_exp2)
        FastTau_exp2_mu, FastTau_exp2_err = get_mu_sem(FastTau_exp2)
        
        if df.shape[0] > 2:
            df_exp3 = df.loc[3,:].iloc[:,1:].dropna(axis=1, how='all').dropna(axis=0, how='all')
            FastTau_exp3, SlowTau_exp3, AmpProp_exp3 = extract_exp_params(df_exp3, order=3, prop=False)
            FastTau_exp3_mu, FastTau_exp3_err = get_mu_sem(FastTau_exp3)
            SlowTau_exp3_mu, SlowTau_exp3_err = get_mu_sem(SlowTau_exp3)
        
        """ Plot taus """
        # twinx for slow time constant 
        ax1_slow = ax1.twinx()
        
        plot_exp_params(ax1, vrange, taus_exp1_mu, taus_exp1_err, order=1)
        
        plot_exp_params([ax1, ax1_slow], vrange, [FastTau_exp2_mu, SlowTau_exp2_mu], 
                        [FastTau_exp2_err, SlowTau_exp2_err], order=2)
        
        if df.shape[0] > 2:
            plot_exp_params([ax1, ax1_slow], vrange, [FastTau_exp3_mu, SlowTau_exp3_mu], 
                        [FastTau_exp3_err, SlowTau_exp3_err], order=3)
        
            ax1_slow.set_ylabel(r"$2^o$ and $3^o$, 1/$\tau_f$ ($s^{-1}$)")
        else:
            
            ax1_slow.set_ylabel(r"$\tau_{s}^{-1}$", labelpad=15, fontsize=12)
        
        # get legend and handles for previous data, then add label and handle for twin axes 
        h, l = ax1.get_legend_handles_labels()
        h_twin, l_twin = ax1_twin.get_legend_handles_labels() 
        h = h + h_twin 
        l = l + l_twin 
        # sort labels and handles by labels         
        by_label = OrderedDict(zip(l, h))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper center')

        """ Plot ratio of component amplitudes for exp2 fits """
        AmpProp_exp2_mu, AmpProp_exp2_err = get_mu_sem(AmpProp_exp2)
        props = dict(marker='s', markersize=5, lw=1, capsize=5)
        plot_exp_params(ax2, vrange, AmpProp_exp2_mu, AmpProp_exp2_err, props)
        
        # plot delay 
        df_delay = out_files["exp_delay"] 
        props = dict(marker='o', markersize=5, lw=1, c='white', capsize=5)
        Delay_mu, Delay_err = get_mu_sem(df_delay)
        plot_exp_params(ax3, vrange, Delay_mu, Delay_err, props)    

        if isinstance(title, str):
            ax1.set_title(title + " Exponential kinetics")
        else:
            ax1.set_title("Exponential kinetics")
        
        self.saver(fig=f_xp, title=title, name="exp", 
                df=[taus_exp1_mu, taus_exp1_err,
                    SlowTau_exp2_mu, SlowTau_exp2_err,
                    FastTau_exp2_mu, FastTau_exp2_err,
                    AmpProp_exp2_mu, AmpProp_exp2_err,
                    Delay_mu, Delay_err,
                    "Tau_exp1", "SlowTau_exp2", "FastTau_exp2", "AmpProp_exp2", "Delay_exp1"]
        )
