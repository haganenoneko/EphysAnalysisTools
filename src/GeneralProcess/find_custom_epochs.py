"""
Find and generate epochs from custom protocol file. 
"""
import pandas as pd 
import numpy as np 
import glob 
import os 
import math 
import matplotlib.pyplot as plt 

def MinMaxNorm(A):
    """
    Return (A - min(A))/(max(A) - min(A))
    """
    if isinstance(A, np.ndarray):
        min_A = np.amin(A)
        max_A = np.amax(A)
    elif isinstance(A, pd.DataFrame):
        min_A = A.min().min()
        max_A = A.max().max()
        
    return (A - min_A)/(max_A - min_A)

def GroupTimes(dt, dv, dvdt, ms=0.5, t_avg=None, min_t=10):
    """
    Find epochs by grouping timepoints with non-zero voltage time derivative by proximity
    
    `dt` = times with non-zero voltage derivative 
    `dv` = voltages with non-zero voltage derivative 
    `dvdt` = first time derivative of voltage protocol 
    `ms` = sampling time 
    `t_avg` = if used, specifies how times in `dt` (and voltages in `dv`) are grouped,
        i.e. t_avg <= max time - min time for a given grouping 
    `min_t` = minimum duration before epochs are recorded
        i.e. first epoch occurs at `t_0` where `t_0 > min_t` 
    """
    dtdt = dt[1:] - dt[:-1] 
    
    if t_avg is None:
        t_avg = np.mean(dtdt)
    
    # initialize lists for each epoch of each sweep 
    # list of epoch times 
    m = [[dt[0]]] if dt[0] > 0 else [[dt[1]]]
    # list of epoch voltages 
    v = [[dv[0]]] if dt[0] > 0 else [[dv[1]]]
    
    for i, t in enumerate(dt[1:]):                
        
        if t - m[-1][-1] < t_avg:
            if i+2 < len(dt) and dt[i+2] - t < 2*t_avg:
                m[-1].append(t)
                v[-1].append(dv[i+1])
            else:
                m.append([t])
                v.append([dv[i+1]])     
        else:
            m.append([t])
            v.append([dv[i+1]])     
    
    # reduce each epoch-th element from list of values to a single value 
    # for each grouping, pick minimum voltage if increasing and maximum voltage if decreasing 
    # pick first time index for each grouping 

    for i in range(len(m)):
        if abs(v[i][0] - v[i][-1]) >= 5:
            if v[i][0] < v[i][-1]:
                v[i] = int(max(v[i]))
            else:
                v[i] = int(min(v[i]))
        else:
            v[i] = v[i][0] 
        
        m[i] = m[i][0]
    
    # assume all epochs are steps for now 
    E = ['Step'] * len(m)
    
    # find ramp and ramp midpoints if present 
    N = len(m)
    for i, t in enumerate(m[:-1]):
        
        j = len(m) - N + 1
        
        # at least 50 samples
        if m[i+j] - t < 50:
            continue 
                
        if E[i+j-1] == 'Ramp':
            continue 

        # mean time derivative of voltage from `t` to next epoch is almost equal         
        if abs(dvdt[t]) > 1e-5:
        
            # midpoint between i and i+1 epoch 
            thalf = int((t + m[i+j]) / 2)
            vhalf = v[i+j-1] + sum(dvdt[t:thalf])*ms 

            if thalf-t < 50:
                continue 
                                    
            # plt.plot(dvdt[t:thalf])
            # plt.axhline(np.median(dvdt[t:thalf]))
            # plt.show()
                        
            # check for difference in direction of derivative 
            if dvdt[thalf-50] * dvdt[thalf+50] < 0 and vhalf - v[i+j-1] > 5:
                m.insert(i+1, thalf)
                v.insert(i+1, vhalf)
            
                # reassign epoch type
                E[i] = 'Ramp' 
                E.insert(i+j, 'Ramp')
                continue 
                    
            else: E[i+j-1] = 'Ramp'
    
    # enforce minimum time difference `avg`
    t0 = next(i for i, t in enumerate(m) if t > min_t)
    
    ts = [m[t0]]
    vs = [v[t0]]
    E2 = [E[t0]]

    for i, t in enumerate(m[t0+1:]):
        
        if (t - m[i]) > t_avg:
            if (t - m[i]) > 100:
                ts.append(t)
                vs.append(v[i+1])
                E2.append(E[i+1])
            
            # add epoch if type Ramp even if elapsed time from previous epoch is < 100
            elif i+1 < len(m) and E[i+1] == "Ramp":
                ts.append(t)
                vs.append(v[i+1])
                E2.append(E[i+1])

    # print("%d epochs found" % len(ts))    
    
    # voltages to integer multiples of 5 
    vs = [ round(round(v)/5)*5 for v in vs]
    
    return ts, vs, E2

def GenerateEpochs(pro, epath=None, save=False, dt=0, dv=0, show=False):
    """
    pro = dataframe for protocol, consisting of T x (N + 1) columns, with 1st column being time 
    epath = path of epochs file (save destination), only used if `save = True`
    save = whether output will be saved to CSV, requires `epath` 
    dv = minimum difference in voltage between two timepoints 
    dt = time window for computing difference
    
    Returns: 
        EpochTimes, EpochLevels, EpochTypes 
        Dictionaries of form {j : ...} where j is sweep number (0-indexed) and `...` is the data type
        
        EpochTimes[0] = list of epoch times (in units of sample numbers) for sweep 1 
        EpochLevels[0] = list of epoch levels (voltages, integer multiples of 5) for sweep 1 
        EpochTypes[0] = list of epoch types (strings, 'Ramp' or 'Step') for sweep 1 
    """
    
    if save:
        if epath is None:
            raise Exception("saving requires `epath` to be specified to act as destination for output file.")

    # sample rate 
    ms = pro.index[1] - pro.index[0]

    pro.reset_index(drop=True, inplace=True)
    
    # time window for first difference
    dt = int(dt/ms) if (dt > 0) else 1 
    # rate of change 
    dvdt = (pro.iloc[dt:, :].values - pro.iloc[:-dt, :].values)/ms 
    
    # boolean mask that is True when change is non zero 
    mask = dvdt != 0
        
    # output 
    EpochTimes = {}
    EpochLevels = {} 
    EpochTypes = {} 
    
    times = pro.index.values[1:]
    for j in range(mask.shape[1]):
        t_j = times[mask[:,j]]
        v_j = pro.iloc[1:, j].values[mask[:,j]]
        
        t_j, v_j, e_j = GroupTimes(t_j, v_j, dvdt[:,j], ms=ms)

        EpochTimes.update({j : np.array(t_j, dtype=int)})
        EpochLevels.update({j : np.array(v_j, dtype=float)})
        EpochTypes.update({j : e_j})
        
        if show:
            f, ax = plt.subplots(figsize=(12, 6))
            ax.plot(pro.iloc[:,j], alpha=0.5, lw=2, label=None, c='r')
            
            for i, t in enumerate(t_j):
                if e_j[i] == "Ramp":
                    ax.axvline(t, ls='--', c='white', lw=2, alpha=0.5)
                else:
                    ax.axvline(t, ls='--', c='r', lw=2, alpha=0.5)
            plt.show()
            
    if save and epath:
        # create a E x 3N dataframe where N = number of sweeps and E = number of epochs 
        df_list = [] 
        for k, v in EpochTimes.items():
            df = pd.DataFrame(data={
                "Time_%d" % k : v,
                "Voltage_%d" % k : EpochLevels[k],
                "Type_%d" % k : EpochTypes[k]
            })
            df_list.append(df)
            
        df_list = pd.concat(df_list, axis=1)
        df_list.to_csv(epath)
        print("Successfully saved automatically detected epochs to < %s > " % epath)
    
    return EpochTimes, EpochLevels, EpochTypes 
    
def ExtractEpochs(epochs):
    """
    epochs = dataframe from CSV file containing Time_i, Voltage_i, Type_i for epochs of ith sweep
    """
    N = int(epochs.shape[1]/3)
    EpochTimes = {}
    EpochLevels = {}
    EpochTypes = {}
    
    def Add2Dict(dic, col, n):
        if col == "Type": 
            val = epochs.loc[:, "%s_%d" % (col, n)].dropna().astype(str).values.tolist()
        else:
            val = epochs.loc[:, "%s_%d" % (col, n)].dropna().astype(int).values.tolist()
            
        dic.update({ n : val })
    
    for i in range(N):
        # EpochTimes.update({i : epochs.loc[:, "Time_%d" % i].dropna().values.tolist()})
        Add2Dict(EpochTimes, "Time", i)
        Add2Dict(EpochLevels, "Voltage", i)
        Add2Dict(EpochTypes, "Type", i)
        
    print("Epoch times, levels, and types successfully extracted from CSV.")
    return EpochTimes, EpochLevels, EpochTypes 
    
class DummyABF_Epochs():
    def __init__(self, levels=[], p1s=[], types=[]):
        self.levels = levels 
        self.p1s = p1s 
        self.types = types 
        
class DummyABF():
    def __init__(self, epochs, pro, df, khz):
        # EpochTimes, EpochLevels, EpochTypes = epochs 
        """
        `pro` and `df` may be dataframes or lists of lists 
            if latter, then `times` must also be a list, and dataframes will be constructed,
            also, `epochs` must be a pyABF object
        """
        # object with attributes matching pyABF for retrieving epoch properties 
        self.sweepEpochs = DummyABF_Epochs()
                
        if isinstance(pro, list) and isinstance(df, list):
            self.EpochTimes = [] 
            self.EpochLevels = [] 
            self.EpochTypes = [] 
            
            df = [] 
            pro = [] 
            for j in epochs.sweepList:
                epochs.setSweep(j)
                
                self.EpochTimes.append(np.array(epochs.sweepEpochs.p1s))
                self.EpochLevels.append(np.array(epochs.sweepEpochs.levels))
                self.EpochTypes.append(epochs.sweepEpochs.types)
                
                df.append(epochs.sweepY)
                pro.append(epochs.sweepC)
                                        
            self.data = pd.DataFrame(df).T
            self.pro = pd.DataFrame(pro).T
            
            self.data.index *= 1/khz 
            self.pro.index *= 1/khz                
            self.sweepX = epochs.sweepX
            
        else:
            self.data = df 
            self.pro = pro 
            self.sweepX = self.pro.index.values 
            self.EpochTimes, self.EpochLevels, self.EpochTypes = epochs 
        
        self.dataRate = int(khz)
        
        self.sweepC = self.pro.iloc[:,0].values 
        self.sweepY = self.data.iloc[:,0].values
        self.sweepList = list(range(self.pro.shape[1]))
        
    def setSweep(self, n):
        self.sweepC = self.pro.iloc[:, n].values
        self.sweepY = self.data.iloc[:, n].values
        self.sweepEpochs.levels = self.EpochLevels[n] 
        self.sweepEpochs.p1s = self.EpochTimes[n]
        self.sweepEpochs.types = self.EpochTypes[n]
        
class CleanProtocols():
    """
    'Clean up' voltage protocols by making sure transitions are discrete. 
    """
    def __init__(self, data, intervals, khz):
        
        self.N = int(data.shape[1]/2)
        self.data = data
        self.pro = data.iloc[:,N:] 
        self.intervals = intervals 
        self.khz = khz 
        
    def ConvertInt(self):
        """
        Convert self.pro to integer type
        """
        return self.pro.astype(int)
        
class FindCustomEpochs():
    def __init__(self, df, pname, fname, abf, save=False, show_epochs=True):
        """
        df = dataframe containing columns as time, N current, N voltage (N = # sweeps)
        pname = protocol name 
        fname = filename 
        abf = pyABF object 
        
        `show_epochs` = if an epochs CSV file was not found, shows automatically detected epochs found by `GenerateEpochs`
        """
        # protocol directory 
        pdir = r"./data/protocols/"
        # path to protocol, given `pname`
        ppath = pdir + "%s.csv" % pname 
        # path to protocol epochs corresponding to `pname`
        epath = pdir + "%s_epochs.csv" % pname 
        
        # sample rate 
        khz = int(1/(df.index[1] - df.index[0]))
                
        # check if `pname` specifies an available protocol 
        if os.path.isfile(ppath):
            print(" Found custom protocol for %s." % pname)
            pro = pd.read_csv(ppath, index_col=0, header=None)
            
            # make sure first row of protocol is whole number 
            pro.iloc[0,:] = pro.iat[0,0]
                        
            # append additional holding rows to start of `pro` if the number of rows don't match `df`
            if pro.shape[0] < df.shape[0]:
                
                n_ = df.shape[0] - pro.shape[0]
                print("Number of rows don't match. \n \
                    Default behaviour is to extend the protocol, because \
                    pre-trigger is normally accounted for during protocol creation. \n \
                    Adding %d rows of holding potential." % n_)
                
                # array of values to add to `pro`
                A = np.full((n_, df.shape[1]), int(pro.iloc[:10, 0].mean()), dtype=float)
                # convert to dataframe 
                A = pd.DataFrame(A) 
                A.columns = pro.columns[:df.shape[1]] 
                
                # concatenate row-wise with `pro`
                # pro = pd.concat([A, pro], axis=0).reset_index(drop=True)
                pro = pd.concat([pro, A], axis=0).reset_index(drop=True)
                pro.iloc[0,:] = pro.iat[0,0]
                
                if pro.shape[0] != df.shape[0]:
                    print("Shape of protocol: ", pro.shape)
                    print("Shape of data: ", df.shape)
                    raise Exception("After adding rows to protocol `pro` to account for pre-trigger duration, shape of `pro` and data `df` don't match.")
                else:
                    f, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
                
                    ax2 = ax.twinx()
                    
                    ax.plot(df, alpha=0.5, lw=2)
                    ax2.plot(pro, alpha=0.5)
                    
                    n_ *= df.index[1] - df.index[0]
                    ax.axvline(n_, ls='--', lw=3, label=n_)
                    
                    ax.legend(loc='upper right', fontsize=11)
                    
                    # plt.show()
                    plt.close()
                    
            else: n_ = 0
            
            if os.path.isfile(epath):
                print(" Found epochs for %s." % pname)                
                epochs = pd.read_csv(epath, index_col=0, header=0)
                
                # extract times, levels, and types of epochs for each sweep 
                epochs = ExtractEpochs(epochs) 
                
            else:
                print("   Could not find epochs for %s.")                            
                epochs = GenerateEpochs(pro, epath, save=save, show=show_epochs)
        
            # append protocol to data columnwise if missing from data 
            # protocol is missing from data `df` if:
            #       odd number of columns
            #       or, range of values is outside [-200, 100]
            if df.shape[1] % 2 > 0 or df.min().min() < -200 or df.max().max() > 100: 
                print("Data was detected to not contain protocol. Protocol will be added columnwise.")
                self.HasProtocol = False 
                
                df.reset_index(drop=True, inplace=True)
                
                if pro.shape[1] == df.shape[1]:
                    df = pd.concat([df, pro], axis=1)
                else:
                    df = pd.concat([df, pro.iloc[:,:df.shape[1]]], axis=1)
                                
                df.columns = range(df.shape[1])
                df.index *= 1/khz                 
                
            else:
                self.HasProtocol = True 
            
            self.khz = khz 
            self.df = df 
            self.pro = pro 
            self.epochs = epochs 
            self.abf = abf 
            
        else:
            raise Exception(" Custom protocol for %s could not be found.")
        
    def ReplaceABFVariables(self, test=False):
        """
        Replace variables in `abf` with class variables 
        """
        # EpochTimes, EpochLevels, EpochTypes = self.epochs 
        abf = DummyABF(self.epochs, self.pro, self.df, self.khz)
                
        # test setSweep
        if test:
            N = int(self.df.shape[1]/2)
            
            f, ax = plt.subplots()
            ax2 = ax.twinx()
            
            for j in abf.sweepList:
                if N + j >= self.df.shape[1]:
                    break 
                
                abf.setSweep(j)
                
                ax.plot(self.df.iloc[:,N+j], lw=2, alpha=0.4)
                # ax.plot(self.df.index, abf.sweepC, ls='--', alpha=0.7)
                
                # plot epochs
                EpochTimes, EpochTypes = self.epochs[0][j], self.epochs[2][j]

                for i, t in enumerate(EpochTimes):
                    t *= 1/self.khz
                    
                    ax2.scatter(t, self.epochs[1][j][i], s=50, alpha=0.8)            
                    
                    if EpochTypes[i] == 'Step':
                        ax.axvline(t, c='r', ls=':', lw=2, alpha=0.5)
                    elif EpochTypes[i] == 'Ramp':
                        ax.axvline(t, c='lightblue', ls=':', lw=2, alpha=0.8)
                    elif EpochTypes[i] == 'Sine':
                        ax.axvline(t, c='orange', ls=':', lw=2, alpha=0.8)
                
            ax.set_ylabel("Voltage (mV)")
            ax.set_xlabel("Time (samples)")
            ax.set_title("ReplaceABFVariables()")
            plt.show()
            # exit()
            
        return abf 
                