"""
main processing script
1. read files in data folder
2. get protocol name from ephys data spreadsheet
3. get voltage protocol from used protocols
4. normalize data 
5. assemble votlage protocol and normalized data in a single dataframe 
"""

import numpy as np 
import glob 
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import rcParams 
import matplotlib.gridspec as gridspec 
import matplotlib.patheffects as pe 
from scipy.optimize import curve_fit 
from scipy.integrate import simps 

cmap = plt.cm.get_cmap("gist_rainbow")

#boltzmann 
def boltz(v, imax, s, vh):
    return (imax/(1 + np.exp((v-vh)/s)))
    
#fit boltzmann 
def fit_boltz(v, o):
    p, c = curve_fit(boltz, v, o, p0=[1, 10, -85], bounds=([0.2, 2, -130], [2, 20, -40]))
    
    perr = np.sqrt(np.diag(c))
    return [p, perr] 

#monoexponential
def exp1(t, A, tau, c):
    return A*np.exp(-t/tau) + c 
#double exponential 
def exp2(t, A1, tau1, A2, tau2, c):
    return A1*np.exp(-t/tau1) + A2*np.exp(-t/tau2) + c
#fit exponential to test durations and ampltidues 
def fit_exp(t, y, method=1, i0=None):
    if method == 1:
        if isinstance(i0, float):
            p, c = curve_fit(exp1, t, y, 
                p0=[y[0], 1e4, i0], 
                bounds=([-np.inf, 100, i0-10], [np.inf, 1e6, i0+10])
                )
        else:
            p, c = curve_fit(exp1, t, y, 
                    p0=[y[0], 1e4, y[-1]], 
                    bounds=([-np.inf, 100, -np.inf], [np.inf, 1e6, np.inf])
                    )
    else:
        if isinstance(i0, float):
            p, c = curve_fit(exp2, t, y, 
                    p0=[y[0], 1e4, y[0], 1e3, i0], 
                    bounds=([-np.inf, 100, y[-1], 10, i0-10], 
                            [np.inf, 1e6, np.inf, 1e6, i0+10])
                    )
        else:
            p, c = curve_fit(exp2, t, y, 
                    p0=[y[0], 1e4, y[0], 1e3, y[-1]], 
                    bounds=([-np.inf, 100, y[-1], 10, -np.inf], 
                            [np.inf, 1e6, np.inf, 1e6, np.inf])
                    )
    e = np.sqrt(np.diag(c))
    return [p, e] 

#linear leak with current and voltage offset 
def ohmic_leak(v, g, v0):
    return g*v + v0 
    
#fit linear leak to given voltages and current 
def fit_leak(v, i):
    p, c = curve_fit(ohmic_leak, v, i, 
                # p0 = [10, -5, -10],
                p0 = [10, -5],
                # bounds = ([-20, -50, -50], [20, 50, 50])
                bounds = ([-20, -50], [20, 50])
                )
    
    perr = np.sqrt(np.diag(c))
    return [p, perr]
#get correlation cooefficient for linear fit of leak 
def get_r2(x, y, p):
    ysim = np.array([ohmic_leak(v, p[0], p[1]) for v in x])
    # y = np.array(y)
    
    # plt.plot(x, y, alpha=0.7)
    # plt.plot(x, ysim, lw=3, c='k')
    # plt.show()
    # exit()
    
    #residual sum of squares 
    ss_res = np.sum((y - ysim)**2)
    #total sum of squares 
    ss_tot = np.sum((y - np.mean(y))**2)
    
    #r_squared 
    r2 = 1 - (ss_res/ss_tot)
    return r2 

def get_tail_mins(df, khz, out_lists, show=False):
    """
    df = dataframe for tail or post-tail 
    out_lists = either [tail_peaks, tail_mins] or [tail_post_peaks] 
    """
    df_rolled_abs = df.rolling(15*khz).mean().abs().iloc[15*khz:]  
    # df_rolled_abs = df_rolled.abs() 
    
    # df_peak = df_rolled.iloc[df_rolled_abs.argmax()] 
    df_peak = df_rolled_abs.max() 
    
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(9,5))
        ax.plot(df.abs().reset_index(drop=True))         
        
        df_rolled_abs.reset_index(drop=True, inplace=True) 
        ax.plot(df_rolled_abs, c='k', lw=3, ls=':')
        
        ax.axhline(df_peak, ls='--', lw=3, c='r') 
        ax.axhline(df_min, ls='--', lw=3, c='k')
    
    if len(out_lists) > 1:
        out_lists[0].append(df_peak) 
        out_lists[1].append(df_rolled_abs.iloc[-10:].min() / df_peak)
        return out_lists 
    else:
        out_lists[0].append(df_peak) 
        return out_lists[0] 
    
def sep_de(df_sub, times=[0]*4, caps=[5]*3, khz=1, show=False): 
    """
    df_sub = leak_subtracted dataframe 
    
    t0, t1, t2, t3 = times 
        Each time is 0-indexed (i.e. real time)
        t0 = start of activation
        t1 = end of activation 
        t2 = end of deactivation
        t3 = end of post-pulse  
        
    ca, cd, cp = caps 
        capacitance for start of activating, deactivating, and post-deactivation ('post-pulse'), respectively, in time (ms) 
        
    khz = frequency of samples
    
    show = `True` or `False`; if True, plots separated segments (pre,test,post-pulses) 
    """
    
    ca, cd, cp = [c*khz for c in caps]     
    t0, t1, t2, t3 = times
    
    tail_mins = [] 
    tail_peaks = [] 
    tail_post_peaks = [] 
    act_list = [] 
    de_list = [] 
    
    if show:
        #close all active plots 
        plt.close('all') 
        
        f_temp, ax_temp = plt.subplots(1, 4, figsize=(12,5))
        ax_temp[0].set_title("Pre")
        ax_temp[1].set_title("Activation")
        ax_temp[2].set_title("Deactivation")
        ax_temp[3].set_title("Post")
    
    for j in range(df_sub.shape[1]):
        dfa = df_sub.iloc[t0+ca:t1-1*khz, j]
        dfd = df_sub.iloc[t1+cd:t2-2*khz, j]        
        dfp = df_sub.iloc[t2+cp:t3, j]

        #zero time index 
        dfa.index -= dfa.index[0] 
        dfd.index -= dfd.index[0] 
        dfp.index -= dfp.index[0] 
        
        tail_peaks, tail_mins = get_tail_mins(dfd, khz, [tail_peaks, tail_mins], 
                                            show=show)
        tail_post_peaks = get_tail_mins(dfp, khz, [tail_post_peaks], show=show) 
        
        act_list.append(dfa)
        de_list.append(dfd)
        
        if show:
            ax_temp[0].plot(df_sub.iloc[:t0+ca,j])
            ax_temp[1].plot(dfa)
            ax_temp[2].plot(dfd)
            ax_temp[3].plot(dfp) 
    
    if show:
        f_temp.tight_layout()
        plt.show() 
        exit() 
    else:
        return act_list, de_list, tail_mins, tail_peaks, tail_post_peaks 
    
#compute hysteresis for trace (without NaNs)
def get_hysteresis(trace, thalf, dt):
    """
    trace is the trace, without NaNs; numpy array 
    thalf = midpoint of ramp 
    dt = list of times
    """
    
    y1 = trace[:thalf]
    y2 = trace[thalf:]
    
    #integrate 
    #hyperpolarizing 
    ht_1 = simps(y1, dt)
    # ht_1 = simps(y1)
    
    #depolarizing, flip dt 
    # dt2 = np.flip(dt)
    
    ht_2 = simps(y2, dt) 
    # ht_2 = simps(y2) 
        
    return [ht_1, ht_2] 

def normalize(df, finf, pmin, khz, method='act', r_de=10, r_act=2):
    """
    r_de, r_act = rolling interval for de/activation 
    """
    # df is a series 
    y = df.dropna().abs()
    yr = y.rolling(r_act*khz).mean().iloc[r_act*khz:]
    
    if method == 'de':
        i0 = yr.iloc[-r_de*khz:].min()
        imax = yr.iloc[:r_de*khz].max()
    else:
        i0 = yr.iat[0]
        imax = yr.iat[-1]
        
    h = 0 
    # if abs(i0 - pmin) > 0.001:        
    while abs(i0 - pmin) > 0.001:        
        X = ((pmin*imax/finf) - i0)/((pmin/finf) - 1)
        y = finf*(y - X) / (imax - X)
        
        yr = y.rolling(r_act*khz).mean().iloc[r_act*khz:]
        if method == 'de':
            i0 = yr.iloc[-r_de*khz:].min()
            imax = yr.iloc[:r_de*khz].max()
        else:
            i0 = yr.iloc[:r_act*khz].min()
            imax = yr.iloc[-r_act*khz:].max()
                                    
        h += 1
        if h % 100 == 0:
            # plt.plot(df)
            # plt.show()
            print(h, pmin, finf, i0, imax, X, abs(i0 - pmin))            
            raise Exception("Normalization failed")
            
    return y

def norm_env(dfa_list, dfd_list, dfp_list, dfc_list, 
            khz=1, pmin=0.02, 
            voltage_protocol=[-120, 20, -120, 20]):
    """
    dfa_list, dfd_list, dfp_list = list of activation, deactivation, and 2nd activation pulses, respectively; isolated from full trace, capacitance removed, and leak-subtracted 
    
    1. Activation steady states will all become 1, initial current level will become pmin 
    2. Deactivation initial level will become 1, steady state will be 0.02, or divided by deactivation peak, whichever is larger 
    3. "Envelope" pulse will follow activation steady state (i.e. treat as if a direct continuation of activation pulse)
    4. All pulses are then concatenated  
    
    Some protocols don't have the same tail voltage. If this is the case, set:
        `same_tail_voltage=False.` 
        Then, the 2nd depolarizing pulse is not concatenated. 
        
    Voltage protocol is a list of voltages corresponding to activation, 1st tail, 2nd hpol, and 2nd tail 
    """
    print(voltage_protocol)
    if voltage_protocol[1] == voltage_protocol[3]:
        same_tail_voltage=True 
    else:
        same_tail_voltage=False 
    
    merge_list = [] 
    N = len(dfa_list)
    
    #in order to simulate later, we need to record the number of data points corresponding to each segment of the protocol, and access these after concatenation 
    #   to do this, get a list [# of data points]*N*S for each of N protocol segments for S sweeps 
    #   once done, we will make a string containing these values and name the respective column with it 
    protocol_shapes = [] 
    
    for j in range(N):
        dfa = dfa_list[j].copy() 
        dfd = dfd_list[j].copy()
        dfp = dfp_list[j].copy()
        dfc = dfc_list[j].copy() 

        ya_min = dfa.iloc[-1] 
        dfa *=  1/ya_min
        dfp *= 1/ya_min
        
        yd_peak = dfd.iloc[0] 
        dfc *= 1/yd_peak  
        dfd *= 1/yd_peak 
        
        # plt.plot(dfp, c='r')
        # plt.plot(dfa, c='k')
        # plt.plot(dfd, c='r', lw=2)
        # plt.plot(dfc, c='k', lw=2)
        
        ya_avg = dfa.rolling(5*khz).mean().iloc[5*khz:]
        yp_avg = dfp.rolling(5*khz).mean().iloc[5*khz:]
        yd_avg = dfd.rolling(5*khz).mean().iloc[5*khz:]
        # yc_avg = dfc.rolling(5*khz).mean().iloc[5*khz:]
        
        yd_peak = yd_avg.max() 
        yd_min = yd_avg.iloc[-2:].mean()  
        
        yd_inf = yd_min/yd_peak 
        if yd_inf < pmin:
            yd_inf = pmin 
        
        # print("yd_inf = ", yd_inf) 
        while (yd_min - yd_inf) > 0.01:
            Xd = (yd_min - yd_inf*yd_peak)/(1-yd_inf) 
            dfd = (dfd - Xd)/(yd_peak - Xd) 
            dfc = (dfc - Xd)/(yd_peak - Xd) 
            
            yd_avg = dfd.rolling(5*khz).mean().iloc[5*khz:]
            
            yd_peak = yd_avg.max()
            yd_min = yd_avg.iloc[-2:].mean() 
        
        # plt.plot(dfd, c='r', lw=2)
        
        #   use normalized peak tail (dfc) to set steady state of 2nd hpol pulse 
        yp_inf = dfc.rolling(5*khz).mean().max() 
        #   use steady state of 1st tail to set initial of 2nd hpol 
        yp_0 = dfd.rolling(5*khz).mean().min() 
        
        yp_init = yp_avg.iloc[0] 
        yp_final = yp_avg.iloc[-1] 
        
        # plt.plot(dfp, c='r')
        # plt.plot(dfa, c='k')
        # plt.plot(dfd, c='r', lw=2)
        # plt.plot(dfc, c='k', lw=2)
        
        ya_peak = ya_avg.max() 
        ya_min = ya_avg.min()
        if same_tail_voltage: 
            while (yp_init - yp_0) > 0.01 or (yp_final - yp_inf) > 0.01:
                Xp = ( (yp_init - ((yp_0/yp_inf)*yp_final))/(1 - (yp_0/yp_inf)) )
                dfp = yp_inf*(dfp - Xp)/(yp_final - Xp) 
                
                yp_avg = dfp.rolling(5*khz).mean().iloc[5*khz:]
                yp_init = yp_avg.iloc[0] 
                yp_final = yp_avg.iloc[-1] 
            
            while (ya_min - pmin) > 0.01:
                Xa = (ya_min - pmin*ya_peak)/(1-pmin)
                
                dfa = (dfa - Xa)/(ya_peak - Xa) 
                # dfp = (dfp - Xa)/(ya_peak - Xa) 
                
                ya_avg = dfa.rolling(5*khz).mean().iloc[5*khz:]
                # yp_avg = dfp.rolling(5*khz).mean().iloc[5*khz:]
                
                ya_peak = ya_avg.max()
                ya_min = ya_avg.min()
        else:
            while (ya_min - pmin) > 0.01:
                Xa = (ya_min - pmin*ya_peak)/(1-pmin)
                
                dfa = (dfa - Xa)/(ya_peak - Xa) 
                dfp = (dfp - Xa)/(ya_peak - Xa) 
                
                ya_avg = dfa.rolling(5*khz).mean().iloc[5*khz:]
                # yp_avg = dfp.rolling(5*khz).mean().iloc[5*khz:]
                
                ya_peak = ya_avg.max()
                ya_min = ya_avg.min()
        
        # plt.plot(dfp, c='r')
        # plt.plot(dfa, c='k')
        
        if same_tail_voltage == True:
            df_merge = pd.concat([dfa, dfd, dfp, dfc], axis=0)
            protocol_shapes.append([dfa.shape[0], dfd.shape[0], dfp.shape[0], dfc.shape[0]])
        else:
            protocol_shapes.append([dfa.shape[0], dfd.shape[0], dfp.shape[0]])
            df_merge = pd.concat([dfa, dfd, dfp], axis=0)
        
        df_merge.reset_index(drop=True, inplace=True) 
        merge_list.append(df_merge) 

    df_merge = pd.concat(merge_list, axis=1).apply(lambda x: pd.Series(x.dropna().values).fillna(''))
    df_merge.reset_index(drop=True, inplace=True)
    
    if len(protocol_shapes[0]) > 3:
        df_merge.columns = ["%d_%d_%d_%d" % (x[0], x[1], x[2], x[3]) for x in protocol_shapes]
    else:
        df_merge.columns = ["%d_%d_%d" % (x[0], x[1], x[2]) for x in protocol_shapes]
    
    if khz > 1:
        df_merge.index *= 1/khz 
    
    # plt.plot(df_merge, lw=2)
    # plt.show()
    # exit()
    
    return df_merge 
        
class process():
    def __init__(self, dates, protocol_name, dates_to_save='None', show_protocols=True):
        """
        dates = list of dates in ephys_info.xlsx, e.g. 20916, 209, etc. 
        protocol_name = protocols named in used_protocols_v#.csv
        show_protocols = prints named protocols recorded in `dates` 
        """
        #ephys data info 
        ephys_info = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/records/ephys_data_info.xlsx"
        ephys_info = pd.read_excel(ephys_info, header=0, index_col=None)
        #convert Files column into string 
        ephys_info = ephys_info.astype({'Files' : 'str'})
        
        #load files 
        mask = [x[:5] in dates for x in ephys_info['Files'].values]
        ephys_info = ephys_info[mask].reset_index(drop=True)
        print(ephys_info['Protocol'])
        
        #load protocol 
        ephys_info = ephys_info.loc[ephys_info['Protocol'] == protocol_name]         
        #names of files 
        filenames = ephys_info['Files'].values.tolist()
        
        if len(filenames) < 1:
            print("No files found for protocol: %s " % protocol_name)
            exit()
        
        #add directory to filenames 
        data_path = "C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/Github_repos/hcn-gating-kinetics/test/Sept2020/data/"
        paths = [data_path + "%s.csv" % f for f in filenames]

        #read in data files as dataframes 
        # for x in paths:
            # try:
                # print(pd.read_csv(x, header=None, index_col=0).head())
            # except:
                # print(x)
        # exit()        
        data_files = [pd.read_csv(x, header=None, index_col=0) for x in paths if os.path.exists(x)]   
        
        #read in protocol csv as dataframe 
        pro_file = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/Github_repos/hcn-gating-kinetics/test/Sept2020/data/used_protocols_v1.csv"
        pro = pd.read_csv(pro_file, header=0, index_col=None)

        #unpack protocol 
        pro = pro.loc[pro['Protocol'] == protocol_name]
        # print(pro)
        # exit()
                
        #separate steps in voltage protocol 
        khz = pro['khz'].values[0]
        #separate voltage commands in protocol 
        pro = pro.iloc[0,:].dropna().values 
        steps = [pro[j:j+3] for j in range(3, pro.shape[0], 3)]
        
        #assign self variables for class 
        self.protocol_name = protocol_name 
        self.data_files = data_files
        self.filenames = filenames 
        self.steps = steps 
        self.khz = khz 
        self.dates_to_save = dates_to_save
        
        self.save_path = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/Github_repos/hcn-gating-kinetics/test/Sept2020/"
        
    def env(self):
        
        if len(self.data_files) < 1:
            print("No data files found!")
            exit()
        
        #assuming no delay, go through steps and find start and end of initial leak ramp                 
        if "FA" in self.protocol_name:
            t_i = (1436-500)*self.khz
        else:
            t_i = (906-500)*self.khz                         
            
        for i in range(len(self.steps)):
            t, v, x = self.steps[i] 
                
            if x == 'r':
                t_ramp = [t_i, t_i + (2*t), t_i + t]
                
                #prepulse
                pre, test, post, post_de = self.steps[i+3:i+7]
                #start of prepulse 
                t_on = t_i + 2*t + int(self.steps[i+2][0])  
                test_protocol = [pre, test, post, post_de, t_on] 
                break 
            else:
                try:
                    t_i += t 
                except:
                    print(t)
                    
                continue 

        # print(t_ramp)
        # exit()

        #end of prepulse
        t1 = t_on + int(pre[0])

        x = test[-1].split(", ")
        if x[0] != 'l':
            raise Exception("Test type is not a list")
            exit()
            
        #unpack test durations 
        t2 = [int(t) for t in test[0].split(", ")]
        #end of deactivating pulse 
        t2 = [t1 + t for t in t2] 
        #end of 2nd hyperpolarizing pulse ('envelope pulse')
        t3 = [int(post[0]) + t for t in t2] 
        # end of 2nd depolarizing pulse (after envelope) 
        t4 = [int(test_protocol[3][0]) + t for t in t3]        

        test_protocol.extend([t1, t2, t3, t4])
        # print(test_protocol)
        # exit()
        
        pmin = 0.02
        for i in range(len(self.data_files)):
            if self.filenames[i] == "20903005":
                #a problem in the recording causes the 40s step to be < 40s
                #2nd hpol starts ~40705 instead of 47935
                dt = 47935 - 40705
                t2[0] -= dt 
                t3[0] -= dt   
                t4[0] -= dt               
            
            print("Processing %s..." % self.filenames[i]) 
            
            df = self.data_files[i] 
            N = int(df.shape[1]/2)       
            
            sigma = np.std(df.iloc[10*self.khz:110*self.khz:10,:N], axis=0)
            print("Mean std...", np.mean(sigma))
            # print(df.head)
            # exit()
            
            #set default color cycle 
            rcParams['axes.prop_cycle'] = plt.cycler(plt.cycler('color', cmap(np.linspace(0, 1, N+1))))
            
            if self.filenames[i] == "20903005":
                print("For %s, voltage protocol needs to be corrected. Correcting....")
                #recompute protocol for first trace to ensure appropriate leak subtraction 
                vpro1 = df.iloc[:,N].values 
                t2, t3 = test_protocol[-3:-1]    
                #2nd hyperpolarizing pulse 
                vpro1[t2[0]-1:t3[0]-1] = int(test_protocol[0][1])
                #duration of post de             
                t4, v4, x = test_protocol[3]   
                t4 = int(t4 + t3[0] - 1)
                
                vpro1[t3[0]-1:t4] = int(v4) 
                #set remaining to holding 
                vpro1[t4:] = int(self.steps[0][1])
                #reassign corrected protocol 
                df.iloc[:,N] = vpro1 
            
            #check raw data 
            # plt.plot(df.iloc[:,N:], lw=3)
            # plt.legend(range(N), loc='upper right')
            # plt.plot(df.iloc[:,:N])
            # plt.show()
            # plt.close()
            # exit()
                    
            #final figure for output 
            #create figure 
            f_fnl = plt.figure(constrained_layout=True, figsize=(10, 12))
            #create gridspec
            gs = f_fnl.add_gridspec(6, 6)
                
            f_ax1 = f_fnl.add_subplot(gs[:2, :])
            f_ax2a = f_fnl.add_subplot(gs[2:4, :4])
            f_ax2b = f_fnl.add_subplot(gs[2:4, 4:])
            f_ax3a = f_fnl.add_subplot(gs[4, :2])
            f_ax3b = f_fnl.add_subplot(gs[4, 2:4])
            f_ax3c = f_fnl.add_subplot(gs[4, 4:])
            f_ax4a = f_fnl.add_subplot(gs[-1, :3])
            f_ax4b = f_fnl.add_subplot(gs[-1, 3:])    
            
            #time of initial ramp 
            t1, t2, thalf = t_ramp         
            t1 -= 1
            t2 -= 1
            thalf -= t1 + 1 
            
            # # f, ax = plt.subplots()
            # f, ax = plt.subplots(2, 1)
            #check ramp 
            # ax[0].plot(df.index[t1:t2], df.iloc[t1:t2,:N])
            # ax[1].plot(df.index[t1:t2], df.iloc[t1:t2,N:])
            # plt.show()
            # exit()
            
            #plot raw data nad protocol 
            f_ax4a.plot(df.index.values, df.iloc[:,:N],
                    lw=2)
            f_ax4b.plot(df.index.values, df.iloc[:,N:],
                    lw=2)
            
            #leak subtracted traces 
            subtracted = [0]*N 
            
            #get initial leak ramp current and voltages 
            fit_params = [0]*N 
            for j in range(N): 
                clr = cmap(j/N)
                
                current = df.iloc[t1:t2,j].values        
                volts = df.iloc[t1:t2,N+j].values

                #average current in both arms
                i_mu = (current[:thalf] + np.flip(current[thalf:]))/2
                v_mu = (volts[:thalf] + np.flip(volts[thalf:]))/2
                        
                #fit averaged ramp 
                p, e = fit_leak(v_mu, i_mu)
                
                #plot average ramp current 
                f_ax3a.plot(v_mu, i_mu, c=clr, alpha=0.5, lw=2)
                #compute ohmic leak 
                iL = [ohmic_leak(v, p[0], p[1]) for v in v_mu]
                #plot ohmic leak (fitted) with a black border 
                f_ax3a.plot(v_mu, iL, c=clr, lw=2,
                    path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()]
                    )
                
                #get r2 value for fit 
                r2_j = get_r2(v_mu, i_mu, p)
                
                #conductance, Erev, r2 
                fit_params[j] = [p[0], -p[1]/p[0], r2_j]
                        
                #subtract leak from jth trace 
                i_leak = [ohmic_leak(v, p[0], p[1]) for v in df.iloc[:,N+j]]
                sub = df.iloc[:,j] - np.array(i_leak)
           
                subtracted[j] = sub
                
            #plot fitted leak parameters 
            fit_params_labels = [r"$\gamma$", r"$E_{rev}$", r"$r^2$"]
            
                        
            #use twinx for r2 
            f_ax3c_x2 = f_ax3c.twinx()
            
            #average fitted leak params 
            leak_params_mu = np.mean(fit_params, axis=0)
            
            for k in range(len(fit_params[0])):        
                clr = cmap(k/len(fit_params[0]))
                
                if k == 0:
                    f_ax3c.plot(range(1, N+1), 
                            [x[k] for x in fit_params],
                            marker='o', markersize=4, c=clr,
                            ls='none', markeredgewidth=1, 
                            label = "%s=%.1f" % (fit_params_labels[k], leak_params_mu[k])
                            )
                elif k == 1:
                    f_ax3b.plot(range(1,N+1), 
                            [x[k] for x in fit_params],
                            marker='o', markersize=4, c=clr,
                            ls='none', markeredgewidth=1, 
                            label = "%s=%.1f" % (fit_params_labels[k], leak_params_mu[k])
                            )
                else:
                    f_ax3c_x2.plot(range(1, N+1), 
                            [x[k] for x in fit_params],
                            marker='x', markersize=6, c='k',
                            ls='none', markeredgewidth=1,
                            label = "%s=%.1f" % (fit_params_labels[k], leak_params_mu[k])
                            )
                
            #concatenate subtracted traces into dataframe
            df_sub = pd.concat(subtracted, axis=1)
            #scale index values to self.khz 
            # if 1/(df_sub.index[1] - df_sub.index[0]) != self.khz:
            df_sub.index = df.index 
            
            #plot leak-subtracted traces 
            f_ax2a.plot(df_sub.index.values, df_sub, lw=2)

            #tail peaks for activation curve 
            pre_peaks = [] 
            tail_peaks = [] 
            
            #take only test pulse and deactivation
            ca = 7*self.khz #capacitance, activation 
            ca2 = 0 # end of hpol trim 
            cd = 11*self.khz #capacticance, deactivation 
            cd2 = 0 #end of depol 
            cp = [0]*N 
            cp2 = [0] # end of post pulse  
            
            if 'FA' in self.protocol_name:
                if '20903' in self.filenames[i]:                
                    cp = [8*self.khz]*N 
                    cp[0] = 5*self.khz
                    cp[-1] = 100*self.khz
                else:
                    cp = [x*self.khz for x in [2, 8, 8, 15, 15]]  #capacitance, post pulse 
            else:
                cp = [3*self.khz]*N 
                cd = 14*self.khz 
                cd2 = 3*self.khz 
                ca = 2*self.khz 
                ca2 = 3*self.khz 
                cp2 = 3*self.khz 

            #start of prepulse, end of prepulse, end of tail 
            t0, t1, t2, t3, t4 = test_protocol[-5:]
            t0 -= 1 #start of prepulse 
            t1 -= 1 #end of prepulse/start of depolarization
            t2 = [t-1 for t in t2] #end of depolarization 
            t3 = [t-1 for t in t3] #start of 'envelope' pulse 
            t4 = [t-1 for t in t4] 
            
            #separate act, de, and post pulses 
            act_list = [] 
            de_list = [] 
            env_list = [] 
            closed_list = [] 
            
            # plot separated protocol segments 
            # f, ax = plt.subplots(1,4,figsize=(14,4))
            for j in range(N):
                dfa = df_sub.iloc[t0+ca:t1-ca2, j]

                # if duration given by (end of depol) - (end of hpol) > 0 
                if t2[j] - (t1+cd) > 0:
                    dfd = df_sub.iloc[t1+cd:t2[j]-cd2, j]
                else:
                    dfd = df_sub.iloc[t1:t2[j]-cd2, j] 
                    ax[0].plot(df_sub.iloc[:, j])
                    print(t0, t1, [t-t1 for t in t2])
                    exit()
                    
                # print(t2[j], cp[j], t3[j], df_sub.shape[0])
                dfp = df_sub.iloc[t2[j]+cp[j]:t3[j]-cp2, j]
                dfc = df_sub.iloc[t3[j]+cd:t4[j]-cd2, j] 
                
                try:
                    #zero the time index         
                    dfa.index -= dfa.index[0] 
                    dfd.index -= dfd.index[0] 
                    dfp.index -= dfp.index[0]
                    dfc.index -= dfc.index[0] 
                except:
                    print(dfa.head)
                    print(dfd.head)
                    print(dfp.head)
                    continue 
                    # exit()
                
                # ax[0].plot(dfa)
                # ax[1].plot(dfd)
                # ax[2].plot(dfp)
                # ax[3].plot(dfc) 
                # plt.show()
                # exit()
                
                #append peak and initial hpol currents to respecitve lists
                # use last/first 2ms 
                y1 = dfa.rolling(2*self.khz).mean().iloc[2:] 
                y2 = dfp.rolling(2*self.khz).mean().iloc[2:] 
                pre_peaks.append(y1.max()) # 1st hpol peak 
                tail_peaks.append(y2.max()) # 2nd hpol peak 
                
                act_list.append(dfa)
                de_list.append(dfd) 
                env_list.append(dfp)
                closed_list.append(dfc) 
            
            # plt.show()
            # exit()
            
            #list of deactivation durations 
            #   where t2 is end of depolarization 
            durations = [t - t1 for t in t2][:N] 
            # print(durations)

            #   subtract corresponding initial and post pulse peaks to get delta values 
            #   to fit, set steady-state to 0 
            deltas = [a-b for (a, b) in zip(tail_peaks, pre_peaks)]
            
            try:
                f_ax2b.plot(durations, deltas, 
                            marker='o', markersize=8, ls='none')
            except:
                print("Number of ramp durations != number of deltas")
                print("Durations = ", durations, 
                      "\n Deltas = %d, \n Number of traces = %d, \n DataFrame columns = %d" % 
                      (len(deltas), N, df.shape[1])
                )
                exit()
            
            #alternatively, fit exponentials using mean initial peak value as steady-state 
            pre_peak_mu = np.mean(pre_peaks)
            #timerange for simulating exponential fits 
            if "FA" in self.protocol_name:
                Tsim = 50000
            else:
                Tsim = 20000
            
            t0 = int((t1+cd)/self.khz) # end of hpol, in time rather than samples
            tsim = range(t0, Tsim + t0)
            try:
                #exp1 
                p1, e1 = fit_exp(durations, tail_peaks, method=1, i0=pre_peak_mu)
                fit1 = [exp1(d, p1[0], p1[1], p1[2]) for d in range(Tsim)] 
                f_ax2a.plot(tsim, fit1, c='k', lw=3, ls='dotted', alpha=0.7, 
                        label=r"$\tau_1$=%.1e ms" % int(p1[1]))
                
                p1, e1 = fit_exp(durations, deltas, method=1, i0=0.0)
                fit1 = [exp1(d, p1[0], p1[1], p1[2]) for d in range(Tsim)] 
                f_ax2b.plot(range(Tsim), fit1, c='k', lw=3, ls='dotted', alpha=0.7, 
                        label=r"$\tau_1$=%.1e ms" % int(p1[1]))
            except:
                pass 
            try:
                #exp2
                p2, e2 = fit_exp(durations, tail_peaks, method=2, i0=pre_peak_mu)
                fit2 = [exp2(d, p2[0], p2[1], p2[2], p2[3], p2[4]) for d in range(Tsim)]
                f_ax2a.plot(tsim, fit2, c='k', lw=3, ls='-', alpha=0.5,
                        label=r"$\tau_2$=%.1e ms" % int(max([p2[1], p2[3]])))
                        
                p2, e2 = fit_exp(durations, tail_peaks, method=2, i0=0.0)
                fit2 = [exp2(d, p2[0], p2[1], p2[2], p2[3], p2[4]) for d in range(Tsim)]
                f_ax2b.plot(range(Tsim), fit2, c='k', lw=3, ls='-', alpha=0.5, 
                        label=r"$\tau_2$=%.1e ms" % int(max([p2[1], p2[3]])))
            except:
                pass 
            # plt.show()
            # exit()   
            
            # find out whether the 1st and 2nd depolarizations are same voltage
            voltage_protocol = [int(x[1]) for x in test_protocol[:4]] 
            voltage_protocol[1] = int(df.iat[t2[1]-100,N+1])
            #normalize 
            df_norm = norm_env(act_list, de_list, env_list, closed_list,
                                khz=self.khz, pmin=0.02, 
                                voltage_protocol=voltage_protocol 
                                )
            
            #legend columns 
            if N <= 4:
                ncol = 2
            elif 4 < N <= 6:
                ncol = 3 
            else:
                ncol = 4
            
            #add durations to xticklabel of f_ax3b-c 
            new_labels = ["%d\n%.1f" % (j, int(durations[j-1]/1000)) for j in range(1, N+1)]
            for j in range(len(new_labels)):
                if durations[j] < 1000:
                    new_labels[j] = "%d\n%.2f" % (j, durations[j]/1000)
                    
            df_norm.columns = ["%d_%s" % (durations[j], df_norm.columns[j]) for j in range(df_norm.shape[1])] 

            f_ax1.plot(df_norm)
            f_ax1.legend(new_labels,
                        loc='lower left', title="Durations (s)", 
                        edgecolor='k', frameon=True, ncol=ncol,
                        labelspacing=0.4, columnspacing=1.,
                        handletextpad=0.6, fontsize=10, framealpha=0.5)
                            
            #axes labels, legends 
            f_ax1.set_title(self.filenames[i])    
            # f_ax1.legend(df_act.columns, loc='lower rig
            f_ax1.set_xlabel("Time (ms)")
            f_ax1.set_ylabel(r"$P_{open}") 
            
            f_ax2a.set_xlabel("Time (ms)")
            f_ax2a.set_ylabel("Leak- \nsubtracted pA")
            f_ax2a.legend(loc='lower right',
                        ncol=1, edgecolor='k',
                        labelspacing=0.4, handletextpad=0.6, columnspacing=1.5,
                        fontsize=10, framealpha=0.5)
            
            f_ax2b.set_xlabel("Duration (ms)")
            f_ax2b.set_ylabel(r"Peak Current, pA") 
            f_ax2b.legend(loc='lower right',
                        ncol=1, edgecolor='k', frameon=True,
                        labelspacing=0.4, handletextpad=0.6, columnspacing=1.5,
                        fontsize=10, framealpha=0.5)
                
            f_ax3a.set_ylabel(r"Fitted Leak, pA")
            f_ax3b.set_ylabel(r"$E_{rev}$")
            f_ax3c.set_ylabel(r"pS")
            f_ax3c_x2.set_ylabel(r"$r^2$")
            f_ax3b.set_xlabel(r"Sweep # / Duration (s)")
            f_ax3c.set_xlabel(r"Sweep # / Duration (s)")

            #add durations to xticklabel of f_ax3b-c 
            f_ax3b.set_xticks(range(1, N+1))
            f_ax3c.set_xticks(range(1, N+1))
            f_ax3b.set_xticklabels(new_labels)
            f_ax3c.set_xticklabels(new_labels)
            
            #set ylim to 0.5 - 1 
            f_ax3c_x2.set_ylim(bottom=0.4, top=1.05)
            
            #legends for fitted leak params     
            f_ax3b.legend(loc='lower right', 
                        edgecolor='k', labelspacing=0.2,
                        handletextpad=0.1, fontsize=10, framealpha=0.5)        
            h1, l1 = f_ax3c.get_legend_handles_labels()
            h2, l2 = f_ax3c_x2.get_legend_handles_labels()
            f_ax3c.legend(h1+h2, l1+l2, loc='lower right',
                        edgecolor='k', frameon=True, 
                        labelspacing=0.4, columnspacing=1.,
                        handletextpad=0.6, fontsize=10, framealpha=0.5)
                
            f_ax4a.set_ylabel("Raw, pA")
            f_ax4b.set_ylabel("mV")
            f_ax4a.set_xlabel("Time (ms)")
            f_ax4b.set_xlabel("Time (ms)")
                
            # plt.tight_layout()
            plt.show()
            plt.close(f_fnl)
            
            if self.save_leaksub:
                df_sub.columns = durations 

                #add voltage to subtracted dataframe 
                df_sub = pd.concat([df_sub, df.iloc[:,N:]], axis=1) 
                df_sub.to_csv(self.save_path + r"normalized_data/leaksub/%s_leaksub.csv" % self.filenames[i])
                
            if self.all_output:
                f_fnl.savefig(self.save_path + r"process_output/%s.png" % self.filenames[i])
                
                if self.filenames[i] in self.dates_to_save:
                    df_norm.to_csv(self.save_path + r"normalized_data/%s_env.csv" % self.filenames[i])
                
            # exit()
            
    def act(self):
        #assuming no delay, go through self.steps and find start and end of initial leak ramp 
        t_ramp = []
        
        if 'FA' in self.protocol_name:
            t_i = (1306-500)*self.khz
        else:
            t_i = (905-500)*self.khz 
        
        for i in range(len(self.steps)):
            t, v, x = self.steps[i] 
                
            if x == 'r':
                t_ramp = [t_i, t_i + (2*t), t_i + t]
                
                #test voltages follow one step after the initial ramp, so just take the step
                t_test = self.steps[i+3].tolist()
                #append start time of test step, which is current time + durations of current and next self.steps 
                t_on = t_i + t + self.steps[i+1][0] + self.steps[i+2][0]
                t_test.append(t_on) 
                #append tail current duration 
                # print(self.steps[i+2:i+6])
                # exit()
                t_off = int(self.steps[i+4][0])
                t_test.append(t_off) 
                
                break 
            else:
                try:
                    t_i += t 
                except:
                    print(t)
                    
                continue 

        # print(t_ramp)
        # exit()

        #get list of start and end times for test pulses 
        dt, v, x, t0, t_tail = t_test 
        #list of test pulse durations 
        s = [int(a) for a in dt.split(", ")]
        #add t0 to each duration to get end times 
        if self.filenames[i] == '20917008':
            t2 = [t0 + a - 2*self.khz for a in s]
        else:
            t2 = [t0 + a for a in s] 
            
        #add tail duration to get end of tail 
        t3 = [a + int(t_tail) for a in t2] 

        #unpack step type 
        x = x.split(", ")
        if len(x) > 1:
            dT = x[1]
            if len(x) > 2:
                dv = x[2]
        x = x[0] 

        if x != 'l':
            raise Exception("test pulse step type is not list...")

        #create list of voltage levels for each test pulse
        test_volts = [v + i*int(dv) for i in range(len(t2))]

        # print(t2, t3, test_volts)
        test_protocol = [int(t0), t2, t3, test_volts] 
        # print(test_protocol)
        # exit()    
   
        pmin = 0.02
        for i in range(len(self.data_files)):            
            df = self.data_files[i]      
            N = int(df.shape[1]/2)        
                                    
            if self.filenames[i] in ['20917008', '20910002']:
                print("Modifying protocol to remove -145mV trace. \n Subsequent data files will raise errors because of this. \n Alternatively, process '20917008' separately. Currently, processing will exit after processing this file.")
                df.columns = list(range(df.shape[1]))
                df.drop(df.columns[[0, 7]], axis=1, inplace=True)
                
                #modify protocol 
                test_protocol[1] = test_protocol[1][1:] 
                test_protocol[2] = test_protocol[2][1:]
                test_protocol[-1] = test_protocol[-1][1:]
                
                N -= 1
            
            #set default color cycle 
            rcParams['axes.prop_cycle'] = plt.cycler(plt.cycler('color', cmap(np.linspace(0, 1, N+1))))
            
            #create figure and gridspec 
            f_fnl = plt.figure(constrained_layout=True, figsize=(10, 12))
            #create gridspec
            gs = f_fnl.add_gridspec(5, 6)
            f_ax1 = f_fnl.add_subplot(gs[0, :])
            f_ax2 = f_fnl.add_subplot(gs[1, :])
            f_ax3a = f_fnl.add_subplot(gs[2, :-2])
            f_ax3b = f_fnl.add_subplot(gs[2, -2:])
            f_ax4a = f_fnl.add_subplot(gs[3, :2])
            f_ax4b = f_fnl.add_subplot(gs[3, 2:4])
            f_ax4c = f_fnl.add_subplot(gs[3, 4:])
            f_ax5a = f_fnl.add_subplot(gs[-1, :3])
            f_ax5b = f_fnl.add_subplot(gs[-1, 3:])
            
            #time of initial ramp 
            t1, t2, thalf = t_ramp         
            t1 -= 1
            t2 -= 1
            thalf -= t1 + 1 
            
            f, ax = plt.subplots()
            # f, ax = plt.subplots(2, 1)
            # ax[0].plot(df.index[t1:t2], df.iloc[t1:t2,:N])
            # ax[1].plot(df.index[0:t2], df.iloc[0:t2,N:], alpha=0.5)
            # ax[1].plot(df.index[t1:t2], df.iloc[t1:t2,N:])
            # plt.show()
            # exit()
            
            #plot raw data nad protocol 
            f_ax5a.plot(df.index.values, df.iloc[:,:N],
                    lw=2)
            f_ax5b.plot(df.index.values, df.iloc[:,N:],
                    lw=2)
            
            #leak subtracted traces 
            subtracted = [0]*N 
            
            #get initial ramp current and voltages 
            fit_params = [0]*N 
            for j in range(N): 
                clr = cmap(j/N)
                
                current = df.iloc[t1:t2,j].values        
                volts = df.iloc[t1:t2,N+j].values

                #average current in both arms
                i_mu = (current[:thalf] + np.flip(current[thalf:]))/2
                v_mu = (volts[:thalf] + np.flip(volts[thalf:]))/2
                        
                #fit averaged ramp 
                p, e = fit_leak(v_mu, i_mu)
                
                #plot average ramp current 
                f_ax4a.plot(v_mu, i_mu, c=clr, alpha=0.8)
                #compute ohmic leak 
                iL = [ohmic_leak(v, p[0], p[1]) for v in v_mu]
                #plot ohmic leak (fitted) with a black border 
                f_ax4a.plot(v_mu, iL, c=clr, lw=2,
                    path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()]
                    )
                
                #get r2 value for fit 
                r2_j = get_r2(v_mu, i_mu, p)
                
                # fit_params[j] = [p[0], p[1], r2_j]
                fit_params[j] = [p[0], -p[1]/p[0], r2_j]
                
                #subtract leak from jth trace 
                i_leak = [ohmic_leak(v, p[0], p[1]) for v in df.iloc[:,N+j]]
                sub = df.iloc[:,j] - np.array(i_leak)

                subtracted[j] = sub
                
            #plot fitted leak parameters 
            fit_params_labels = [r"$\gamma$", r"$E_{rev}$", r"$r^2$"]
            
            #use twinx for r2 
            f_ax4c_x2 = f_ax4c.twinx()
            
            #average fitted leak params 
            leak_params_mu = np.mean(fit_params, axis=0)
            
            #plot fit params 
            for k in range(len(fit_params[0])):        
                clr = cmap(k/len(fit_params[0]))
                
                if k == 0:
                    f_ax4c.plot(range(1, N+1), 
                            [x[k] for x in fit_params],
                            marker='o', markersize=4, c=clr,
                            ls='none', markeredgewidth=1, 
                            label = "%s=%.1f" % (fit_params_labels[k], leak_params_mu[k])
                            )
                elif k == 1:
                    f_ax4b.plot(range(1,N+1), 
                            [x[k] for x in fit_params],
                            marker='o', markersize=4, c=clr,
                            ls='none', markeredgewidth=1, 
                            label = "%s=%.1f" % (fit_params_labels[k], leak_params_mu[k])
                            )
                else:
                    f_ax4c_x2.plot(range(1, N+1), 
                            [x[k] for x in fit_params],
                            marker='x', markersize=6, c='k',
                            ls='none', markeredgewidth=1,
                            label = "%s=%.1f" % (fit_params_labels[k], leak_params_mu[k])
                            )
                            
            #concatenate subtracted traces into dataframe
            df_sub = pd.concat(subtracted, axis=1)
            #scale index values to self.khz 
            df_sub.index = df.index.values 
            #plot leak-subtracted traces 
            f_ax3a.plot(df_sub.index.values, df_sub, lw=2)
            
            #save leak ubstracted data 
            if self.save_leaksub == True:
                #add voltage traces 
                df_sub_with_voltage = pd.concat([df_sub, df.iloc[:,N:]], axis=1)
                df_sub_with_voltage.to_csv(self.save_path + r"normalized_data/leaksub/%s_leaksub.csv" % self.filenames[i] )
            
            #tail peaks for activation curve 
            tail_peaks = [] 
            #tail mins for normalizing deactivation 
            tail_mins = [] 
            
            #take only test pulse and deactivation
            if 'FA' in self.protocol_name:
                ca = 6*self.khz #capacitance, activation 
                cd = 18*self.khz #capacticance, deactivation 
            else:
                ca = 14*self.khz 
                cd = 12*self.khz
            
            #separate activation and tail currents 
            act_list = [] 
            de_list = [] 
            for j in range(N):
                dfa = df_sub.iloc[test_protocol[0]+ca:test_protocol[1][j], j]
                dfd = df_sub.iloc[test_protocol[1][j]+cd:test_protocol[2][j], j]
                
                #zero the time index 
                dfa.index -= dfa.index[0]
                dfd.index -= dfd.index[0]                
                
                # ax.plot(dfa)
                # ax.plot(dfd)
                # plt.show()
                # exit()

                #append peak and minimum tail currents to respecitve lists
                #use last/first 2ms 
                tail_peaks.append(dfd.rolling(2*self.khz).mean().max())
                tail_mins.append(dfd.rolling(2*self.khz).mean().min())
                
                act_list.append(dfa)
                de_list.append(dfd) 
                
            # plt.show()
            # exit()            
            
            #maximal tail current 
            m = max(tail_peaks) 
            #normalize tail mins 
            tail_mins = [x/m for x in tail_mins]
            # set the minimum tail_min to 0.02, then divide by max tail peak 
            # tail_mins = [(x-min(tail_mins))/(m-min(tail_mins)) for x in tail_mins]
            # delta value to add to all tail_mins so that minimum tail_min is 0.02 
            dmin = pmin - min(tail_mins)
            tail_mins = [x+dmin for x in tail_mins] 
            # ax.plot(tail_mins, marker='o')
            # ax.show()
                
            #normalize tail peaks 
            ac = [x/m for x in tail_peaks] 
            
            #plot activation curve 
            f_ax3b.plot(test_protocol[-1][:N], ac, marker='o', ls='none')
            
            #fit boltzmann 
            try:
                p_boltz, e_boltz = fit_boltz(test_protocol[-1][:N], ac)               
                #plot fitted boltzmann 
                vrange = range(-20, -165, -5)
                fitted_boltz = [boltz(v, p_boltz[0], p_boltz[1], p_boltz[2]) for v in vrange]
                f_ax3b.plot(vrange, fitted_boltz, 
                        ls='--', lw=2, 
                        label="s=%d, \nVh=%d" % (p_boltz[1], p_boltz[2]))        
            except:
                print("Failed to fit Boltzmann")
            
            #normalize activation and deactivation traces 
            if len(act_list) != len(de_list):
                print("Number act and de traces don't match.")
                exit()    
            
            #if normalization fails, remove guilty traces and reduce value of N 
            to_delete = [] 
            
            r_act = 2
            for j in range(len(act_list)):
                try:
                    act_list[j] = normalize(act_list[j], ac[j], 
                                            pmin, self.khz, r_act=r_act)                
                    #for deactivation, we need to use the minimum leak-suubtracted tail current (divided by peak tail current)
                    de_list[j] = normalize(de_list[j], ac[j], 
                                        tail_mins[j], self.khz, method='de')
                except:                    
                    to_delete.append(j - len(to_delete))
                    continue 
            
            #delete traces where normalization failed 
            if len(to_delete) > 0:
                for j in to_delete:
                    N -= 1
                    del act_list[j]
                    del de_list[j] 
                    del test_protocol[-1][j]
                    del fit_params[j]
                                    
            #concatenate normalized activation and deactivation traces 
            df_act = pd.concat(act_list, axis=1)
            df_de = pd.concat(de_list, axis=1)
                        
            #name dataframe columns with test voltages 
            df_act.columns = test_protocol[-1][:N]
            df_de.columns = test_protocol[-1][:N]
            
            #plot normalized Po 
            # df_act.index *= 1/self.khz 
            # df_de.index *= 1/self.khz 
            f_ax1.plot(df_act)
            f_ax2.plot(df_de)
                
            #axes labels, legends 
            f_ax1.set_title(self.filenames[i])    
            f_ax1.legend(df_act.columns, loc='upper right', 
                        ncol=4, edgecolor='k',
                        labelspacing=0.4, handletextpad=0.6, columnspacing=1.5,
                        fontsize=10, framealpha=0.5)
            f_ax2.legend(df_de.columns, loc='upper right',
                        ncol=4, edgecolor='k',
                        labelspacing=0.4, handletextpad=0.6, columnspacing=1.5,
                        fontsize=10, framealpha=0.5)
            f_ax1.set_xlabel("Time (ms)")
            f_ax1.set_ylabel(r"Activation $P_{open}$")
            f_ax2.set_ylabel(r"Tail $P_{open}$")
            f_ax2.set_xlabel("Time (ms)")
            
            f_ax3a.set_xlabel("Time (ms)")
            f_ax3a.set_ylabel("Leak- \nsubtracted pA")
            f_ax3a.legend(df_act.columns, loc='lower right',
                        ncol=4, edgecolor='k',
                        labelspacing=0.4, handletextpad=0.6, columnspacing=1.5,
                        fontsize=10, framealpha=0.5)
            f_ax3b.set_xlabel("Voltage (mV)")
            f_ax3b.set_ylabel(r"$I/I_{max}$")
            f_ax3b.legend(loc='upper right', 
                        edgecolor='k', labelspacing=0.4, 
                        handletextpad=0.6, columnspacing=1.5,
                        fontsize=10, framealpha=0.5)    
                            
            f_ax4b.set_ylabel(r"mV")
            f_ax4c.set_ylabel(r"pS")
            f_ax4c_x2.set_ylabel(r"$r^2$")
            
            f_ax4a.set_ylabel(r"Fitted Leak pA")
            f_ax4a.set_xlabel(r"Voltage (mV)")
            
            f_ax4b.set_xlabel(r"Sweep # / Test Pulse (mV)")
            f_ax4c.set_xlabel(r"Sweep # / Test Pulse (mV)")

            #add test pulse level to xticklabel of f_ax4b 
            new_labels = ["%d\n%d" % (i, test_protocol[-1][i-1]) for i in range(1, N+1)]
            f_ax4b.set_xticks(range(1, N+1))
            f_ax4b.set_xticklabels(new_labels)
            f_ax4c.set_xticks(range(1, N+1))
            f_ax4c.set_xticklabels(new_labels)
            
            #set ylim to 0.5 - 1 
            f_ax4c_x2.set_ylim(bottom=0.6, top=1.05)
            #average the fit params 
            fit_params = np.mean(fit_params, axis=0)        
            #combine legends in 4th row into one legend 
            h1, l1 = f_ax4c.get_legend_handles_labels()
            h2, l2 = f_ax4c_x2.get_legend_handles_labels()
            #make legends
            # hb, lb = f_ax4b.get_legend_handles_labels()
            f_ax4b.legend(loc='lower right', 
                        edgecolor='k', labelspacing=0.2,
                        handletextpad=0.1, fontsize=10, framealpha=0.5) 
            f_ax4c.legend(h1+h2, l1+l2, 
                        loc='lower right', 
                        edgecolor='k', labelspacing=0.2,
                        handletextpad=0.1, fontsize=10, framealpha=0.5)                        
                 
            f_ax5a.set_ylabel("Raw, pA")
            f_ax5b.set_ylabel("mV")
            f_ax5a.set_xlabel("Time (ms)")
            f_ax5b.set_xlabel("Time (ms)")
            # f_ax5a.legend(df_act.columns, loc='lower right',
                        # ncol=4, edgecolor='k',
                        # labelspacing=0.4, handletextpad=0.6, columnspacing=1.5,
                        # fontsize=10, framealpha=0.5)
            
            # plt.tight_layout()
            
            if self.all_output: 
                f_fnl.savefig(self.save_path + r"process_output/%s.png" % self.filenames[i])
            
                if self.filenames[i] in self.dates_to_save:
                    df_act.to_csv(self.save_path + r"normalized_data/%s_act_act.csv" % self.filenames[i])
                    df_de.to_csv(self.save_path + r"normalized_data/%s_act_de.csv" % self.filenames[i])
        
            print(self.filenames[i])
            plt.show()
            plt.close(f_fnl)
            
            if self.filenames[i] == '20917008':
                exit()
            
    def de(self):
        
        #number of samples before the ramp 
        #   first step is 500 samples
        t0_plus_delay = 410       
                
        #assuming no delay, go through self.steps and find start and end of initial leak ramp 
        t_ramp = []
        t_i = (t0_plus_delay - 250)*self.khz
        for i in range(len(self.steps)):
            t, v, x = self.steps[i] 
                
            if x == 'r':
                t_ramp = [t_i, t_i + (2*t), t_i + t]
                
                #prepulse is two self.steps after the initial ramp        
                #hyperpolarizing prepulse 
                t_pre = self.steps[i+3] 
                #depolarizing test pulse 
                t_test = [t_pre, self.steps[i+4]] 
                #post pulse 
                t_test.append(self.steps[i+5])
                #start time of prepulse  
                t0 = t_i + t + self.steps[i+1][0] + self.steps[i+2][0]
                t_test.append(t0) 
                #end of prepulse / start of tail 
                t1 = t0 + int(t_pre[0])
                t_test.append(t1)
                #end time of tail 
                t2 = t1 + int(t_test[1][0]) 
                t_test.append(t2)       
                break 
            else:
                try:
                    t_i += t 
                except:
                    print(t)
                    
                continue 

        # print(t_ramp)
        # exit()

        #unpack test pulse 
        t, v0, x = t_test[1] 
        x = x.split(", ")
        dv = int(x[-1])
        
        pmin = 0.02
        # print(len(self.data_files))
        
        for i in range(len(self.data_files)):
            print("Processing... %s" % self.filenames[i])
            
            df = self.data_files[i]    
            N = int(df.shape[1]/2)        
            
            #set default color cycle 
            rcParams['axes.prop_cycle'] = plt.cycler(plt.cycler('color', cmap(np.linspace(0, 1, N+1))))
            
            #check raw data 
            # plt.plot(df.iloc[:,N:])
            # plt.show()
            # plt.close()
            # exit()
            
            #test volts from dv and initial v0 
            test_volts = [v0 + (j*dv) for j in range(N)] 
            
            #final figure for output 
            """
            Rows 
                1. normalized data 
                2. voltage protocol 
                3. Leak subtracted data,  
                4. each of N leak current and fitted ohmic leak equation 
                5a. Raw data, 5b. Raw voltage protocol 
            """
            #create figure 
            f_fnl = plt.figure(constrained_layout=True, figsize=(10, 12))
            #create gridspec
            gs = f_fnl.add_gridspec(5, 6)
            M = 3 
                
            f_ax1 = f_fnl.add_subplot(gs[0, :])
            f_ax2 = f_fnl.add_subplot(gs[1, :])
            f_ax3a = f_fnl.add_subplot(gs[2, :-2])
            f_ax3b = f_fnl.add_subplot(gs[2, -2:])
            f_ax4a = f_fnl.add_subplot(gs[3, :M])
            f_ax4b = f_fnl.add_subplot(gs[3, M:])
            f_ax5a = f_fnl.add_subplot(gs[-1, :M])
            f_ax5b = f_fnl.add_subplot(gs[-1, M:])    
            
            #time of initial ramp 
            t1, t2, thalf = t_ramp         
            t1 -= 1
            t2 -= 1
            thalf -= t1 + 1 
            
            # # f, ax = plt.subplots()
            # print(t_ramp) 
            # f, ax = plt.subplots(2, 1)
            # #check ramp 
            # ax[0].plot(df.index[t1:t2], df.iloc[t1:t2,:N])
            # ax[1].plot(df.index[t1:t2], df.iloc[t1:t2,N:])
            # plt.show()
            # exit()
            
            #plot raw data nad protocol 
            f_ax5a.plot(df.index.values, df.iloc[:,:N],
                    lw=2)
            f_ax5b.plot(df.index.values, df.iloc[:,N:],
                    lw=2)
            
            #leak subtracted traces 
            subtracted = [0]*N 
            
            #get initial ramp current and voltages 
            fit_params = [0]*N 
            for j in range(N): 
                clr = cmap(j/N)
                
                current = df.iloc[t1:t2,j].values        
                volts = df.iloc[t1:t2,N+j].values

                #average current in both arms
                i_mu = (current[:thalf] + np.flip(current[thalf:]))/2
                v_mu = (volts[:thalf] + np.flip(volts[thalf:]))/2
                        
                #fit averaged ramp 
                p, e = fit_leak(v_mu, i_mu)
                
                #plot average ramp current 
                f_ax4a.plot(v_mu, i_mu, c=clr, alpha=0.8)
                #compute ohmic leak 
                iL = [ohmic_leak(v, p[0], p[1]) for v in v_mu]
                #plot ohmic leak (fitted) with a black border 
                f_ax4a.plot(v_mu, iL, c=clr, lw=2,
                    path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()]
                    )
                
                #get r2 value for fit 
                r2_j = get_r2(v_mu, i_mu, p)
                
                #conductance, Erev, r2 
                fit_params[j] = [p[0], -p[1]/p[0], r2_j]
                        
                #subtract leak from jth trace 
                i_leak = [ohmic_leak(v, p[0], p[1]) for v in df.iloc[:,N+j]]
                sub = df.iloc[:,j] - np.array(i_leak)
           
                subtracted[j] = sub
                
            #plot fitted leak parameters 
            fit_params_labels = [r"$\gamma$", r"$E_{rev}$", r"$r^2$"]
            
            #use twinx for r2 
            f_ax4b_x2 = f_ax4b.twinx()
            
            for k in range(len(fit_params[0])):        
                clr = cmap(k/len(fit_params[0]))
                
                if k == 2:
                    f_ax4b_x2.plot(range(1, N+1), 
                            [x[k] for x in fit_params],
                            marker='x', markersize=6, c='k',
                            ls='none', markeredgewidth=1,
                            label = fit_params_labels[k]
                            )
                else:
                    f_ax4b.plot(range(1,N+1), 
                            [x[k] for x in fit_params],
                            marker='o', markersize=4, c=clr,
                            ls='none', markeredgewidth=1, 
                            label = fit_params_labels[k]
                            )
                            
            #concatenate subtracted traces into dataframe
            df_sub = pd.concat(subtracted, axis=1).reset_index(drop=True)
            #scale index values to self.khz 
            df_sub.index *= 1/self.khz 
            #plot leak-subtracted traces 
            f_ax3a.plot(df_sub.index.values, df_sub, lw=2)

            #save leak ubstracted data 
            if self.save_leaksub == True:
                #add voltage traces 
                df_sub_with_voltage = pd.concat([df_sub, df.iloc[:,N:]], axis=1)
                df_sub_with_voltage.to_csv(self.save_path + r"normalized_data/leaksub/%s_leaksub.csv" % self.filenames[i] )
            
            #start of prepulse, end of prepulse, end of tail 
            t0, t1, t2 = [t-1 for t in t_test[-3:]]
            #end of post pulse 
            t3 = t2 + int(t_test[2][0]) + 1 
            
            
            act_list, de_list, tail_mins, tail_peaks, tail_post_peaks = sep_de(df_sub,
                                                                        [t0, t1, t2, t3],
                                                                        [6, 10, 15],
                                                                        self.khz,
                                                                        show=False 
                                                                        )
            #get voltage of post pulse 
            v_post = t_test[2][1]
            #index of post pulse voltage in test pulses 
            idx_v = test_volts.index(int(v_post))
            #normalize tail_post_peaks 
            tail_post_peaks = [x/tail_peaks[idx_v] for x in tail_post_peaks] 
                        
            #set minimal pmin to 0.02
            # plt.plot(test_volts, tail_mins, marker='o')
            # plt.show()
            # exit()
            
            #plot activation curve 
            f_ax3b.plot(test_volts, tail_mins, 
                        c='r',
                        marker='o', label="Tail")
            f_ax3b.plot(test_volts, tail_post_peaks, 
                        c='b', 
                        marker='o', label="Post")
            
            #set ylim of P_min plot 
            largest_Pmin = int(10*max([max(tail_mins), max(tail_post_peaks)]))/10 
            if largest_Pmin > 0:
                f_ax3b.set_ylim(bottom=-0.05, top=largest_Pmin+0.05)
            else:
                f_ax3b.set_ylim(bottom=-0.05, top=0.1)
            
            #normalize activation and deactivation traces 
            if len(act_list) != len(de_list):
                print("Number act and de traces don't match.")
                exit()   
            if self.filenames[i] == '20917009':
                de_list[4] = de_list[4].iloc[14*self.khz:]
                
            for j in range(len(act_list)):
                act_list[j] = normalize(act_list[j], 1, pmin, self.khz)
                
                #for deactivation, we need to use the minimum leak-suubtracted tail current (divided by peak tail current)
                de_list[j] = normalize(de_list[j], 1, tail_mins[j], self.khz, method='de')
                # de_list[j] = normalize(de_list[j], 1, tail_post_peaks[j])
                                  
            #concatenate normalized activation and deactivation traces 
            df_act = pd.concat(act_list, axis=1)
            df_de = pd.concat(de_list, axis=1)
            #name dataframe columns with test voltages 
            df_act.columns = test_volts
            df_de.columns = test_volts
            
            #plot normalized Po 
            f_ax1.plot(df_act, lw=2)
            f_ax2.plot(df_de, lw=2)
                
            #legend columns 
            if N <= 4:
                ncol = 2
            elif 4 < N <= 6:
                ncol = 3 
            else:
                ncol = 4
                
            #axes labels, legends 
            f_ax1.set_title(self.filenames[i])    
            f_ax1.legend(df_act.columns, loc='lower right', 
                        ncol=ncol, edgecolor='k',
                        labelspacing=0.4, handletextpad=0.6, columnspacing=1.5,
                        fontsize=10, framealpha=0.5)
            # f_ax2.legend(df_de.columns, loc='upper right',
                        # ncol=4, edgecolor='k',
                        # labelspacing=0.4, handletextpad=0.6, columnspacing=1.5,
                        # fontsize=10, framealpha=0.5)
            f_ax1.set_xlabel("Time (ms)")
            f_ax1.set_ylabel(r"Activation $P_{open}$")
            f_ax2.set_ylabel(r"Tail $P_{open}$")
            f_ax2.set_xlabel("Time (ms)")
            
            f_ax3a.set_xlabel("Time (ms)")
            f_ax3a.set_ylabel("Leak- \nsubtracted pA")
            # f_ax3a.legend(df_act.columns, loc='lower right',
                        # ncol=ncol, edgecolor='k',
                        # labelspacing=0.4, handletextpad=0.6, columnspacing=1.5,
                        # fontsize=10, framealpha=0.5)
            f_ax3b.set_xlabel("Voltage (mV)")
            f_ax3b.set_ylabel(r"$P_{min}$") 
            f_ax3b.legend(loc='best',
                        ncol=2, edgecolor='k',
                        labelspacing=0.4, handletextpad=0.6, columnspacing=1.5,
                        fontsize=10, framealpha=0.1)            
            
            f_ax4a.set_ylabel(r"Fitted Leak pA")
            f_ax4a.set_xlabel(r"Voltage (mV)")
            f_ax4b_x2.set_ylabel(r"$r^2$")
            f_ax4b.set_ylabel(r"pS or mV")
            f_ax4b.set_xlabel(r"Sweep # / Test Pulse (mV)")

            #add test pulse level to xticklabel of f_ax4b 
            new_labels = ["%d\n%d" % (i, test_volts[i-1]) for i in range(1, N+1)]
            f_ax4b.set_xticks(range(1, N+1))
            f_ax4b.set_xticklabels(new_labels)
            
            #set ylim to 0.5 - 1 
            f_ax4b_x2.set_ylim(bottom=0.6, top=1.05)
            #average the fit params 
            fit_params = np.mean(fit_params, axis=0)        
            #combine legends in 4th row into one legend 
            h1, l1 = f_ax4b.get_legend_handles_labels()
            h2, l2 = f_ax4b_x2.get_legend_handles_labels()
            f_ax4b.legend(h1+h2, l1+l2, 
                        loc='center right', ncol=3,
                        edgecolor='k', labelspacing=0.2,
                        handletextpad=0.1, columnspacing=0.5,
                        fontsize=10, framealpha=0.5)        
                
            f_ax5a.set_ylabel("Raw, pA")
            f_ax5b.set_ylabel("mV")
            f_ax5a.set_xlabel("Time (ms)")
            f_ax5b.set_xlabel("Time (ms)")
            # f_ax5a.legend(df_act.columns, loc='lower right',
                        # ncol=4, edgecolor='k',
                        # labelspacing=0.4, handletextpad=0.6, columnspacing=1.5,
                        # fontsize=10, framealpha=0.5)
            
            # plt.tight_layout()
            f_fnl.savefig(self.save_path + r"process_output/%s.png" % self.filenames[i])
            
            plt.show()
            plt.close(f_fnl)
            
            if self.filenames[i] in self.dates_to_save:
                df_act.to_csv(self.save_path + r"normalized_data/%s_de_act.csv" % self.filenames[i])
                df_de.to_csv(self.save_path + r"normalized_data/%s_de_de.csv" % self.filenames[i])
            
            # exit()        
        
    def ramp(self):        
        #assuming no delay, go through self.steps and find start and end of initial leak ramp 
        t_ramp = []

        t_i = (2111-500)*self.khz
        # if "WT" in self.protocol_name:
        #     t_i = (1600-500)*self.khz 

        for i in range(len(self.steps)):
            t, v, x = self.steps[i] 
                
            if x == 'r':
                #append start, end, and half-time of initial ramp 
                t_ramp = [t_i, t_i + t, t_i + (2*t)]  
                
                #test ramp starts one step after end of initial leak ramp
                test_start = t_ramp[-1] + self.steps[i+2][0]
                
                # print(self.steps[i:])
                #unpack test step 
                test_duration, v, x = self.steps[i+3]
                # print(self.steps[i:])
                
                #split step type 
                s = x.split(", ")
                #delta duration 
                delta_test_duration = int(s[1])
                
                test_ramp = [test_start, int(test_duration), delta_test_duration]
                # print(test_ramp)
                
                break 
                
            else:
                try:
                    t_i += int(t) 
                except:
                    print(t)                    
                continue
                
        for i in range(len(self.data_files)):
            
            if self.filenames[i] == '20903005':
                continue 
            
            df = self.data_files[i]    
            N = int(df.shape[1]/2)  
            
            #see waht data looks like 
            # plt.figure()
            # plt.plot(df.index.values, df.iloc[:,N:])
            # plt.show()
            # exit()
            
            #set default color cycle 
            rcParams['axes.prop_cycle'] = plt.cycler(plt.cycler('color', cmap(np.linspace(0, 1, N+1))))
            
            #final figure for output 
            """
            Rows 
                1. normalized data 
                2. voltage protocol 
                3a. flipped normalized ramps,   3b. hysteresis 
                4. Leak subtracted data,  
                5. each of N leak current and fitted ohmic leak equation 
                6a. Raw data, 6b. Raw voltage protocol 
            """
            f_fnl = plt.figure(figsize=(10, 12))

            gs = f_fnl.add_gridspec(6, 6)
            #create axes 

            f_ax1 = f_fnl.add_subplot(gs[0, :])
            f_ax2a = f_fnl.add_subplot(gs[1, :-3])
            f_ax2b = f_fnl.add_subplot(gs[1, -3:])
            f_ax3a = f_fnl.add_subplot(gs[2, :3])
            f_ax3b = f_fnl.add_subplot(gs[2, 3:])
            f_ax4 = f_fnl.add_subplot(gs[3, :])
            f_ax5a = f_fnl.add_subplot(gs[4, :3])
            f_ax5b = f_fnl.add_subplot(gs[4, 3:])
            f_ax6a = f_fnl.add_subplot(gs[5, :3])
            f_ax6b = f_fnl.add_subplot(gs[5, 3:])              
            
            #plot raw data and voltage protocol 
            f_ax6a.plot(df.index.values, df.iloc[:,:N])
            f_ax6b.plot(df.index.values, df.iloc[:,N:], lw=3)
            
            #subtract 1 from each of t_ramp times to account for 0-indexing 
            t1, t2, t3 = [t-1 for t in t_ramp]
            
            subtracted = [] 
            test_ramps = [0]*N
            fit_params = [0]*N 
            save_leaksub = [] 
            for j in range(N):
                clr = cmap(j/N)
                #get start time, end time, and midpoint time of jth test ramp 
                test_ramps[j] = [test_ramp[0], 
                                test_ramp[0] + 2*(test_ramp[1] + j*test_ramp[2])]
                test_ramps[j].append(int(0.5*(test_ramps[j][1]-test_ramps[j][0])))
                
                # f, ax = plt.subplots()
                # ax.plot(df.iloc[test_ramps[j][0]-1:test_ramps[j][1]-1, j])
                # plt.show()
                # plt.close()
                # exit()
                
                ### LEAK SUBTRACTION ###
                #get both arms of the initial leak ramp 
                # arm1 = df.iloc[t1:t2, [N+j, j]]
                # arm2 = df.iloc[t2:t3, [N+j, j]]
                
                # f, ax = plt.subplots()
                # ax.plot(arm1.iloc[:,0], arm1.iloc[:,1])
                # ax.plot(arm2.iloc[:,0], arm2.iloc[:,1])
                # #same as above, but mV x time, not mV x pA  
                # ax.plot(df.index.values[t1:t3], df.iloc[t1:t3, N+j])
                # plt.show()
                # plt.close()
                # exit()
                
                #average current in both arms
                current = df.iloc[t1:t3,j].values 
                volts = df.iloc[t1:t3,N+j].values 
                i_mu = (current[:t2-t1] + np.flip(current[t2-t1:]))/2
                v_mu = (volts[:t2-t1] + np.flip(volts[t2-t1:]))/2
                
                #fit averaged ramp 
                p_avg = fit_leak(v_mu, i_mu)[0] 
                #get r2 value for fit 
                r2_j = get_r2(v_mu, i_mu, p_avg)
                #conductance, Erev, r2 
                fit_params[j] = [p_avg[0], -p_avg[1]/p_avg[0], r2_j]
                                    
                ### PLOT LEAK SUBTRACTION ###               
                #compute leak current 
                fitted_leak_ramp = [ohmic_leak(v, p_avg[0], p_avg[1]) for v in v_mu] 
                
                #plot linear leak in 5th row 
                #plot fitted linear leak 
                f_ax5a.plot(v_mu, fitted_leak_ramp, 
                            lw=2, c='k', 
                            label=None
                            )
                #plot currents of both arms of initial ramp 
                f_ax5a.plot(v_mu, i_mu, lw=2, c=clr, alpha=0.8)
                
                ### APPLY LEAK SUBTRAACTION ###
                #compute leak current for entire sweep 
                i_leak = [ohmic_leak(v, p_avg[0], p_avg[1]) for v in df.iloc[:,N+j]]
                
                #subtract leak current 
                df_subtracted = df.iloc[:,j] - i_leak 
                #append leak-subtracted test ramp/pulse                 
                subtracted.append(df_subtracted.iloc[test_ramps[j][0]-1:test_ramps[j][1]-1])                                  

                # f, ax = plt.subplots()
                # ax.plot(df_subtracted.iloc[test_ramps[j][0]-1:test_ramps[j][1]-1])
                                
                ### PLOT LEAK-SUBTRACTED CURRENT ###  
                f_ax4.plot(df.index.values, df_subtracted, c=cmap(j/N))
                
                if self.save_leaksub == True:
                    save_leaksub.append(df_subtracted) 
                
            #plot fitted leak parameters 
            fit_params_labels = [r"$\gamma$", r"$E_{rev}$", r"$r^2$"]
            
            #use twinx for r2 
            f_ax5b_x2 = f_ax5b.twinx()
            
            for k in range(len(fit_params[0])):        
                clr = cmap(k/len(fit_params[0]))
                
                if k == 2:
                    f_ax5b_x2.plot(range(1, N+1), 
                            [x[k] for x in fit_params],
                            marker='x', markersize=6, c='k',
                            ls='none', markeredgewidth=1,
                            label = fit_params_labels[k]
                            )
                else:
                    f_ax5b.plot(range(1,N+1), 
                            [x[k] for x in fit_params],
                            marker='o', markersize=4, c=clr,
                            ls='none', markeredgewidth=1, 
                            label = fit_params_labels[k]
                            )
            
            #concatenate leak-subtracted sweeps into single dataframe 
            df_sub = pd.concat(subtracted, axis=1) 
            # print(df_sub)
            #zero time axis 
            df_sub = df_sub.apply(lambda x: pd.Series(x.dropna().values).fillna(''))
            # f, ax = plt.subplots()
            # ax.plot(df_sub.index.values, df_sub)
            # plt.show()            
            # exit()

            #save leak ubstracted data 
            if self.save_leaksub == True:
                #add voltage traces 
                save_leaksub.append(df.iloc[:,N:])
                df_sub_with_voltage = pd.concat(save_leaksub, axis=1)
                df_sub_with_voltage.to_csv(self.save_path + r"normalized_data/leaksub/%s_leaksub.csv" % self.filenames[i] )
            
            #normalize peak current amplitudes
            i3 = df_sub.rolling(3, axis=0).mean() 
            peaks = i3.min(axis=0)
  
            #normalize peaks 
            peaks_norm = [x/min(peaks) for x in peaks]
        
            #normalize to maximal amplitude
            pmin = 0.02
            for j in range(N):
                i5 = df_sub.iloc[:,j].dropna().rolling(5).mean().iloc[4:]

                #baseline 
                i0 = i5.iat[0]
                #peak current 
                imax = i5.min() 
                #normalized peak 
                finf = peaks_norm[j] 

                #scale 
                X = ((pmin/finf)*imax - i0)/((pmin/finf)-1)
                df_sub.iloc[:,j] = finf*(df_sub.iloc[:,j] - X)/(imax - X)

                #average every 3 values in ith trace 
                i5 = df_sub.iloc[:,j].dropna().rolling(5).mean().iloc[4:]    
                #baseline 
                i0 = i5.iat[0]
                #peak current 
                imax = i5.max() 
                
                h = 0
                while abs(i0 - pmin) > 0.001:
                    #scale 
                    X = ((pmin/finf)*imax - i0)/((pmin/finf)-1)
                    df_sub.iloc[:,j] = finf*(df_sub.iloc[:,j] - X)/(imax - X)
                    
                    #average every 3 values in ith trace 
                    i5 = df_sub.iloc[:,j].dropna().rolling(5).mean().iloc[4:]
                    #baseline
                    i0 = i5.min()
                    #peak current 
                    imax = i5.max()
                    
                    h += 1
                    if h % 100 == 0:
                        print(h, i0, imax, X, abs(i0 - pmin))
            
            #compute hysteresis for leak-subtracted, normalized data 
            hysteresis = [] 
            for j in range(N):
                clr = cmap(j/N)
                
                #dropna 
                y = df_sub.iloc[:,j].dropna().values 
                #halftime of ramp 
                thalf = test_ramps[j][-1] 
                #half-duration of ramp 
                dt = np.array(range(thalf))
                
                #compute hysteresis 
                # print(len(y), len(dt))
                a1, a2 = get_hysteresis(y, thalf, dt)
                
                #flipped time, for plotting depolarizing ramp 
                dt2 = np.flip(dt) 
                
                #plot flipped ramps and hysteresis in 3rd row of final figure 
                f_ax3a.plot(dt, y[:thalf], c=clr)
                f_ax3a.plot(dt2, y[thalf:], c=clr, alpha=0.3)
                #hysteresis 
                f_ax3b.plot(thalf/1000, a2-a1, 
                    marker='o', markersize=8, ls='none', 
                    c=clr, label=str(int(thalf/1000)) + "s") 
                    
                hysteresis.append(a2-a1) 
            
            #plot normalized hysteresis in final figure 
            ramp_durations = [t[-1]/1000 for t in test_ramps]
            f_ax3b.plot(ramp_durations, hysteresis, 
                        lw=2, ls='--', alpha=0.5, 
                        label=None) 
            #add legend to bottom right of 3rd row of final figure 
            f_ax3b.legend(loc='lower right', fontsize=10, framealpha=0.5)
            #set yscale to log 
            # f_ax3b.set_yscale('log')
            f_ax3b.set_xticks(ramp_durations, minor=False)
            
            #plot full normalized, leak-subtracted df sub with protocol in final figure
            times = df.index.values 
            truncated_voltage = [] 
            for j in range(N):
                clr = cmap(j/N)
                t1, t2 = test_ramps[j][:2] 
                t1 -= 1 
                t2 -= 1 
                
                #time x po 
                f_ax1.plot(df_sub.index.values, df_sub.iloc[:,j], 
                            lw=2, c=clr,
                            label=str(int(test_ramps[j][-1]/1000)) + "s")
                
                #time x volts 
                volts = df.iloc[t1:t2,N+j]
                f_ax2b.plot(times[t1:t2] - t1, 
                            volts,
                            lw=3, c=clr,
                            label=str(int(test_ramps[j][-1]/1000)) + "s")
                #volts x po 
                thalf = int((t2-t1)/2)  #half duration of ramp 
                f_ax2a.plot(volts[:thalf], 
                            df_sub.iloc[:thalf,j], 
                            lw=2, c=clr,
                            label=None)
                f_ax2a.plot(volts[thalf:], 
                            df_sub.iloc[thalf:2*thalf,j], 
                            alpha=0.5,
                            lw=2, c=clr,
                            label=str(int(test_ramps[j][-1]/1000)) + "s")
                
                #save isolated voltage command for output csv 
                if self.filenames[i] in self.dates_to_save:
                    truncated_voltage.append(volts) 
                
            f_ax1.legend(loc='upper right', fontsize=10)
            f_ax2b.legend(loc='lower right', fontsize=10)
            
            #axes labels and titles 
            f_ax1.set_ylabel(r"$P_{open}$")
            f_ax1.set_xlabel("Time (ms")
            f_ax1.set_title(self.filenames[i])

            f_ax2b.set_ylabel("mV")
            f_ax2b.set_xlabel("Time (ms)")
            f_ax2a.set_ylabel(r"$P_{open}$")
            f_ax2a.set_xlabel("mV")

            f_ax3a.set_ylabel(r"$P_{open}$")
            f_ax3a.set_xlabel("Time (ms)")

            f_ax3b.set_ylabel(r"$H(t)$")
            f_ax3b.set_xlabel(r"Ramp duration (s)")

            f_ax4.set_ylabel("Leak-\n subtracted pA")
            f_ax4.set_xlabel("Time (ms)")

            f_ax5a.set_ylabel("Fitted Leak")                
            f_ax5a.set_xlabel("Voltage (mV)") 
            f_ax5b_x2.set_ylabel(r"$r^2$")
            f_ax5b.set_ylabel(r"pS or mV")
            f_ax5b.set_xlabel(r"Sweep #")
            f_ax5b.set_xticks(range(1,N+1))
            
            fit_params = np.mean(fit_params, axis=0)        
            #combine legends in 4th row into one legend 
            h1, l1 = f_ax5b.get_legend_handles_labels()
            h2, l2 = f_ax5b_x2.get_legend_handles_labels()
            f_ax5b.legend(h1+h2, l1+l2, 
                        loc='center right', ncol=3,
                        edgecolor='k', labelspacing=0.2,
                        handletextpad=0.1, columnspacing=0.5,
                        fontsize=10, framealpha=0.5)  
            
            f_ax6a.set_xlabel("Time (ms)")
            f_ax6a.set_ylabel("Raw, pA")

            f_ax6b.set_xlabel("Time (ms)")
            f_ax6b.set_ylabel("Raw, mV")
            
            f_fnl.tight_layout() 
            
            #save final figure
            print(self.filenames[i])
            plt.savefig(self.save_path + r"process_output/%s.png" % self.filenames[i])
            
            #output csv of normalized data 
            if self.filenames[i] in self.dates_to_save:
                ramp_durations = [t[-1] for t in test_ramps] 
                df_sub.columns = ramp_durations

                truncated_voltage = pd.concat(truncated_voltage, axis=1).apply(lambda x: pd.Series(x.dropna().values).fillna(''))
                
                df_sub = pd.concat([df_sub, truncated_voltage], axis=1)
                df_sub.to_csv(self.save_path + r"normalized_data/%s_ramp_norm.csv" % self.filenames[i])
                exit()
            
            plt.show()
            # plt.close()
                
    def go(self, to_skip={}, all_output=True, save_leaksub=False):
        self.all_output = all_output 
        self.save_leaksub = save_leaksub 
        
        if len(to_skip.keys()) > 0:
            for i in range(len(self.data_files)):
                df = self.data_files[i]
                N = int(df.shape[1]/2)
                
                print(df.columns)
                print(df.shape)
                
                if self.filenames[i] in to_skip.keys():
                    print("Dropping traces ", to_skip[self.filenames[i]], " from ", self.filenames[i])
                    
                    skipping = [] 
                    for k in to_skip[self.filenames[i]]:
                        skipping.append(df.columns[k])
                        skipping.append(df.columns[k+N])
                
                    df.drop(skipping, axis=1, inplace=True)
                    df.columns = list(range(1, df.shape[1]+1))
                    self.data_files[i] = df
                
        if 'act_stag_R' in self.protocol_name:
            print("Processing with `act_stag_R`...")
            self.act()
        elif 'de_R' in self.protocol_name:
            print("Processing with `de_R`...")
            self.de()
        elif 'envelope' in self.protocol_name:
            print("Processing with `envelope`...")
            self.env() 
        elif "ramp_dt" in self.protocol_name: 
            print("Processing with `ramp_dt`...")
            self.ramp()
        
        else:
            raise Exception("No function to process protocol: %s" % self.protocol_name)


def oct27(dates=["20917"]):
    p1 = process(dates, "WT_act_stag_R", dates_to_save=['20917000', '20917008', '20917009'])
    p2 = process(dates, "WT_ramp_dt", dates_to_save=['20917000', '20917008', '20917009'])
    p3 = process(dates, "WT_de_R", dates_to_save=['20917000', '20917008', '20917009'])
    
    # p1.go(all_output=False, save_leaksub=True)
    p2.go(save_leaksub=True)
    # p3.go(save_leaksub=True)
    
def oct28(dates=['20918', '20o01', '20o08', '20o16', '20o22']):
    protocols = ["FA_ramp_dt", "FA_de_envelope_v2", "FA_act_stag_R", "FA_de_R", "WT_act_stag_R"]
    
    for pro in protocols:
        p = process(dates, pro) 
        p.go(all_output=True)
       
def oct31a(dates=["20917"]):
    # p1 = process(dates, "WT_de_envelope_A", dates_to_save=["20917002", "20917007", "20917010"])
    p2 = process(dates, "WT_de_envelope_B", dates_to_save=["20917002", "20917007", "20917010"])
    
    # p1.go(save_leaksub=False, all_output=False) 
    p2.go(save_leaksub=True, all_output=True) 
        
# oct27()
# oct28(dates) 
# oct31a()

def oct31b(dates=["20917"]):
    p = process(dates, "WT_ramp_de", dates_to_save=["20917009"])
    p.go(save_leaksub=False, all_output=False)
    
# oct31b() 

def nov17(dates=["20910"]):
    p = process(dates, "FA_act_stag_R", dates_to_save=["20910002", "20o08004", "20918006"])
    p.go(to_skip={"20910002" : [0]}, save_leaksub=True, all_output=True)

nov17()

