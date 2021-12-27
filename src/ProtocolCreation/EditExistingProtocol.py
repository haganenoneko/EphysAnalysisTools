"""
Methods to edit existing protocols to add:
    - new traces
    - reversal ramp (activating prepulse + deactivating ramp)
    - interleaved pulse trains 
"""
import pyabf 
import numpy as np 
import pandas as pd 
import glob 
import os 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("dark_background")
cmap = plt.cm.get_cmap("coolwarm")

# where output will be saved, if saving is enabled 
out_dir = "./data/protocols/"
# whether to save output 
save_output = False 

def find_abf_epochs(fname):
    data_dir = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/data_files/delbert/"
    fname += ".abf"
    
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if fname in name:
                path = os.path.join(root, name)
                break 
            
    # open abf file 
    abf = pyabf.ABF(path)
    
    # return epochs 
    p1s = []
    levels = [] 
    for i in abf.sweepList:
        abf.setSweep(i)
        p1s.append(abf.sweepEpochs.p1s)
        levels.append(abf.sweepEpochs.levels)
    
    return p1s, levels 

class edit_existing_protocol():
    def __init__(self, fname, out_name, csv_path=None):
        
        if csv_path is None:
            csv_path = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/Github_repos/hcn-gating-kinetics/data/current_time_course/Pooled_2020/"
                
        try:
            df = pd.read_csv(csv_path + fname + ".csv", header=None, index_col=0)
        
            # extract voltage command 
            N = int(df.shape[1]/2)
            
            # check that there are equal number of current and voltage sweeps 
            if df.shape[1] != 2*N:
                print(" Uneven number of current/voltage sweeps. Deleting last voltage sweep.")
                df = df.iloc[:,:-1]
                N = int(df.shape[1]/2)
        
            # voltage protocol 
            self.df = df.iloc[:,N:]
                
        except:
            print(" %s not found in csv_path." % fname)
            
            try:
                # CellML csv export 
                df = pd.read_csv(fname, header=0, index_col=None)
                df.index *= 1/2
                self.df = df 
                                
            except:
                print(" Could not open `fname` as file.")
                exit()
                
        # sampling frequency 
        self.khz = int( 1/(df.index[1] - df.index[0])) 
        
        self.fname = fname 
        self.out_name = out_name 
        
    def create_reversal_ramp(self, Vact=-120, Tact=3000, Vramp=[-50, 10], Tramp=150, Tcap=10):
        """
        Create steps for reversal ramp: maximally activating prepulse, followed by deactivating ramp.
        
        `Vact` = prepulse voltage \\
        `Tact` = duration of activation prepulse \\ 
        `Vramp` = start and end voltages of deactivating ramp \\
        `Tramp` = duration of reversal ramp 
        `Tcap` = short pulse of same voltage as `Vramp[0]` to cancel capacitive currents 
        
        Returns `ramp`, array containing prepulse and ramp command 
        """
        
        # duration of capacitive step 
        Tcap = 20
        # slope of reversal ramp 
        dvdt = (Vramp[1] - Vramp[0])/Tramp
        # times of reversal ramp 
        Tramp = np.arange(0, Tramp+Tcap, 1/self.khz)
        
        # convert Tcap to samples 
        Tcap = int(Tcap * self.khz)        
        
        ramp = Tramp.copy()
        ramp[:Tcap] = Vramp[0] 
        ramp[Tcap:] = [(dvdt*(t-Tramp[Tcap]) + Vramp[0]) for t in Tramp[Tcap:]]
        
        return ramp 
    
    def create_leak_ramp(self, volts=[-35, 35], thalf=500, khz=2, add_MT_step=True):
        """
        Create array of voltages for leak ramp 
        """
        if add_MT_step:
            out = np.zeros((thalf*2*khz + 2000*khz,))
        else:
            out = np.zeros((thalf*2*khz,))
        
        out[:1000*khz] = -35 
        
        ts = np.arange(0, thalf, 1/khz)        
        out[1000*khz:(thalf+1000)*khz] = (ts*(volts[1]-volts[0])/(thalf)) + volts[0]
        
        if add_MT_step:
            out[(thalf+1000)*khz:-1000*khz] = (ts*(volts[0]-volts[1])/thalf) + volts[1]
            
            t = 1000 
            while t > 400:
                out[-t*khz:-(t-200)*khz] = -35 
                out[-(t-200)*khz:-(t-400)*khz] = 20 
                
                t -= 400 
            
            out[-t*khz:] = -35
            
        else:
            out[(thalf+1000)*khz:] = (ts*(volts[0]-volts[1])/thalf) + volts[1]
        
        return out 
    
    def add_leak_ramp(self):
        leak_ramp = self.create_leak_ramp(khz=int(1/self.df.index[1]))
        
        out = pd.DataFrame(np.array([leak_ramp,]*self.df.shape[1])).T        
        try:
            out = pd.concat([out, self.df], axis=0, ignore_index=True).reset_index(drop=True)
        except:
            exit()
        out.index *= 1/self.khz 
        
        plt.plot(out.index * 1e-3, out)
        plt.show()
        
        return out 
    
    def add_traces(self, N=2, addto=-1):
        """
        Returns start and stop points for varying-level pulse, assuming two-step protocol. 
        
        `N` = number of traces to add \\
        `addto` = adds to the end if `addto = -1` or to the beginning if `addto = 0`. New traces take duration of nth pulse, where `n = addto` if `addto` is 0 or -1. 
        """
        if addto in [0, -1]:
            pass
        else:
            raise Exception("   `addto` must be one of 0 (add new traces to the beginning) or -1 (add new traces to the end).")
            exit()
        
        # get pulse durations and levels addto abf file 
        p1s, levels = find_abf_epochs(self.fname)
        
        # find indices of varying-level pulses in each sweep 
        dL = [[j for j in range(len(L)) if (L[j] - levels[0][j]) != 0][0] for L in levels[1:]]
        # print(" Indices of varying-level pulses in protocol. \n", dL)
        
        if any(x != dL[0] for x in dL[1:]):
            raise Exception("   Indices of varying-level pulses are not the same.")
            exit()
        
        # index of varying-level pulse 
        idx = dL[0]
        # difference in level between varying-level pulses 
        dv = abs(levels[0][idx] - levels[1][idx])
        
        # copy N columns addto self.df
        new_sweeps = self.df.iloc[:,-N:].copy()
        
        # add new traces addto end of protocol 
        if addto < 0:
            # start and stop points of varying-level pulse 
            t0, t1, t2 = p1s[-1][idx:idx+3]
            
            # initial voltage of new traces 
            # v0 = levels[-1][idx] + dv
            v0 = levels[-1][idx] 
        else:
            t0, t1, t2 = p1s[0][idx:idx+3]
            v0 = levels[0][idx]
        
        # add N sweeps of increasing voltage addto the end of the recording 
        for i in range(N):
            new_sweeps.iloc[t0:t1+1,i] = v0 
            v0 += dv 
            
            # halve duration of deactivating pulse
            # dt = int((t2 - t1)/2) + t1 + 1
            # new_sweeps.iloc[dt:,i] = -35 
            
            # remove leak ramp
            new_sweeps.iloc[t2:, i] = -35
            
            # plt.plot(new_sweeps.iloc[:,i])
            
        # plt.show()
        # exit()
    
        # concatenate new sweeps into self.df 
        if addto < 0:
            self.df = pd.concat([self.df, new_sweeps], axis=1)
        else:
            self.df = pd.concat([new_sweeps, self.df], axis=1)
        
        # remove post-deactivation ramps in all sweeps 
        for i in range(self.df.shape[1]-N):
            t2 = p1s[i][idx+2]
            self.df.iloc[t2+1:,i] = -35 
            
    def add_interleaved_train(self, numlist=[], period=4000, volt=0, dt=500, spacing=2):
        """
        Interleave sweeps of `self.df` with sweep of same total duration containing trains of fixed-druation, fixed-voltage steps 
        
        `numlist` = list of indices after which train will be inserted; if empty, uses `spacing` to interleave trains instead \\ 
        `period` = time between steps \\
        `volt` = level of steps \\
        `dt` = duration of steps \\
        `spacing` = how to add sweeps, e.g. between every nth 
        """
        period = int(period * self.khz) 
        dt = int(dt * self.khz)
        
        # number of sweeps 
        N = int(self.df.shape[1]/spacing)
        # don't add trains after final test sweep 
        if N*spacing > self.df.shape[1]:
            N -= 1 
            
        # create trains 
        train = np.ones((self.df.shape[0], 1)) * -35 
        
        # padding between start and end of sweep
        pad = int(5000 * self.khz)
        
        def create_train(v):
            # onset of first step 
            t0 = pad 
            while t0 < (len(train) - pad):
                train[t0:t0+dt+1] = v
                t0 += period + 1 
        
        if not isinstance(volt, list):
            create_train(volt)
        
        # count number of added trains 
        t = 0 
        if len(numlist) > 0:
            for (i, n) in enumerate(numlist):
                if isinstance(volt, list):
                    create_train(volt[i])
                    
                self.df.insert(loc=n+t, column="Train%d" % t, value=train)  
                t += 1 
        else:
            for i in range(1, self.df.shape[1], spacing):
                if isinstance(volt, list):
                    create_train(volt[i-1])
                    
                self.df.insert(loc=i+int(), column="Train%d" % t, value=train)
                t += 1 
        
    def add_reversal_ramp(self, traces=[], Vact=-115, Tact=2000, 
                            Vramp=[-50, 20], Tramp=140, SS=15000,
                            ramp_spacing = -1,
                            save_output=save_output, add_new=False):
        """
        Add reversal ramp to voltage protocol given by `df`
        
        `traces` = indices of sweeps to add reversal ramp to. If empty, added to all sweeps, if possible, i.e. if enough time for both original protocol and satisfying sweep-to-sweep interval `SS`. \\
        `Vact` = activation potential \\
        `Tact` = duration of activation step \\
        `Vramp` = start and end voltages of reversal ramp \\
        `Tramp` = duration of reversal ramp. \\
        `SS` = sweep-to-sweep interval \\ 
        `ramp_spacing` = non-zero; every `nth` sweep to add reversal ramp to. 
            If 1, adds to the first only. 
            If -1, adds to the first and last. 
            If -2, adds to the last only.
            Else, adds every nth.
        
        NOTE: A capacitive pulse of 10ms at Vramp[0] is added before the reversal ramp to account for capacitive currents. 
        """
        
        # minimum number of samples to add reversal ramp to the end of the protocol 
        total = int((Tact + Tramp*1.1 + SS)*self.khz) 
        
        # create reversal ramp 
        ramp = self.create_reversal_ramp(Vact=Vact, Tact=Tact, Vramp=Vramp, Tramp=Tramp)
        Tact = int(Tact * self.khz)
        Tramp = ramp.shape[0] 
        
        # indices of sweeps to add reversal ramp to 
        if ramp_spacing == 0:
            raise Exception("   `ramp_spacing` must be non-zero.")
            exit()
        else:
            if ramp_spacing > 1:
                to_add = list(range(0, self.df.shape[1], ramp_spacing))
            elif ramp_spacing == 1:
                to_add = [0]
            elif ramp_spacing == -1:
                to_add = [0, self.df.shape[1]-1]
            elif ramp_spacing == -2:
                to_add = [self.df.shape[1]-1]
        
        if self.df.shape[1]-1 in to_add:
            pass 
        else:
            to_add.append(self.df.shape[1]-1)
        
        if add_new:
            out = np.zeros((Tact+Tramp+2000*self.khz, ))
            
            out[:Tact] = Vact 
            out[Tact:-2000*self.khz] = ramp 
            out[-2000*self.khz:] = -35 
            
            out = pd.DataFrame(out)
            print(out.shape) 
            
            self.df = pd.concat([self.df, out], axis=0).reset_index(drop=True)
            self.df.index *= 1/self.khz       
            
            return None 
        
        added = [] 
        for i in to_add:
            # boolean mask to test condition that enough holding voltages are present at the end of the ith sweep 
            mask = (self.df.iloc[-total:,i] + 35).abs() < 3
            
            if mask.all():
                t0 = Tact - total 
                self.df.iloc[-total:t0, i] = Vact 
                self.df.iloc[t0:t0 + Tramp, i] = ramp 
                # t0 += Tramp 
                # self.df.iloc[t0:t0 + 2000, i] = 0
                print("     Reversal ramp added to %dth sweep." % i)
                added.append(i+1)
 
            else:
                print("     Insufficient holding points in %d-th sweep. Reversal ramp not added." % i)
                continue 
   
    def plotter(sweep_offset=True, save_output=True):
        """
        Plot `self.df` \\
        `sweep_offset` = whether to offset sweeps along z axis 
        """
        
        if sweep_offset:
            f = plt.figure(figsize=(12, 7))
            ax = f.gca(projection='3d')
            
            x = self.df.index.values * 1e-3
            dt = x[-1] 
            z = np.ones(len(x))
            
            i = 0
            while i < self.df.shape[1]:
                ax.plot(x + i*dt*0.7, z+i, self.df.iloc[:,i], lw=2)
                i += 1
            
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Sweep #")   
            ax.set_zlabel("Voltage (mV)")
            
            # make pane transparent 
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
            # remove grid lines      
            ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)       
            
            # remove spines, interfere with tight_layout 
            for spine in ax.spines.values():
                spine.set_visible(False)
                
        else:
            f, ax = plt.subplots(figsize=(9,4))
            ax.set_title("Reversal ramp added to " + str(added))
            
            for i in range(self.df.shape[1]):
                clr = cmap((i+1)/self.df.shape[1])
                ax.plot(self.df.iloc[:,i], lw=2, c=clr)
                        
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Voltage (mV)")
    
        if save_output:
            plt.savefig(out_dir + self.out_name + ".png", dpi=300)
            
        plt.tight_layout()
        plt.show()   

    def go(self, add_reversal_ramp={}, 
            add_interleaved_train = {},
            add_traces={}):
        """
        Apply class methods to `self.df` \\
        
        `add_reversal_ramp` = dict of args to `self.add_reversal_ramp`
        `add_interleaved_train` = dict of args to `self.add_interleaved_train`
        `add_traces` = dict of args to `self.add_traces`
        """
        if len(add_traces.keys()) > 0:
            self.add_traces(**add_traces)
        
        if len(add_reversal_ramp.keys()) > 0:
            self.add_reversal_ramp(**add_reversal_ramp)
        
        if len(add_interleaved_train.keys()) > 0:
            self.add_interleaved_train(**add_interleaved_train)
                    
    def out(self):
        #reset column names 
        self.df.columns = list(range(self.df.shape[1]))        
        print(" Shape of df:", self.df.shape)
        print(self.df.head)
                
        save_path = out_dir + self.out_name + ".csv"
        self.df.to_csv(save_path)
        print(" Successfully saved csv at: \n", save_path)

d = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/Github_repos/hcn-gating-kinetics/data/CellML_models/severi2012_default_40s.csv"
e = edit_existing_protocol(d, "severi2012_pacing")
e.add_reversal_ramp(SS=50000, Vact=-130, add_new=True)
e.add_leak_ramp()

# 20o16004 = FA act stag R -145up
# 21326015 = WT act stag R -145up 
# m = process("21219005", out_name="FA_de_RR_IL")
# m.add_reversal_ramp(SS=40000)
# if save_output:
#     m.out()