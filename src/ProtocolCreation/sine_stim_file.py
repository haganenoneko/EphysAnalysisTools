"""
Create a sine wave protocol with arbitrary parameters (time offset, amplitude, frequency, etc.), plot and save as .csv. To prepare as .ATF, use `create_atf.py`
"""

import numpy as np 
import pandas as pd 
from datetime import datetime
import matplotlib.pyplot as plt

# create sine wave
def siner(t, f, A, t0=39100, y0=0):
    return A*np.sin(f*(t-t0)) + y0 

# sampling frequency in kilohertz
khz = 2

def create_sine_waves(v0, freqs, amps, durations, dx=None, y0s=[]):
    """
    Create series of sine waves with adjustable parameters. 
    
    v0 = list of y offset 
    y0 = list of y offset per sine wave 
    freqs = list of sine wave frequencies
    amps = list of sine wave amplitudes 
    durations = list of sine wave durations
    dx = time offset 
    """
    Pr_sine = []     
        
    for i in range(len(freqs)):
        f = freqs[i]
        A = amps[i]
        dt = durations[i]
        t = np.arange(0, dt, 1/khz)
        
        if len(y0s) < 1:
            y0 = [0]*len(f)
        else:
            y0 = y0s[i] 
        
        # select amplitudes such that sum = range 
        # centered about -30, range is -140 < V < +40 => 90mV each way 
        if dx is None:
            wave = v0[i] + np.sum([siner(t, f[j], A[j], y0=y0[j]) for j in range(len(f))], axis=0)
        else:
            wave = v0[i] + np.sum([siner(t, f[j], A[j], t0=dx[i], y0=y0[j]) for j in range(len(f))], axis=0)
            
        Pr_sine.append(wave)     
    
    return Pr_sine 

def create_trace(waves, vhold=-35, ti=1500, slow=False, Erev_method="fast_ramp"):
    """
    create full trace including given sine wave 
     
    waves = list of generated sine waves 
    vhold = holding potential 
    ti = initial length of time before epochs start 
    slow = if True, pre-activate with 4s -145mV. Else, pre-activate with 3s -130mV.
    Erev_method = method to estimate reversal potential.
        'fast_ramp' to use Yelhekar et al.'s method of using fast, symmetrical voltage ramps 
        'staircase' to use a series of increasingly positive voltage steps of fixed duration
    """
    max_wave = max([len(w) for w in waves])
            
    # total length of other epochs
    other = 1e4*khz + 2*ti*khz
    
    Pr_full = np.full((max_wave + int(other), len(waves)), vhold)
    print(" Total protocol length: %d samples, %.0f ms" % (Pr_full.shape[0], Pr_full.shape[0]/khz))
        
    # downwards ramp from -35 to -65
    ramp1 = np.array([-(30/100)*t - 35 for t in np.arange(0, 100, 1/khz)])
    # upwards ramp from -65 to +35 
    ramp2 = np.array([(100/300)*t - 65 for t in np.arange(0, 300, 1/khz)])
    # downwards ramp from +35 to -35 
    ramp3 = np.array([-(70/200)*t + 35 for t in np.arange(0, 200, 1/khz)])

    # large activating prepulse 
    if slow:
        # 4s at -120, then 1s at -90
        act = np.full((4000*khz,), -145)            
    else:
        act = np.full((3000*khz,), -130)    
    
    if Erev_method not in ["staircase", "fast_ramp"]:
        raise Exception("   `Erev_method` must be one of 'staircase' or 'fast_ramp'.")
    
    # staircase deactivation
    if Erev_method == "staircase":
        de = np.full((1000*khz), -40)
        
        # a linearly-spaced series of increasingly positive steps of fixed duration 
        # initial step level in staircase 
        v = -50
        # duration of each step in staircase 
        step_dt = 120*khz 
        # change in voltage with each step 
        step_dV = 15
        # list of step durations 
        steps = list(range(0, de.shape[0]-200*khz, step_dt))
        for i in steps:
            de[i:i + step_dt] = v 
            v += step_dV 
            
        # times and levels of staircase steps for tracking epochs 
        stairs = {}
        stairs.update({"t" : steps})
        stairs["t"].append(stairs["t"][-1] + step_dt) 
    
    # fast, symmetrical voltage ramps with total duration 150ms
    elif Erev_method == "fast_ramp":
        # set holding of deactivation to -10 
        de = np.full((600*khz), -10)
        
        # create ramps 
        fr = np.full((160*khz), -10)
        # timepoints of ramp 
        ts = np.arange(0, 40, 1/khz)
        fr[:40*khz] = -ts - 10
        fr[40*khz:80*khz] = ts + fr[40*khz - 1]
        fr[80*khz:120*khz] = ts + fr[80*khz - 1]
        fr[120*khz:] = -ts + fr[120*khz - 1] 
        
        # add ramps to protocol, after an initial offset of 200ms 
        de[200*khz:len(fr)+200*khz] = fr 
        # return to holding potential to allow capacitive transients to relax before commencing sine waves
        # de[300*khz + len(fr):] = -40 
        
        # epochs for tracking time, level, and type of each step 
        fr_epochs = [
            [0, -10, "Ramp"],
            [40*khz, fr[40*khz - 1], "Ramp"],
            [120*khz, fr[80*khz - 1], "Ramp"],
            [160*khz, -10, "Step"],
            # [260*khz, fr[-1], "Step"]
        ]
        # add offset to initial times in epochs 
        for i in range(len(fr_epochs)):
            fr_epochs[i][0] += 200*khz 
    
    # epoch information   
    dflist = [] 
    
    L = Pr_full.shape[1]
    for i, w in enumerate(waves):
        epochs = [] 
        
        t0 = ti*khz
        epochs.append([t0, Pr_full[t0, i], "Step"])
        
        # 3 ramps for leak subtraction 
        Pr_full[t0:t0 + len(ramp1), i] = ramp1 
        epochs.append([t0, Pr_full[t0 + len(ramp1) - 1, i], "Ramp"])
        t0 += len(ramp1)
        
        Pr_full[t0:t0 + len(ramp2), i] = ramp2
        epochs.append([t0, Pr_full[t0 + len(ramp1) - 1, i], "Ramp"])
        t0 += len(ramp2)
        
        Pr_full[t0:t0 + len(ramp3), i] = ramp3 
        epochs.append([t0, Pr_full[t0, i], "Ramp"])
        t0 += len(ramp3) 
        epochs.append([t0, Pr_full[t0, i], "Ramp"])
        t0 += 1000*khz
        
        # pre-activation 
        Pr_full[t0:t0 + len(act), i] = act 
        epochs.append([t0, Pr_full[t0, i], "Step"])
        t0 += len(act)
        
        # deactivation staircase 
        if Erev_method == "staircase":
            Pr_full[t0:t0 + len(de), i] = de 
            
            # add epochs from staircase 
            for j in range(len(stairs["t"])):
                t_j = t0 + stairs["t"][j]
                epochs.append(
                    [t_j, Pr_full[t_j, i], "Step"]
                )
            t0 += len(de) 
            
            # may not need this (step that returns to deactivation holding potential)
            # epochs.append([t0, Pr_full[t0, i], "Step"])
            
        elif Erev_method == "fast_ramp":
            Pr_full[t0:t0 + len(de), i] = de 
            
            # plt.plot(Pr_full[:t0+len(de), i])
            # plt.show()
            # exit()
            
            # add epochs from fast ramps; includes the last step (return to deactivating holding potential)
            for j, e in enumerate(fr_epochs):
                t_j = t0 + e[0] 
                epochs.append(
                    [t_j, e[1], e[2]] 
                )
            t0 += len(de) 
            
        # sine wave 
        Pr_full[t0:t0+w.shape[0], i] = w.flatten()
        epochs.append([t0, Pr_full[t0, i], "Sine"])
        t0 += w.shape[0] 
        
        # final -150 envelope test 
        if t0 < (Pr_full.shape[0] - 2000*khz):            
            Pr_full[t0:t0 + 500*khz, i] = -145
            epochs.append([t0, Pr_full[t0, i], "Step"])
            t0 += 500*khz 
        else:
            print(" Final -145 mV envelope pulse was not applied to the end of the %d-th trace due to insufficient space.\n  500 ms is required. %.0f are available." % (i, (Pr_full.shape[0] - t0)/khz ))
        
        epochs.append([t0, Pr_full[t0, i], "Step"])
        
        dflist.append(pd.DataFrame.from_records(epochs)) 
    
    df = pd.concat(dflist, axis=1) 
    return Pr_full, df 

#hcn_sine_20210214_00-03
'''Pr_sine = create_sine_waves(
        v0 = [-60]*4, 
        freqs = [
            [1.4e-5, 5.5e-4, 4.1e-3, 3.7e-2],
            [5.5e-4, 2.7e-3, 1.9e-2],
            [6.5e-4, 4.3e-3, 3.7e-2],
        ],
        amps = [
            [46, 35, 32, 7],
            [56, 33, 11],
            [53, 35, 12]
        ],
        durations = [10000, 10000, 10000]
)'''

#hcn_sine_20210214_00-33
"""Pr_sine = create_sine_waves(
        v0 = [-40, -90, -110], 
        freqs = [
            [1.3e-5, 5.3e-4, 2.1e-3, 4.7e-3],
            [1.5e-4, 2.9e-3, 7.7e-3],
            [1.1e-4, 4.7e-3, 8.3e-3],
        ],
        amps = [
            [26, 35, 42, 57],
            [23, 36, 47],
            [41, 35, 24]
        ],
        durations = [10000, 10000, 10000]
)"""

#hcn_sine_20210214_01-48
"""Pr_sine = create_sine_waves(
        v0 = [-55, -55],
        freqs = [
            [1.3e-4, 7.1e-4, 2.2e-3, 9.7e-3],
            [7.5e-4, 2.2e-4, 2.7e-3],
        ],
        amps = [
            [22, -38, 32, 8],
            [43, -11, 26],
        ],
        durations = [20000, 20000],
        dx = [0]*2
)"""

#hcn_sine_20210219_10-44
"""Pr_sine = create_sine_waves(
        v0 = [-40, -40],
        freqs = [
            [2.5e-4, 2.9e-3, 8.3e-3],
            [1e-3, 2.7e-3],
        ],
        amps = [
            [48, -22, 10],
            [45, 30],
        ],
        durations = [26100, 19540],
        dx = [0]*2
)"""

Pr_sine = create_sine_waves(
        v0 = [-50, -50],
        freqs = [
            [3.33e-4, 2.9e-3, 7.57e-3],
            [1.91e-3, 2.7e-3],
        ],
        amps = [
            [-48, -19, 13],
            [45, 30],
        ],
        durations = [19540, 19540],
        dx = [0]*2,
#         y0s = [[-50, 0, -10], [-10, 0]]
)

# current datetime 
now = datetime.now().strftime("%Y%m%d_%H-%M")
# save to protocols 
save_dir = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/Github_repos/hcn-gating-kinetics/data/protocols/"
save_path = save_dir + "hcn_sine_%s.csv"  % now 
# save_path = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/Github_repos/hcn-gating-kinetics/data/protocols/hcn_sine_%s.csv"  % "20210214_01-48"  

Pr_full, df_pro = create_trace(Pr_sine, slow=True, Erev_method="fast_ramp")    

# save .csv file with information on epoch times and levels (workaround for when Analog OUT #0 error occurs)
df_pro.to_csv(save_dir + "hcn_sine_20210304_01-10_epochs.csv")

# Pr_full = pd.read_csv(save_path, index_col=0, header=0)
# print(Pr_full)
exit()
    
# visualize protocol 
f, ax = plt.subplots(figsize=(12,4))
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (mV)")
ax.plot(np.arange(0, Pr_full.shape[0]/khz, 1/khz), Pr_full, lw=3, alpha=0.7)

cmap = plt.cm.get_cmap("gist_rainbow")
for j in range(0, df_pro.shape[1], 3):
    clr = cmap(j/df_pro.shape[1])
    
    for i in range(df_pro.shape[0]):
        ax.axvline(df_pro.iloc[i,j]/khz, c=clr, lw=2, alpha=0.4)

ax.set_title("20210304_01-10")
# ax.set_title(now)
plt.tight_layout()

plt.savefig(save_path[:-4] + ".png")
plt.show()
exit()

# add time column to protocol array 
Pr_full = np.c_[np.arange(0, Pr_full.shape[0]/khz, 1/khz), Pr_full]
# print(Pr_full.shape[0]/khz)
# print(Pr_full[:5,:])
# exit()

# save protocol array to csv (this is used to make the atf)
# np.savetxt(save_path, Pr_full, delimiter=",")