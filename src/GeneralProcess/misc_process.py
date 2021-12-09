"""
Collection of miscellaneous processing methods 
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit 

cmap = plt.cm.get_cmap("gist_rainbow")

def reduce_rampdt(file, dv=1, 
    filename=None, save=False,  re_save=False, output_dir=r"C:\\Users\\delbe\\Downloads\\wut\\wut\\Post_grad\\UBC\\Research\\lab\\Github_repos\\hcn-gating-kinetics\\output\\Processing\\Processed_Time_Courses\\", show=False):
    """
    Reduce `ramp_dt` data to satisfy a voltage difference between timepoints equal to `dv`. The output has 3*N columns, where N is the number of traces. Each ith trace is arranged in triplets: [time, observable, voltage command].   
    
    `file` = dataframe or path to .csv file that can be read as a dataframe. The file/dataframe should have time in the index column, current in the first N columns, and voltage command in the following N columns.    
    `dv` = difference between voltages at two timepoints  
    
    `filename` = filename, only needed if saving output to a .csv file    
    `save` = whether output will be saved  
    `re_save` = whether to re-write the input file. Usually do this to rewrite column names with ramp durations, instead of running the main process pipeline entirely.  
    `output_dir` = where the reduced dataframe will be saved.  
    `show` = whether to plot output  
    """
    if isinstance(file, str):
        df = pd.read_csv(file, header=0, index_col=0)
    
    # number of traces 
    N = int(df.shape[1]/2)
    # time values 
    ts = df.index.values 
    
    # find ramp half-durations
    tmids = [int((df.iloc[:,i].dropna().shape[0]+1)/2) for i in range(N)]
    
    new_colnames = [str(x) + "_i" for x in tmids]
    new_colnames.extend([str(x) + "_v" for x in tmids])
    df.columns = new_colnames 
    
    resampled = [] 
    for i in range(N):
        df_i = df.iloc[:,[i,N+i]].dropna() 
        n0 = df_i.shape[0] 
        
        dv0 = df_i.iat[1,1] - df_i.iat[0,1] 
        
        # number of data points that lie in the `dv` interval 
        nt = abs(int(dv/dv0))

        # resample time and observables to satisfy `dv`
        df_i = df_i.iloc[::nt, :].dropna() 
        
        resampled.append(pd.Series(df_i.index))
        resampled.append(df_i)
        
        print(" dv: {dv0} -> {dv}.".format(dv0=dv0, dv=df_i.iat[1,1] - df_i.iat[0,1] ))
        print(" Data points: {n0} -> {n1}".format(n0=n0, n1=df_i.shape[0]))
        
    df_merge = pd.concat(resampled, axis=1).apply(lambda x: pd.Series(x.dropna().values))
    
    new_colnames2 = []
    for i, x in enumerate(tmids):
        new_colnames2.append(str(x) + "_t")
        new_colnames2.append(str(x) + "_i")
        new_colnames2.append(str(x) + "_v")
    df_merge.columns = new_colnames2
    
    print(filename)
    print(df_merge.head)
            
    if show:
        f = plt.figure(constrained_layout=True, figsize=(10, 5))
        gs = f.add_gridspec(7, 1)
        ax1 = f.add_subplot(gs[:5, 0])
        ax2 = f.add_subplot(gs[5:, 0])
        
        clrs = [cmap((i+1) / N*1.1) for i in range(N)]
        for i in range(0, df_merge.shape[1], 3):
            j = int(i/3)
            
            ax1.plot(df_merge.iloc[:,i], df_merge.iloc[:, i+1], marker='o', ls='none', c=clrs[j], alpha=0.7)
            ax2.plot(df_merge.iloc[:,i], df_merge.iloc[:, i+2], marker='o', ls='none', c=clrs[j], alpha=0.7)
            
            ax1.plot(df.index, df.iloc[:,j], lw=2, c=clrs[j], alpha=0.7)
            ax2.plot(df.index, df.iloc[:,N+j], lw=2, c=clrs[j], alpha=0.7)
        
        ax1.set_xticklabels([])
        
        plt.show()
        plt.close()
    
    if re_save:
        df.to_csv(file)
    
    if save:
        if output_dir is None:
            print(" `save = True`, but `output_dir` is None.")
        elif filename is None:
            print(" `save = True`, but `filename` is None.")
        else:
            df_merge.to_csv(output_dir + "{fname}_reduced-dv.csv".format(fname=filename))
    
    return df_merge 

def estimate_g(file):
    if isinstance(file, str):
        df = pd.read_csv(file, header=0, index_col=0)

    N = int(df.shape[1]/2)
    
    min_volts = [] 
    min_current = [] 
    for i in range(N):
        df_i = df.iloc[:,[i, N+i]].dropna() 
        
        min_current.append(df_i.iloc[:,0].min())
        min_volts.append(df_i.iloc[:,1].min())
    
    return min(min_current)/min(min_volts)
