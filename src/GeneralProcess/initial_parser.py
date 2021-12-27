# Copyright (c) 2021 Delbert Yip
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os, glob
 
import numpy as np 
import pandas as pd

from typing import List, Dict, Tuple, Any

from GeneralProcess.base import NDArrayFloat
from GeneralProcess.base import AbstractRecording, AbstractAnalyzer

from GeneralProcess.ephys_info_filter import EphysInfoFiltering

# ---------------------------------------------------------------------------- #

class BaseProcessor:
    """Select and find CSV and ABF files"""
    def __init__(
        self, main_dir: str, csv_path: str, abf_path: str,
        filter_criteria: Dict[str, Any]
    ) -> None:
        
        if not filter_criteria:
            raise ValueError(f"No filter criteria provided.")
        
        self._criteria = filter_criteria 
        self.readFilterCriteria()
        
    def readFilterCriteria(self) -> None:
        ephysInfo = EphysInfoFiltering(self._criteria)
        filenames, ephys_info = ephysInfo.filter()
        paired_files, exp_params = ephysInfo.ExpParams(ephys_info)
        
        self._filenames = filenames
        self._ephysInfo = ephys_info 
        self._expParams = exp_params 
        self._pairedFiles = paired_files
        
    @staticmethod 
    def readCSVFile(file: str, filename: str) -> pd.DataFrame:
        df = pd.read_csv(file, header=None, index_col=0)

        # check if first index is not 0
        if df.index[0] != 0:
            
            try: 
                df.index = df.index.astype(float)
                df.index -= df.index[1]
            except:
                df = df.iloc[1:, :]
                df.index = df.index.astype(float)
                
        return file_specific_transform(filename, df=df)

    def readCSVFiles(self, csv_path: str) -> List[int]:
        """Load csv files containing corresponding data"""
        
        files = [f"{csv_path}{f}.csv" for f in self._filenames]
        
        data_files = {}

        # indices of CSV files that weren't found
        to_remove = []  
        removal_msg = "CSV file not found for {0}.\n\
                Removing {0} from processing."
        
        for i, tup in enumerate(zip(files, self._filenames)):
            
            file, fname = tup 
            
            if os.path.isfile(file):
                df = self._readCSVFile(file, self.fname)
                data_files.update({fname : df})
                continue 
            
            to_remove.append(i)
            print(removal_msg.format(fname))

        self._data_csv_files = data_files 
        return to_remove
    
    def removeMissingCSVFiles(self, to_remove: List[str]) -> None:
        """Remove file not found from dataframes of recording parameters"""
        
        if not to_remove: return 
        
        ephysInfo = self._ephysInfo 
        expParams = self._expParams
        
        missing_files = [self._filenames[i] for i in to_remove]
        ephysInfo = ephysInfo[~ephysInfo['Files'].str.isin(missing_files)]
        expParams = expParams.loc[:, ~expParams.str.isin(missing_files)]
        
        print(f"{len(to_remove)} were removed.\n\
            Files kept: \n", ephysInfo.loc[:, ["Files", "Protocol"]])
        
        self._filenames = [f for f in self._filenames if f not in missing_files]
        self._ephysInfo = ephysInfo
        self._expParams = expParams
    
    def findABFFiles(self, abf_path):
        # path to raw abf files
        abf_path = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/data_files/delbert/" 

        # search subdirectories of entire `abf_path` for the original .abf files
        abf_files = [
            glob.glob(abf_path + "**/%s*.abf" % f, recursive=True) for f in filenames
        ]
        abf_files = [pyabf.ABF(a[0]) for a in abf_files]

        # check that abf files were found
        if len(abf_files) != len(filenames):
            print("Number of filenames is %d, but only found %d abf files" % (
                len(filenames), len(abf_files)))
            exit()


def _init__(self,
    filter_criteria={},
    show_protocols=True,
    files_to_skip=[],
    dates_to_save='None',
    show_abf_segments=False,
    show_csv_segments=False,
    show_leak_subtraction=False,
    do_pseudoleak_subtraction=False,
    do_LJP_correction=(False, 3.178),
    VoltageClampQuality=False,
    show_Cm_estimation=False,
    show_MT_estimation=False,
    do_exp_kinetics={},
    manual_cap_offset={},
    do_activation_curves={},
    do_ramp_stuff=False,
    do_inst_IV=False,
    save_AggregatedPDF=False,
    do_pubplots={},
    normalize=False,
    remove_after_normalize={}
    ):


        
       
        # create dictionary that holds, for each unique filename, start and end of first leak ramp
        self.ramp_startend = {}
        # dict to hold start and end of +20mV membrane test steps
        self.mt_startend = {}
        # dictionary to hold filename : {epoch1:[t0, t1], ...}
        self.epoch_startends = {}
        # dictionary to hold sampling frequencies in kHz
        self.dataRates = {}

        # loop over .ABF files found, and find start and end of leak ramps
        for i, a in enumerate(abf_files):
            print("\n Reading...", filenames[i])
            df_data = data_files[filenames[i]]

            # sampling frequency in kHz
            khz = int(a.dataRate / 1000)
            self.dataRates.update({filenames[i]: khz})

            # number of sweeps in ith recording
            N = int(df_data.shape[1]/2)

            # check that protocol is contained in CSV file
            # 1. even number of columns in CSV file
            # 2. voltage is in range [-200, +100]
            try:
                HasProtocol = (df_data.shape[1] == 2*N) and \
                    (df_data.iloc[:, N:].min() >= -200).all() and \
                    (df_data.iloc[:, N:].max() <= 100).all()
            except:
                HasProtocol = False

            # check if epochs are available
            # 1. max epoch time exceeds duration of data
            # 2. sum of abf.sweepC and data_df.iloc[:,N] are nearly equal
            # note that 2) doesn't work unless `df_data.iloc[:,N]` is a voltage command
            a.setSweep(0)
            EpochTimes = a.sweepEpochs.p1s
            EpochLevels = a.sweepEpochs.levels
            EpochTypes = a.sweepEpochs.types

            try:
                HasEpochs = (df_data.shape[0] >= EpochTimes[-1]) or \
                    (abs(df_data.iloc[:, N].sum()/sum(a.sweepC)) > 0.9)
            except:
                HasEpochs = False

            # find epochs for custom protocol if used
            if HasEpochs and HasProtocol:
                pass
            else:
                print(
                    "Uneven number of columns of CSV. Protocol may be missing.\
                    \n First row: \n", df_data.iloc[0, :]
                )

                # protocol name
                pname = ephys_info.loc[ephys_info["Files"] == filenames[i],
                                       "Protocol"].values[0]

                # check if protocol corresponds to an existing CSV file
                if os.path.isfile(r"./data/protocols/%s.csv" % pname):

                    # retrieve epochs for custom protocol, either in CSV, or automated
                    CustomEpochs = FindCustomEpochs(
                        df_data, pname, filenames[i],
                        a, show_epochs=False, save=False
                    )

                    # abf variables (sweep epochs, times, levels, protocol time course)
                    a = CustomEpochs.ReplaceABFVariables(test=True)
                    abf_files[i] = a
                    khz = a.dataRate

                    # redefine epoch properties for first sweep
                    a.setSweep(0)
                    EpochTimes = a.sweepEpochs.p1s
                    EpochLevels = a.sweepEpochs.levels
                    EpochTypes = a.sweepEpochs.types

                    # redefine `df` if it doesn't contain protocol
                    if CustomEpochs.HasProtocol == False:
                        df_data = CustomEpochs.df
                        data_files[filenames[i]] = df_data

                        # redfine number of sweeps
                        N = int(CustomEpochs.df.shape[1]/2)

                    print("pyABF object replaced.")

                else:
                    raise Exception(
                        "`Protocol` entry in `ephys_info` < %s > does not correspond\
                        to a CSV file at './data/protocols/'. If it is a custom protocol,\
                        then make sure such a file is available." % pname
                    )
                        
            # apply LJP correction
            # for pyABF (non-custom protocol) objects, a DummyABF object is created,
            # so some slowness is expected
            if all(x > 0 for x in do_LJP_correction):
                # subtract from voltage command
                df_data.iloc[:, N:] -= do_LJP_correction[1]

                # create DummyABF object to replace original pyABF
                # because, pyABF object doesn't mutate well
                a = DummyABF(a, [], [], khz)

                # apply LJP subtraction to each sweep of voltage command in ABF,
                # regardless of whether it's included in CSV file or not
                a.pro -= do_LJP_correction[1]
                for j in a.sweepList:
                    a.EpochLevels[j] -= do_LJP_correction[1]

                # redefine 0-th sweep properties for finding membrane test and leak ramps
                a.setSweep(0)
                EpochTimes = a.sweepEpochs.p1s
                EpochLevels = a.sweepEpochs.levels
                EpochTypes = a.sweepEpochs.types

                # replace in dictionary
                data_files[filenames[i]] = df_data

            # find first voltage ramp for the first trace, then assume this is constant
            # plot segmented epochs from protocol
            if show_abf_segments:
                f, ax = plt.subplots(2, 1, figsize=(8, 6))

                for sweepNumber in a.sweepList:
                    a.setSweep(sweepNumber)

                    times = np.array(a.sweepX) * 1000
                    ax[0].plot(times[::2], a.sweepY[::2],
                               c='gray', alpha=0.4)
                    ax[1].plot(times[::2], a.sweepC[::2], c='r', alpha=0.4)

                    # print("Step times... \n", a.sweepEpochs.p1s)
                    for x, p1 in enumerate(a.sweepEpochs.p1s):
                        for j in range(2):
                            ax[j].axvline(p1/khz, c='white', ls='--', lw=2)

                        # ax[1].axhline(a.sweepEpochs.levels[x], c='w', ls='--', alpha=0.3)

                ax[0].set_title("show_abf_segments: %s" % filenames[i])
                ax[0].set_ylabel("Current")
                ax[1].set_ylabel("Command Voltage")
                ax[1].set_xlabel("Time")
                plt.tight_layout()

                print(
                    "Showing ABF segments. White dashes show epochs in voltage protocol.")
                plt.show()
                plt.close()

            # find indices for the start and end of the leak ramps
            if "Ramp" in EpochTypes[:10]:
                # find index of first ramp epoch
                # fri = 'first ramp index'
                fri = EpochTypes[:10].index('Ramp')
                # print(EpochTypes[:10], fri)

                # last index of initial ramps; lri = 'last ramp index'
                lri = fri + 1
                while EpochTypes[lri+1] == 'Ramp' and lri <= 10:
                    lri += 1

                t0 = EpochTimes[fri]
                t1 = EpochTimes[lri+1]

                self.ramp_startend.update(
                    {filenames[i]: file_specific_transform(
                        filenames[i], times=[t0, t1])}
                )
            else:
                print("No leak ramp found in first 10 steps of %s." %
                      filenames[i])

                # no leak ramp is present, so we set to `None` to facilitate later processing
                fri = None
                lri = None

            # check for presence of membrane test step and when relevant protocol epochs start
            # if no leak ramp, then we skip testing for membrane test
            if fri is None:

                # when no leak ramp is found, we set `fri` = 0
                # this causes epochs to be cataloged starting from the very beginning
                fri = 0

                # we pre-emptively set `dt_` to None so we can look for it later
                dt_ = None

            else:
                # check if membrane test step is present within 1-5 epochs after the ramp
                # 1. duration < 1s
                # 2. level > 0 mV
                # 3. type = step

                mt_idx = next(
                    (j for j in range(1, 3) if EpochLevels[lri+j] == 20 and
                     (EpochTimes[lri+j+1] - EpochTimes[lri+j]) <= 500*khz and
                     EpochTypes[lri+j] != "Ramp"),
                    None
                )
                
                if mt_idx is None:
                    dt_ = None
                else:
                    mt_idx += lri

                    dt_ = mt_idx + next(
                        (j for j in range(1, 3) if
                         EpochLevels[mt_idx+j] != EpochLevels[0]), 1
                    )

                    self.mt_startend.update({
                        filenames[i]:
                        EpochTimes[mt_idx:dt_]
                    })

            # `dt_` = number of epochs b/w end of leak ramp and start of protocols
            # if `dt_` not found based on where leak ramp and membrane test are,
            # then relevant pulses start after the last -35mV step
            if dt_ is None:
                if lri is None:
                    dt_ = 1 + next(
                        (j for j in range(1, 3)
                         if EpochLevels[j+1] != EpochLevels[0]),
                        0)
                else:
                    dt_ = lri + next(
                        (j for j in range(1, 3)
                         if EpochLevels[j+lri] != EpochLevels[0]),
                        0)
            
            if isinstance(dt_, int):
                print(" %d epochs to start of relevant protocol" % dt_)
            else:
                raise Exception("`dt_` is not an integer: \n ", dt_)

            # 0-indexed list of indices of sweeps, e.g. range(# sweeps)
            # print(a.sweepList)

            # add epoch start and end times for each sweep
            # if csv and abf are 1:1, `h` stays 0
            # else, `h += 1` so that we keep indexing csv at the jth column 
            # while proceeding through the abf file
            h = 0

            # add epoch start and end times for each sweep
            for j in a.sweepList:
                a.setSweep(j)

                # we don't need to compare CSV and ABF sweeps if the number of sweeps are equal
                if N < len(a.sweepList):

                    # don't exceed number of traces in the .csv file
                    if (N + j - h) >= 2*N:
                        print("Exceeded CSV columns while reading epoch intervals")
                        break

                    # sum ABF voltage command
                    abf_sum = np.sum(a.sweepC)
                    if not isinstance(abf_sum, float):
                        raise Exception(
                            "`abf_sum` is not a float.\
                            Check the type of `a.sweepC` : ", type(a.sweepC)
                        )

                    # sum jth sweep of voltage protocol
                    csv_sum = df_data.iloc[:, N+j-h].sum()

                    # apply file-specific transform if necessary to enforce equality
                    csv_sum = file_specific_transform(
                        filenames[i], times=(abf_sum, csv_sum))

                    # print(df_data.iloc[0,:])
                    # plt.plot(df_data.index, a.sweepC, ls='--', lw=2)
                    # plt.plot(df_data.iloc[:,N+j-h], alpha=0.5)
                    # plt.title("abf = %d, csv = %d" % (j, N+j-h))
                    # plt.show()

                    # check if jth sweep is in both abf and csv files by
                    # subtracting their sums (voltage commands)
                    # print(j, h, ((abf_sum - csv_sum)/abf_sum))
                    if abs((abf_sum - csv_sum)/abf_sum) < 0.01:
                        if filenames[i] in self.epoch_startends.keys():
                            self.epoch_startends[filenames[i]].update({
                                j: a.sweepEpochs.p1s[dt_:]
                            })
                        else:
                            self.epoch_startends.update({
                                filenames[i]: {
                                    j: a.sweepEpochs.p1s[dt_:]
                                }
                            })
                    # if the jth sweep isn't found in the .csv file, we skip it
                    else:
                        h += 1
                    
                else:

                    if filenames[i] in self.epoch_startends.keys():
                        self.epoch_startends[filenames[i]].update({
                            j: a.sweepEpochs.p1s[dt_:]
                        })
                    else:
                        self.epoch_startends.update({
                            filenames[i]: {
                                j: a.sweepEpochs.p1s[dt_:]
                            }
                        })

            # verify that epochs have been added for the current file
            if filenames[i] not in self.epoch_startends.keys():
                raise Exception(
                    "No epochs were saved for file < %s >,\
                    this may mean that the protocol was not saved in the CSV file.\
                    if a custom protocol was used, check that the protocol was parsed correctly. \
                    finally, if LJP subtraction is enabled, check that it is applied equally for both CSV and ABF data." % filenames[i]
                )
            
            # check same number of sweep intervals saved as number of sweeps 
            if len(self.epoch_startends[filenames[i]].keys()) < df_data.shape[1]/2:
                print(df_data.shape)
                print(self.epoch_startends[filenames[i]])
                raise Exception("Intervals were not saved for every sweep.")
            
            # apply transform to `mt_startend` and `epoch_startends` if file requires it
            self.mt_startend = file_specific_transform(
                filenames[i], times=self.mt_startend)
            self.epoch_startends = file_specific_transform(
                filenames[i], times=self.epoch_startends)
            
            # plot segmented epochs on .csv file
            # this goes after `show_abf_segments` because it requires `self.epoch_startends` to be filled out
            if show_csv_segments:

                # number of traces
                # N = int(df_data.shape[1]/2)

                f, ax = plt.subplots(2, 1, figsize=(14, 5))
                for j, g in enumerate(self.epoch_startends[filenames[i]].keys()):
                    ax[0].plot(df_data.iloc[::5, j], lw=2, alpha=0.4)
                    ax[1].plot(df_data.iloc[::5, N+j], lw=2, alpha=0.4)

                    for k in range(2):
                        # epochs for sweep `g` of file `i`
                        E = self.epoch_startends[filenames[i]][g]

                        for n, u in enumerate(E):
                            # remove epoch if it (e.g. transformed) exceeds dimensions
                            if u >= df_data.shape[0]:
                                self.epoch_startends[filenames[i]][g] = E[:n]
                                break
                            else:
                                ax[k].axvline(u/khz, c='white', ls='--')

                ax[0].set_title("show_csv_segments: %s" % filenames[i])
                ax[0].set_ylabel("Current")
                ax[1].set_ylabel("Voltage")
                ax[1].set_xlabel("Time (ms)")
                plt.tight_layout()

                print(
                    "Showing CSV segments. White dashes show epochs in voltage protocol.")
                plt.show()
                plt.close()
                # exit()

        # assign self variables for class
        self.dates_to_save = dates_to_save
        self.main_dir = main_dir
        self.save_path = main_dir + "output/Processing/Pooled_Analyses/"
        self.ephys_info = ephys_info

        # experimental recording parameters
        self.exp_params = exp_params

        # recordings from same cell
        # {parent filename : [[subsequent files], [activation files]]}
        self.paired_files = paired_files
        self.filenames = filenames
        self.data_files = data_files
        self.abf_files = abf_files

        # create prefix for output without skipped filenames or exclusion criteria
        self.output_prefix = read_ephys_info.CreatePrefix(
            exclusion=False, skip=False)

        # for downstream options
        # self.show_protocols = show_protocols
        self.show_abf_segments = show_abf_segments
        self.show_csv_segments = show_csv_segments
        self.do_pubplots = do_pubplots
        self.VoltageClampQuality = VoltageClampQuality
        self.do_pseudoleak_subtraction = do_pseudoleak_subtraction
        self.show_leak_subtraction = show_leak_subtraction
        self.manual_cap_offset = manual_cap_offset
        self.show_Cm_estimation = show_Cm_estimation
        self.show_MT_estimation = show_MT_estimation
        self.do_ramp_stuff = do_ramp_stuff
        self.do_exp_kinetics = do_exp_kinetics
        self.do_activation_curves = do_activation_curves
        self.do_inst_IV = do_inst_IV
        self.normalize = normalize
        self.remove_after_normalize = remove_after_normalize
        self.save_AggregatedPDF = save_AggregatedPDF

        # for computations
        self.exp_fit_params = []    # parameters for exp1-3 fitting
        self.exp_fit_delay = {}     # delay in exp1
        self.ac_fit_params = []     # parameters for activation curve
        self.ac_norm_data = []      # normalized activation curve
        self.tail_post_pmins = []   # Pmin from post-pulse currents
        self.ramp_stats = []        # Ramp hysteresis statistics
        self.IV_params = []         # Summary statistics from I-V curves: Erev, P_K, P_Na
        self.IV_currents = []       # instantaneous current and current dnesity vs. voltage
