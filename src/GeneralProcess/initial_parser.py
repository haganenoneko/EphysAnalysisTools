# Copyright (c) 2021 Delbert Yip
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os, glob
import logging 
 
import numpy as np 
import pandas as pd

from pathlib import Path 

from typing import Dict, Tuple, Any, List
from typing import Union, Callable

import GeneralProcess.base as base
from GeneralProcess.base import Recording, AbstractAnalyzer, createLogger
from GeneralProcess.base import NDArrayBool, NDArrayFloat, NDArrayInt
from GeneralProcess.file_specific_transform import linTransEpochTimes
from GeneralProcess.ephys_info_filter import EphysInfoFiltering

from abc import ABC, abstractmethod

import regex as re 

from pydantic import BaseModel, ValidationError, validator

import pyabf

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #

# -------------------------------- Find files -------------------------------- #

class FileFinder(BaseModel):
    """Find files at `path`. Optionally ignore files in `to_ignore`

    :param paths: path(s) to check for files 
    :type paths: List[str]
    :param to_ignore: list of filenames to ignore, defaults to None
    :type to_ignore: List[str], optional
    """
    paths: List[str]
    to_ignore: List[str] = None
    fmt: str = '.csv'
    
    def __post_init__(self):
        self.paths: List[Path] = base.get_valid_paths(self.paths)
        self.data_files: Dict[str, pd.DataFrame] = {} 
        
    @validator('fmt')
    def format_matches_regex(cls: object, fmt: str) -> None:
        
        if not isinstance(fmt, str):
            raise TypeError(f"{fmt} must be type str, not {type(fmt)}")
        
        pattern = re.compile(r"^[a-zA-Z0-9\_\-]*\.[a-zA-Z0-9]$")
        
        if not pattern.match(fmt):    
            raise ValueError(
                f"{fmt} does not match pattern: [prefix].[extension]"
            )
        
        return 
    
    @validator('to_ignore')
    def ignore_are_string(cls: object, to_ignore: List[str]) -> None:
        
        if to_ignore is None:
            return 
        
        if not to_ignore:
            return 
        
        if not all([isinstance(f, str) for f in to_ignore]):
            raise TypeError(f"All files to ignore must be strings.\n{to_ignore}")
        
        return 
        
    @staticmethod 
    def _readFile(file: Path, filename: str) -> pd.DataFrame:  
        
        df: pd.DataFrame = pd.read_csv(file, header=None, index_col=0)
        
        # check if first index is not 0
        if df.index[0] != 0:
            
            try: 
                df.index = df.index.astype(float)
                df.index -= df.index[1]
            except ValueError:
                df = df.iloc[1:, :]
                df.index = df.index.astype(float)
        
        df.index.name = filename         
        return df 

    def _readFiles(self, files: List[Path]) -> None:
        """Load csv files"""
                
        ignore_msg = "File ignored: {0}"
        
        for file in files:
            fname = file.stem 
            
            if not file.is_file():
                continue 
            elif fname in self.to_ignore:
                logging.info(ignore_msg.format(fname))
            else:
                df = self._readFile(file, fname)
                self.data_files[fname] = df
                
        return 
            
    def find(
        self, filenames: List[str], rglob: bool=False
    ) -> Dict[str, pd.DataFrame]:
        
        fmt = self.fmt
        
        for path in self.paths:
            
            if rglob: 
                file_paths = [path.rglob(f + fmt) for f in filenames]
            else:
                file_paths = [path.with_stem(f + fmt) for f in filenames]
                
            self._readFiles(file_paths)

        return self.data_files
    
    def __repr__(self) -> str:
        return f"""
            Paths: {self.paths}\n
            Ignored files: {self.to_ignore}\n
            Loaded files: {self.data_files}
        """

class ABF_Finder(FileFinder):
    
    @staticmethod
    def _readFile(file: Path, filename: str) -> pyabf.ABF:
        
        if not file.is_file():
            logging.info(f"File was not found: {file}")
        
        return pyabf.ABF(file)
    
# -------------------------------- Main parser ------------------------------- #
class BaseProcessor:
    """Select and find CSV and ABF files"""
    
    def __init__(self, 
        main_dir: str, csv_path: str, abf_path: str,
        filter_criteria: Dict[str, Any],
        log_path: str=None, out_path: str=None
    ) -> None:
        
        if not filter_criteria:
            raise ValueError(f"No filter criteria provided.")
        
        self.paths = self.validatePaths(
            dict(main=main_dir, csv=csv_path, abf=abf_path,
                log=log_path, out=out_path))
    
        self.criteria = filter_criteria 
        self.filenames: List[str] = None
        self.ephysInfo: pd.DataFrame = None
        self.expParams: pd.DataFrame = None
        self.pairedFiles: Dict[str, Any] = None
    
    def validatePaths(self, paths: Dict[str, str]) -> Dict[str, Path]:
        """Check that paths are valid and convert to `Path` objects"""
        for key, path in paths.items():
            
            if os.path.isdir(path):
                paths[key] = Path(path)
            elif os.path.isdir(paths['main'] + path):
                paths[key] = Path( paths['main'] + path )
            else:
                raise ValueError(f'{path} is not a valid path.')
            
        return paths 
    
    def run(self):
        ephysInfo = EphysInfoFiltering(self.criteria)
        
    def readFilterCriteria(self) -> None:
        ephysInfo = EphysInfoFiltering(self.criteria)
        filenames, ephys_info = ephysInfo.filter()
        paired_files, exp_params = ephysInfo.ExpParams(ephys_info)
        
        self.filenames = filenames
        self.ephysInfo = ephys_info 
        self.expParams = exp_params 
        self.pairedFiles = paired_files
        
    def getDataFiles(self, to_ignore: List[str]=None):
        
        CSVs = FileFinder(self.paths['csv'], to_ignore=to_ignore, 
                          fmt='.csv').find(self.filenames, rglob=False)
        
        ABFs = ABF_Finder(self.paths['abf'], to_ignore=to_ignore, 
                          fmt='.abf').find(self.filenames, rglob=True)

        for key in set().union(CSVs, ABFs):
            if key in CSVs and key in ABFs:
                continue 
            
            raise KeyError(f"{key} in CSVs: {key in CSVs}\n\
                {key} in ABFs: {key in ABFs}")
        
        self.CSVs = CSVs 
        self.ABFs = ABFs
    
    def removeMissingCSVFiles(self, to_remove: List[str]) -> None:
        """Remove file not found from dataframes of recording parameters"""
        
        if not to_remove: 
            return 
        
        ephysInfo = self.ephysInfo 
        expParams = self.expParams
        
        ephysInfo = ephysInfo[~ephysInfo['Files'].str.isin(to_remove)]
        expParams = expParams.loc[:, ~expParams.str.isin(to_remove)]
        
        print(f"{len(to_remove)} files were removed: {to_remove}.\n\
            Files kept: \n", ephysInfo.loc[:, ["Files", "Protocol"]])
        
        self.filenames = [f for f in self.filenames if f not in to_remove]
        self.ephysInfo = ephysInfo
        self.expParams = expParams


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
