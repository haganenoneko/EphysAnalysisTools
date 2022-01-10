# Copyright (c) 2021 Delbert Yip
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import logging
import pandas as pd

from pathlib import Path

from typing import Dict, Any, List

import GeneralProcess.base as base
from GeneralProcess.ephys_info_filter import EphysInfoFiltering

import regex as re

from pydantic import BaseModel, validator

import pyabf

"""Find CSV and ABF files that meet selection criteria
This module contains classes and methods that specialize in
    1. Applying, but not validating/reading, selection criteria
    2. Finding files with given extensions and at given location(s)
    
The key class is `DataLoader`, which holds information such as
    1. Raw data files (CSV, ABF)
    2. Experimental parameters (e.g. series resistance, recording protocol name)
    3. Paths to all files
    
An alternative use case would be to work with non-ABF file formats. To do so,
implement a subclass of `DataLoader` and override the `getDataFiles` method.
"""

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
            raise TypeError(
                f"All files to ignore must be strings.\n{to_ignore}")

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
        self, filenames: List[str], rglob: bool = False
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


class DataLoader:
    """Select and find CSV and ABF files"""

    def __init__(self,
                 main_dir: str, csv_path: str, abf_path: str, ephys_info_path: str,
                 filter_criteria: Dict[str, Any],
                 log_path: str = None, out_path: str = None
                 ) -> None:
        """Load CSV and ABF files

        :param main_dir: [description]
        :type main_dir: str
        :param csv_path: path to CSV files
        :type csv_path: str
        :param abf_path: path to ABF files
        :type abf_path: str
        :param filter_criteria: criteria used to select files. See `EphysInfoFiltering` for more information
        :type filter_criteria: Dict[str, Any]
        :param log_path: path to save logs, defaults to None
        :type log_path: str, optional
        :param out_path: path for output files, defaults to None
        :type out_path: str, optional
        :raises ValueError: if filter criteria are not provided
        """

        if not filter_criteria:
            raise ValueError(f"No filter criteria provided.")

        self.paths = self.validatePaths(
            dict(main=main_dir, csv=csv_path, abf=abf_path,
                 ephys_info_path=ephys_info_path, log=log_path, out=out_path))

        self.criteria = filter_criteria
        self.filenames: List[str] = None
        self.ephys_info: pd.DataFrame = None
        self.exp_params: pd.DataFrame = None
        self.paired_files: Dict[str, Any] = None

    def validatePaths(self, paths: Dict[str, str]) -> Dict[str, Path]:
        """Check that paths are valid and convert to `Path` objects"""
        for key, path in paths.items():

            if os.path.isdir(path):
                paths[key] = Path(path)
            elif os.path.isdir(paths['main'] + path):
                paths[key] = Path(paths['main'] + path)
            else:
                raise ValueError(f'{path} is not a valid path.')

        return paths

    def getDataFiles(self, filenames: List[str], to_ignore: List[str]) -> List[str]:
        """Get Dataframes and pyABF objects for CSV and ABF files, respectively

        :param filenames: file names
        :type filenames: List[str]
        :param to_ignore: list of file names to ignore
        :type to_ignore: List[str]
        :return: list of missing files
        :rtype: List[str]
        """

        CSVs = FileFinder(self.paths['csv'], to_ignore=to_ignore,
                          fmt='.csv').find(filenames, rglob=False)

        ABFs = ABF_Finder(self.paths['abf'], to_ignore=to_ignore,
                          fmt='.abf').find(filenames, rglob=True)

        missing: List[str] = []

        missing_msg = "{0} in CSVs: {1}\n{0} in ABFs: {2}"

        for f in filenames:

            if f in CSVs and f in ABFs:
                continue

            logging.info(
                missing_msg.format(f, (f in CSVs), (f in ABFs))
            )

            missing.append(f)

        self.CSVs = CSVs
        self.ABFs = ABFs
        return missing

    def run(self, to_ignore: List[str] = None):

        ephysInfo = EphysInfoFiltering(
            self.criteria, ephys_info_path=self.paths['ephys_info_path'])

        # apply filter criteria
        filenames, ephys_info = ephysInfo.filter()

        # extract paired files and experimental parameters
        paired_files, exp_params = ephysInfo.ExpParams()

        missing = self.getDataFiles(to_ignore=to_ignore)

        # remove missing files
        if missing:
            filenames = [f for f in filenames if f not in missing]
            ephysInfo = ephysInfo[ephysInfo['Files'].str.isin(filenames)]
            exp_params = exp_params.loc[:, exp_params.str.isin(filenames)]

        self.filenames = filenames
        self.ephys_info = ephys_info
        self.exp_params = exp_params
        self.paired_files = paired_files

    def __repr__(self) -> str:

        pretty_paths = base.pprint_dict(self.paths)

        pretty_CSVs = base.pprint_dict(self.CSVs, delim="  ",
                                       func=base._get_df_shape)

        pretty_ABFs = base.pprint_dict(self.ABFs, delim="  ",
                                       func=base._get_df_shape)

        return f"""
            Filenames:\n{self.filenames}\n
            Experiment info:\n{self.ephys_info}\n
            Seal parameters:\n{self.exp_params}\n
            Paired files:\n{self.paired_files}\n
            ABF files:\n{pretty_ABFs}\n
            CSV files:\n{pretty_CSVs}\n
            \nSelection criteria: {self.criteria}\n
            {pretty_paths}
        """
