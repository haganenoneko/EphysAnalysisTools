# Copyright (c) 2021 Delbert Yip
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pandas as pd 
import numpy as np 
from pathlib import Path
from random import shuffle
import os, glob, json, shutil
from typing import List, Set, Tuple

T = List[Path]

class dataMigrator:
    
    def __init__(
        self, abf_path: str=None, csv_path: str=None, xl_path: str=None, json_path: str=None
    ) -> None:
        """Copies ABF and CSV files

        :raises FileNotFoundError: if ABF or CSV path are not valid directories
        """
        
        if csv_path is None:
            csv_path = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/Github_repos/hcn-gating-kinetics/data/current_time_course/"
            
        if abf_path is None:
            abf_path = r"C:/Users/delbe/Downloads/wut/wut/Post_grad/UBC/Research/lab/data_files/delbert/"
            
        if xl_path is None:
            xl_path = r"./data/ephys_data_info.xlsx"        
        
        if json_path is None:
            json_path = r"./data/curatedFileNames.json"
        
        if not os.path.isfile(xl_path):
            raise FileNotFoundError(f"{xl_path} is not a valid file.")
        
        for dir in [abf_path, csv_path]:
            if os.path.isdir: continue 
            raise FileNotFoundError(f"{dir} is not a valid directory.")
        
        self.csv_path = csv_path 
        self.abf_path = abf_path 
        
        # read `ephys_data_info.xlsx` and `curatedFileNames.json`
        self.readEphysInfo(xl_path, json_path)
                
    def readEphysInfo(self, xl_path: str, json_path: str):
        """Open ephys info dataframe and curated file names json"""
        
        df = pd.read_excel(xl_path, index_col=None, header=0)
        self.ephysInfo = df 
        
        fileNames = [] 
        with open(json_path, 'r') as io: 
            fnames = json.load(io)
    
        for construct in fnames.values():
            for dates in construct.values():
                fileNames.extend(dates)
    
        self.fileNames = list(set(fileNames))
        
    @staticmethod
    def _extractFileNames(paths: List[Path]) -> Set[str]:
        return set([p.stem for p in paths])
    
    @staticmethod 
    def _matchFileStems(source: T, filter: List[str]) -> T:
        return [s for s in source if s.stem in filter]
    
    @staticmethod
    def _validatePaths(source: T, queries: List[str]) -> bool:
        """Check that each file name in `queries` has a corresponding file in `source`"""
        
        stems = [file.stem for file in source]
        exists = [query in stems for query in queries]
        
        try:
            assert all(exists)
        except AssertionError:
            missing = np.array(queries)[exists]
            raise AssertionError(
                f"The following data files are missing for items in selection:\n{missing}")
        
        return True 
    
    def _filterSelectedFiles(
        self, abf: T, csv: T, stems: List[str], rand_num: int
    ) -> Tuple[T, T]:
        """
        Filter abf and csv files by `stems`
        
        If `rand_num > 0`, this number of files from `stems` is added 
        to the final selection
        """
        try:
            assert len(abf) == len(csv)
            assert len(abf) == len(stems) 
        except AssertionError:
            raise AssertionError(
                f"ABF and CSV files have \
                unequal length:\nABF = {len(abf)}\
                CSV = {len(csv)}, stems = {len(stems)}"
            )
        
        fileNames = self.fileNames 
        self._validatePaths(abf, stems)
        self._validatePaths(csv, stems)
        
        # add random files to be copied 
        if rand_num > 0:
            shuffle(stems)
            fileNames.extend(stems[:rand_num])
            fileNames = list(set(fileNames))
            
            print(
                f"{rand_num} random files added:\n", 
                stems[:rand_num]
            )
        
        sel_abf = self._matchFileStems(abf, fileNames)
        sel_csv = self._matchFileStems(csv, fileNames)
        
        return sel_abf, sel_csv 
    
    def selectFiles(self, rand_num: int) -> Tuple[T, T]:
        """Select file names to migrate.

        :param rand_num: number of random files to add
        :type rand_num: int
        :return: selected ABF and CSV files
        :rtype: Tuple[T, T]
        """
        
        csv = list(Path(self.csv_path).glob("*.csv"))
        abf = list(Path(self.abf_path).rglob("**/*.abf"))
        
        # intersection 
        files = list(
            self._extractFileNames(csv) & \
                self._extractFileNames(abf)
        )
        
        abf_files = self._matchFileStems(abf, files)
        csv_files = self._matchFileStems(csv, files)
        
        return self._filterSelectedFiles(
            abf_files, csv_files, files, rand_num)
        
    @staticmethod
    def _copyFiles(source: List[Path], dest: str) -> int:
        """Copy files from `source` to `dest`

        :param source: data files to copy
        :type source: List[Path]
        :param dest: destination of data files
        :type dest: str
        :return: number of files copied/moved
        :rtype: int
        """
        
        moved = 0 
        for file in source:
            f_out = dest + file.name
            if os.path.isfile(f_out): continue 
            
            shutil.copy2(file, f_out)
            moved += 1 
            
        print(f"Moved {moved} out of {len(source)} files.")
        return moved 
    
    def moveFiles(self, rand_num: int=0) -> None:
        """Copy selected ABF and CSV files to ./data/"""
        
        abf, csv = self.selectFiles(rand_num)
        
        # set up directories 
        basedir = "./data/recording_{fmt}/"
        for format in ['abf', 'csv']:
            folder = basedir.format(fmt=format)
            if os.path.isdir(folder): continue 
            os.mkdir(folder)
        
        moved = 0 
        moved += self._copyFiles(
            abf, basedir.format(fmt='abf'))
        moved += self._copyFiles(
            csv, basedir.format(fmt='csv'))
        
        print(
            f"Moved {moved} of {len(abf) + len(csv)} files."
        )
        return 
    
d = dataMigrator()
d.moveFiles(10)