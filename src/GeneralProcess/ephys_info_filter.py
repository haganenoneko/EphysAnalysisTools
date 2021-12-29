# Copyright (c) 2021 Delbert Yip
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""

"""

import pandas as pd 
import numpy as np 
import math, logging 

from typing import List, Dict, Union, Callable

from GeneralProcess.base import TNumber

import regex as re 

# ----------------------------- Useful constants ----------------------------- #

SEAL_PARAMS = ['R_pp (M)', 'R_sl (G)']               
WHOLE_CELL_PARAMS = ['C_m (pF)', 'R_m (M)', 'R_sr (M)']

WC_MULTIPLE_FORMAT: re.Pattern = re.compile("[\d\.]*[^\\\/]")

WC_AGG_FUNCS = {'C_m (pF)' : None, 'R_m (M)' : None, 'R_sr (M)' : None}

ABF_FILE_NAME_FORMAT: re.Pattern = re.compile(
    # 2 digits for year, 1 digit or letter in [nod] for month,
    # 2 digits for date, 3 digits for recording
    r"(\d{2})([1-9|nod])([0-3][0-9])(\d{3})"
)

# ---------------------------------------------------------------------------- #

class ExpParamCleaner:
    
    def __init__(
        self, df: pd.DataFrame, unfiltered: pd.DataFrame,
        file_name_format: re.Pattern = ABF_FILE_NAME_FORMAT,
        multiple_wc_format: re.Pattern = WC_MULTIPLE_FORMAT,
        pp: List[str] = SEAL_PARAMS, wc: List[str] = WHOLE_CELL_PARAMS,
        wc_agg_funcs: Dict[str, Callable[[List[float]], float]] = WC_AGG_FUNCS
    ) -> None:
        """Format experimental parameters as numbers, or replace with parent values where specified

        :param df: dataframe with selection criteria applied
        :type df: pd.DataFrame
        :param unfiltered: original dataframe before selection filtering
        :type unfiltered: pd.DataFrame
        :param file_name_format: `regex` compiled pattern that matches filenames, defaults to ABF_FILE_NAME_FORMAT
        :type file_name_format: re.Pattern, optional
        :param multiple_wc_format: `regex` compiled pattern that matches how `wc` parameters are written when multiple values are present, defaults to WC_MULTIPLE_FORMAT
        :type multiple_wc_format: re.Pattern, optional
        :param pp: names of parameters which may be inherited from a previous recording in the same preparation such that the cell contains the parent's filename instead of a number, defaults to SEAL_PARAMS
        :type pp: List[str], optional
        :param wc: names of parameters that are not inherited and which should be numeric or unspecified ('x' or 'nan'), defaults to WHOLE_CELL_PARAMS
        :type wc: List[str], optional
        :param wc_agg_funcs: functions used to aggregate `wc` parameters when multiple are specified (e.g. before, during, and after a recording). A function can be defined separately for each parameter, or declared None, defaults to WC_AGG_FUNCS
        :type wc_agg_funcs: Dict[str, Callable[[List[float]], float]], optional
        """
        self.df = df 
        self.unfiltered = unfiltered
        self.file_names = df.loc[:, 'Files'].astype(str)
        
        self.pp = pp 
        self.wc = wc 
        self.wc_agg_funcs = wc_agg_funcs 
        
        self.file_name_format = file_name_format
        self.multiple_wc_format = multiple_wc_format
        
        self.paired: Dict[str, List[Union[str, TNumber]]] = {} 
        
        self.validate_wc_agg_funcs()

    def validate_wc_agg_funcs(self):
        
        for wc in self.wc:
            
            if wc in self.wc_agg_funcs:
                continue 
            
            self.wc_agg_funcs[wc] = None 
    
    def validateDatePair(
        self, parent: Union[str, TNumber], child: Union[str, TNumber]
    ) -> bool:
        """Check that `parent` was recorded before `child`"""
        
        res = [self.file_name_format.search(f).groups() for f in [parent, child]]
        
        try:
            for i in range(3):
                assert res[0][i] == res[1][i]
            
            assert int(res[0][3]) < int(res[1][3])
            
        except AssertionError:
            raise AssertionError(
                f"Parent < {parent} > was not recorded before child < {child} >"
            )
        
        return True 

    def updateFilePairings(
        self, parent: Union[str, TNumber], child: Union[str, TNumber]
    ) -> None:
        
        self.validateDatePair(parent, child)
        
        if parent in self.paired:
            self.paired[parent].append( child )
        else:
            self.paired[parent] = [child]
        
    def parse_wc_element(
        self, item: str, agg_func: Callable[[List[float]], float] = None
    ) -> Union[float, List[float]]:
        """Parse whole-cell elements"""
        
        if item.lower() in ['x', 'nan']:
            return np.nan 

        # see if there is a slash in the current cell value
        slashes = self.multiple_wc_format.findall(item)
        if slashes:
            slashes = [float(x) for x in slashes]
        
            if agg_func is None:
                return slashes 
            
            return agg_func[slashes]
        
        raise ValueError(
            f"{item} could not be parsed. If invalid, assign 'x' or 'nan'."
        )
    
    def parse_pp_element(
        self, item: str, row: Union[str, TNumber], col: str
    ):
        """Parse pipette/seal parameters.
        These are directly inherited from 'parent' files in most cases,
        so values may be names of other files. Order is not checked, so be careful in data entry."""
        
        if not self.file_name_format.search(item):
            return 
            
        if item in self.file_names:
            parent_info = self.unfiltered.loc[item, :]
        
        else:
            raise ValueError(
                f"{item} is a valid file name, but does not correspond to any\
                    file in the data:\n{self.file_names}"
            )

        if item == parent_info.at[col]:
            raise ValueError(
                f"The current cell refers to its own filename: {item}\n\
                If no value exists, then assign 'x' or 'nan'. Offending row:\n\
                {parent_info}"
            )
        
        # update parent-child relationship 
        self.updateFilePairings(parent=item, child=row)
        
        # set value of current cell to value of parent cell
        return parent_info.at[col]

    def parseElement(
        self, df_ij: Union[str, float], row: Union[str, int], col: str
    ) -> Union[float, str]:

        try:
            return float(df_ij)
            
        except ValueError:
            if col in self.wc: 
                return self.parse_wc_element(
                    df_ij, agg_func=self.wc_agg_funcs[col]
                )
                
            elif col in self.pp:
                return self.parse_pp_element(df_ij, row, col)
        
        raise RuntimeError(f"Could not parse {df_ij}.")

    def parse(self):
        
        df = self.df.copy()
        
        for row in self.df.index:
            for col in self.df.columns:
                df_ij = self.df.at[row, col]
                df.at[row, col] = self.parseElement(df_ij, row, col)
                
        self.df = df 
        
