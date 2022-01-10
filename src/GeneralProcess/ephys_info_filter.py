# Copyright (c) 2021 Delbert Yip
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

""" Filter `ephys_info.xlsx`
This module provides methods to apply inclusion/exclusion criteria to `ephys_info.xlsx`. The module is tailored to ABF file formats and how I document my own experiments, but I have left plenty of room for extension.

For example, `ExpParamCleaner`, out of the box, accepts different options for:
    1. how to recognize file names 
    2. how to recognize when multiple values are present in a single cell
    3. column-specific ways to handle cells with multiple values 
    4. names of columns to process using a 'whole-cell' paradigm or a 'seal parameter' paradigm

Of course, a subclass can be written that overrides its methods. 

"""

import pandas as pd 
import numpy as np 
import math, logging 

from typing import List, Dict, Tuple, Union, Callable, Any
from GeneralProcess._ import clean_up_params

from GeneralProcess.base import TNumber

import os 

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

PCLAMP_DATE_FMT = re.compile(r"(\d{2})([ond0-9])(\d{2})(\d{3})")

CriteriaType = Union[str, List[str], List[List[str]]]

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

        # see if multiple values are present 
        values = self.multiple_wc_format.findall(item)
        
        if values:
            values = [float(x) for x in values]
        
            if agg_func is None:
                return values 
            
            return agg_func[values]
        
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
        
def apply_inclusion_exclusion_criteria(
    df: pd.DataFrame, col: str, criteria: List[List[str], List[str]]
) -> pd.Series:
    """Filter out files based on `criteria`, a nested list of row values to include or exclude, respectively

    :param df: dataframe to filter
    :type df: pd.DataFrame
    :param col: column to filter
    :type col: str
    :param criteria: nested list containing row values to include or exclude. May be `None` or `['all']` to indicate that all values are included.
    :type criteria: List[List[str]]
    :return: filtered column of `df`
    :rtype: pd.Series
    """    
    
    if criteria is None:
        return df.loc[:, col]
    
    # copy dataframe to be filtered
    out = df.copy()
    
    # join criteria
    masks = ["|".join(c) for c in criteria]

    # inclusion
    if masks[0] != "all":
        out = out.loc[out[col].str.contains(masks[0], na=False)]

    # exclusion
    out = out.loc[~out[col].str.contains(masks[1], na=False)]

    return out

class EphysInfoFiltering:
    def __init__(
        self, criteria: Dict[str, Any], EphysInfoPath: str,
        column_types={'numeric':[], 'string':[]},
        read_kwargs={'header':0, 'index_col':None},
    ) -> None:
        
        if not os.path.isfile(EphysInfoPath):
            raise FileNotFoundError(f"{EphysInfoPath} is not a valid file.")
        
        if 'header' not in column_types or\
            'string' not in column_types:
            raise KeyError(f"`column_types must have 'numeric' and 'string' keys. Current keys: {column_types.keys()}")
        
        ephys_info = self.read_data(EphysInfoPath, read_kwargs)
        self.ephys_info = self.check_data(ephys_info, column_types)
        
        self.criteria = criteria 
        
    def read_data(self, path: str, **kwargs) -> pd.DataFrame:
        """Read ephys info data and validate column types"""
        # read file 
        ext = os.path.splitext(path)
        if ext == '.csv':
            df = pd.read_csv(path, **kwargs)
        else:
            df = pd.read_excel(path, **kwargs)
        
        return df 
        
    def check_data(
        self, df: pd.DataFrame, 
        col_types: Dict[str, List[str]], infer=True
    ) -> pd.DataFrame:
        
        # validate column types dict 
        col_types = self.validate_column_types(col_types, df)
        
        df = self.apply_column_types(
            df, col_types, infer=infer
        )
        
        return df 
    
    @staticmethod
    def validate_column_types(
        df: pd.DataFrame,
        col_types: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Discard column names that are not in the data"""
        
        data_cols = df.columns
        specified = [] 
        
        for typ, cols in col_types.items():
            
            if not cols:
                if typ == 'numeric':
                    cols = [
                        'DNA (ng)', 'Transfection \nreagent (ul)',
                        'OptiMEM (ul)', 'Time post-seed (h)',
                        'Time post-transfection (h)', 'R_pp (M)', 
                        'R_sl (G)', 'C_m (pF)', 'R_m (M)', 
                        'R_sr (M)', 'Tau, +20 (pA)', 'Leak (pA)'
                    ]
                else: continue
            
            in_data = [c in data_cols for c in cols]
            
            if all(in_data): continue 
            
            col_types[typ] = [c for c in cols if c in data_cols]
            specified.extend(col_types[typ])
        
        col_types['to_infer'] = [c for c in data_cols if c not in specified]
        
        return col_types
    
    def apply_column_types(
        self, df: pd.DataFrame, 
        col_types: Dict[str, List[str]], infer: bool
    ) -> pd.DataFrame:
                
        str_cols = df.loc[:, col_types['string']]
        df.loc[:, col_types['string']] = str_cols.astype(str)
        
        num_cols = df.loc[:, col_types['numeric']]
        df.loc[:, col_types['numeric']] = num_cols.convert_dtypes(
            convert_integer=True, convert_floating=True
        )
        
        if infer and col_types['to_infer']:
            infer_cols = df.loc[:, col_types['to_infer']]
            df.loc[:, col_types['to_infer']] = infer_cols.convert_dtypes(
                infer_objects=True)
        
        self.col_types = col_types
        
        return df 
        
    def filterDates(
        self, df: pd.DataFrame, dates: CriteriaType=None
    ) -> pd.DataFrame:
        
        for col in ['Files', 'Dates', 'Protocol']:
            if col not in self.criteria: 
                raise KeyError(f"{col} not in self.criteria, which has columns:\n{self.criteria.columns}")
        
        if dates is None:
            dates = self.criteria["Dates"]

        if not dates:
            raise ValueError(f"No dates were selected. To process all dates, use ['dates' : ['all']]")
        
        if dates[0] == 'all' and len(dates) == 1:
            logging.info("Selecting all dates.")
            
            df = df.loc[
                    (df['Files'] != 'nan') &\
                    (df['Protocol'] != 'nan')
                ].reset_index(drop=True)

            return df 
        
        # row indices of files to keep 
        keep = [] 
        
        if all(isinstance(d, list) for d in dates):
            if len(dates) == 2:
                return apply_inclusion_exclusion_criteria(
                    df, 'Files', dates
                )
            else:
                raise ValueError(
                    "Criteria < Dates > must be list of Strings, String, or list of [inclusion, exclusion] criteria."
                )
        
        for date in dates:
            if not isinstance(date, str):
                try: 
                    date = str(date)
                except (TypeError, RuntimeError, ValueError):
                    raise TypeError(
                        f"Failed to convert file < {date} > to string."
                    )
            
            # a `full` date is 8 characters long, regardless of month = YY M DD xxx, where M = {1-9, o, n, d}
            # any entry in `dates` that is < 8 characters is interpreted as an attempt to index a collection of files, if present
            if len(date) < 8:
                matches = [i for (i, row) in enumerate(df['Files'])\
                    if date in row]
            else:
                matches = [i for (i, row) in enumerate(df['Files'])\
                    if date == row]
                
            if matches:
                keep.extend(matches)

        if keep: 
            keep = list(set(keep))
            keep.sort()
        else:
            raise ValueError(f"No files were selected.")

        # apply selection
        df = df.iloc[keep, :].loc[df['Protocol'] != 'nan'\
            ].reset_index(drop=True)
        
        return df 
    
    def filterColumn(
        self, df: pd.DataFrame, col: str, sel: CriteriaType
    ) -> None:
        
        if sel == 'all':
            return df 
        
        if col not in df.columns:
            if col == 'Dates':
                raise ValueError(
                    "`FilterProtocol` was called to filter `Dates`, but `FilterDates` should be used instead."
                )
            elif 'Protocol' in col:
                col = 'Protocol'
                filterCol = df.loc[:, col]
            else:
                raise KeyError(f"{col} not in data columns:\n{df.columns}")
        
        if sel is None:
            if col in self.criteria: 
                sel = self.criteria[col]
            else:
                if col == 'Protocol':
                    sel = self.criteria['Protocol_Name']
                else:
                    raise ValueError(f"No criteria values found for filtering on < {col} >.")
        
        # kwargs for `pd.Series.str.contains`
        _str_kw = dict(case=False, na=False)
        
        if isinstance(sel, str):
            return df.loc[filterCol.str.contains(
                sel, **_str_kw)]
                
        if isinstance(sel, list):
            if sel[0] == 'all':
                return df 
            
            # inclusion, exclusion = [List[str], List[str]]
            if len(sel) == 2 and\
                all(isinstance(s, list) for s in sel):
                return apply_inclusion_exclusion_criteria(
                    df, col, sel
                )
            
            # list of filenames = List[str] 
            if all(isinstance(s, str) for s in sel):
                if len(sel) == 1:
                    return df.loc[filterCol.str.contains(
                        sel[0], **_str_kw)]
                
                mask = '|'.join(sel)
                return df.loc[filterCol.str.contains(
                    mask, **_str_kw)]
            
        raise TypeError(f"Criteria values for criteria {col} must be type String, List[String], or a List[List[str], List[str]] of file dates to include and exclude:\n{sel}")

    def apply_criteria(self) -> Tuple[List[str], pd.DataFrame]:

        if not self.criteria: 
            raise ValueError("No filtering criteria found.")

        # read ephys_info dataframe
        ei = self.ephys_info.copy()

        for criteria, vals in self.criteria.items():
            logging.info(f"\n    Filtering... {criteria} = {vals}")

            if criteria.lower() == 'dates':
                ei = self.filterDates(ei, vals)
            
            elif 'skip' in criteria.lower():
                files_to_skip = self.criteria['Files_To_Skip']
                logging.info(f"Skipping...\n{files_to_skip}")
                
                if files_to_skip:
                    ei = ei.loc[~ei["Files"].isin(files_to_skip)]
            
            else:
                logging.info(ei)
                ei = self.FilterColumn(ei, criteria, sel=vals)
            
        if ei.shape[0] < 1:
            raise Exception("No files found.")

        ei.reset_index(drop=True, inplace=True)
        logging.info(f"\n {ei.shape[0]} files found for these\
            criteria...\n{ei.loc[:, ['Files', 'Protocol']]}")
        
        filenames = ei['Files'].values.tolist()

        return filenames, ei
    
    @staticmethod
    def format_paired_files(
        df: pd.DataFrame, 
        paired_files: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        
        """Format dictionary of paired files
        Output has format
        `{ parent filename : [subsequent filenames, [activation filenames]] }`
        """
    
        # find activation protocol in protocols from the same cell
        for parent, children in paired_files.items():
            
            # use a copy to avoid modifying the actual dictionary values
            # insert name of parent file at the beginning 
            _children = children.copy() 
            _children.insert(0, parent)
            
            # find all protocols for children
            protocols = df.loc[df['Files'].isin(children),
                               ['Files', 'Protocol']]
            
            # filenames of recordings with activation protocols
            _act = protocols.loc[protocols['Protocol'].\
                str.contains('_act'), 'Files'].values.tolist()
            
            if _act: 
                paired_files[parent].append(_act)

        logging.info(f"\n Files recorded from the same cell...\
            \n{paired_files}")
        
        return paired_files 
    
    def ExpParams(
        self, df: pd.DataFrame,
        exp_params: List[str] = ['R_pp (M)', 'R_sl (G)', 
                                'C_m (pF)','R_m (M)', 'R_sr (M)']
    ):
        """Isolate experimental parameters"""

        if 'Files' not in self.ephys_info.columns:
            raise KeyError(f"'Files' not in data columns:\n\
                {self.ephys_info.columns}")
        
        unfiltered = self.ephys_info.set_index('Files', drop=True)
        exp_params = df.loc[:, exp_params].set_index(
            'Files', drop=True)

        # make sure experimental params are numbers
        exp_params, paired_files = clean_up_params(
            exp_params, unfiltered, single=True
        )
        
        logging.info(
            f"""
            \n Corresponding experimental parameters...
            \n {exp_params}
            \n {pd.concat(
                [exp_params.mean(), exp_params.sem(), exp_params.count()],
                axis=0, keys=["Mean", "SEM", "Count"]
            )}
            """
        )
        
        if not paired_files:
            logging.info("No paired files were found. The first return variable is an empty list.")
            return [], exp_params 
                
        paired_files = self.format_paired_files(paired_files)

        return paired_files, exp_params
    
    def CreatePrefix(self, skip=True, exclusion=True):
        """
        Create file prefix from dictionary of filter criteria

        Returns `prefix`, where intra-criteria entries are 
        separated by `-` and different criteria types 
        (e.g. Dates, Protocol) are separated by `__`
        
        `exclusion` = include exclusion criteria
        `skip` = include skipped filenames
        """

        # Currently just assume every dict value is a List of Strings, but later add compatibility with nested List for exclusion criteria
        
        if all(isinstance(v, str) for v in self.criteria.values()):
            return '__'.join(
                ["-".join(v) for k, v in self.criteria.items()
                if k != "Files_To_Skip"])
        
        prefix = []
        
        for criteria, vals in self.criteria.items():

            if not skip and criteria == "Files_To_Skip":
                continue

            if isinstance(vals, str):
                prefix.append(vals)
                continue 
            
            if not isinstance(vals, list):
                raise TypeError(f"Filter criteria must be type\
                    str, or list, not {type(vals)}")
            
            if all(isinstance(x, str) for x in vals):
                prefix.append("-".join(vals))
            
            elif all(isinstance(x, list) for x in vals)\
                and len(vals) == 2:
                
                prefix_i = ''
                
                if len(vals[0]) > 1 and vals[0][0] != 'all':
                    prefix_i = "-".join(vals[0][0])

                if exclusion and len(vals[1]) > 1:
                    prefix_i = "-!" + '-'.join(vals[1])

                if len(prefix_i) > 1:
                    prefix.append(prefix_i)
                
        prefix = "__".join(prefix)
        return prefix
    
class FindPairedFiles:
    """Convenience method to find paired files, given criteria in `fname`"""
    
    @staticmethod
    def parseFileName(fname: str) -> tuple:
        
        # parse filename into Dates, Protocol, and "act_norm.csv"
        parts = fname.split("__")
        
        if parts[2] != 'act_norm.csv':
            raise ValueError(
                f"Expected suffix to be 'act_norm.csv', got < {parts[2]} >"
            )
        
        for i, part in enumerate(parts):
            if '-' in part:
                part = part.split('-')
            else:
                parts[i] = [part[i]]

        return parts[0], parts[1] 

    def find_dates_and_protocols(self, fname: str):
        """
        Find paired files by first parsing `fname` into Dates and Protocols, then using these to filter `ephys_info.xlsx`
        """
        # parse `fname`
        if isinstance(fname, str):
            dates, protcls = self.parseFileName(fname)
            
        elif isinstance(fname, list):
            dates = []
            protcls = []

            # parse date and protocol for each filename in `fname` 
            for f in fname:
                d, p = self.ParseFName(f)
                dates.append(d)
                protcls.append(p)
            
        else:
            raise TypeError(
            f"Expected a String or List of Strings, got {type(fname)}")

        return dates, protcls 
    
    def get_pairs(
        self, fname: str, EphysInfoPath: str, **filter_kwargs
    ) -> Dict[str, List[str]]:
        """
        `fname` is expected to have the following structure:
            1. Dates separated by "-", followed by "__", then
            2. Protocol name(s), separated by "-", followed by "__", then
            3. "act_norm.csv"
        """
        
        dates, protcls = self.find_dates_and_protocols(fname)
        
        criteria = {"Dates": dates, "Protocol_Name": protcls}
        
        filtor = EphysInfoFiltering(
            criteria, EphysInfoPath, **filter_kwargs
        )
        
        # apply filtering over Dates and Protocols
        _, filtered = filtor.filter()
        
        # find paired files (included in extraction of 
        # experimental parameters)
        paired_files, _ = filtor.ExpParams(filtered)

        return paired_files
