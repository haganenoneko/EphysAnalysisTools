# Copyright (c) 2021 Delbert Yip
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

""" File-specific transforms 
Use this script to implement file-specific transformations to the data. 

A *justified* use case is when a recording(s) whose protocol did not 
follow the specified command for one or more sweeps. In this case, add 
a subclass of `AbstractSpecificTransform` that either corrects or removes
the 'offending' sweeps.

A justified use case should not modify the current record. An *unjustified* use 
case manipulates the data in such a way that the current record (timepoints, 
current amplitudes, sweep order, etc.) are substantially modified. 

If desired, implement subclasses of `ConstrainedSpecificTransform`, which 
contains simple constraints on how the data may be transformed. These constraints are:
1. The current record is untouched, or removed with the corresponding voltage command
2. The order of sweeps is preserved
3. The maximum and minimum voltage in each command is preserved. 
"""

import numpy as np 
import pandas as pd

from typing import List, Union, Tuple

from abc import ABC, abstractmethod

from GeneralProcess.base import NDArrayFloat

# -------------------------------- Interfaces -------------------------------- #

class AbstractSpecificTransform(ABC):
    
    @abstractmethod
    def __init__(
        self, df: pd.DataFrame, fname: str, 
        times: Union[List[int], NDArrayFloat]
    ) -> None:
        pass 
    
    @staticmethod
    @abstractmethod
    def static_transform(self): 
        pass 

class TransformError(ValueError):
    def __init__(self, validation: str, data: pd.DataFrame) -> None:
        super().__init__(
            f"This transform does not satisfy the constraint:\
                < {validation} >\n{data}"
        )

class ConstrainedSpecificTransform(AbstractSpecificTransform):
    
    def __init__(self, df: pd.DataFrame, fname: str, 
                times: Union[List[int], NDArrayFloat],
                voltage_range: Tuple[int, int]=(-200, 100)) -> None:
        
        self._df = df 
        self._fname = fname 
        self._times = times 
        
        self._voltage_range = voltage_range
        
    def validateVoltageRange(self, df: pd.DataFrame) -> bool:
        """Ensure that all voltages are in `self._voltage_range`"""
        
        vlims = self._voltage_range
        v_min = df.min(axis=0) 
        v_max = df.max(axis=0)
        
        try:
            assert (v_min.copy() > vlims[0]).all()
        except AssertionError:
            raise TransformError(
                f"There are voltages below minimum {vlims[0]}",
                v_min
            )
        
        try:
            assert (v_max.copy() < vlims[1]).all()
        except AssertionError:
            raise TransformError(
                f"There are voltages above the maximum {vlims[1]}",
                v_max
            )
        
        return True 
        
    def validateColumns(self, transformed: pd.DataFrame) -> bool:
        """Ensure that columns match the original or are even in number"""
        
        n_col = self._df.shape[1]
        assert n_col % 2 == 0
        
        if transformed.shape[1] == n_col: 
            return True 
        
        if transformed.shape[1] % 2 != 0: 
            data = f"{self._df.head}\n{transformed.head}"
            raise TransformError("An even number of columns.", data)
        
        n_sweeps = int(n_col/2)
        self.validateVoltageRange(transformed.iloc[:, n_sweeps:])
        
        return True 
        
    def validateRows(self, transformed: pd.DataFrame) -> bool:
        """Ensure that rows match the original or are equal in number"""
        
        if transformed.shape[0] == self._df.shape[0]:
            return True 
            
        def get_nrows(df: pd.DataFrame, col: int) -> int:
            return df.iloc[:, col].dropna().shape[0]
            
        n_0 = get_nrows(transformed, 0)
        for i in range(1, transformed.shape[1]):
            n_i = get_nrows(transformed, i)
            if n_i == n_0: continue 
            
            data = f"First column: {n_0}\t{i}-th column: {n_i}"
            
            raise TransformError(
                "Equal number of non-NaN elements in each column.",
                data
            )
        
        return True 
    
    
        
def file_specific_transform(
    fname: str, times: List[int]=None, df: pd.DataFrame=None
):
    """
    Apply file-specific transformation to times, as necessary.  
    `fname` = filename or 'FA_env_+20'
    `df` = dataframe 
    `times` = bounding intervals for leak ramp  
    `khz` = sampling rate 
    """
    FA_env = [
        '2091004', '20911007', '20o08003', '20o16001',  # +20
        '20o16002', '21521006'                          # -55
    ]
    to_transform = ["20903005"]
    to_transform.extend(FA_env)

    if fname in to_transform:
        if times is not None:

            # ramp start (t0) and end (t1) times
            if isinstance(times, list):

                if fname == '20903005':
                    # after the first trace, multiply all epochs by 2
                    t0, t1 = times
                    return [t0, t1, 2*t0, 2*t1, 2*t0, 2*t1]

            elif isinstance(times, dict):

                if fname in times.keys():
                    if fname == '20903005':
                        d = times[fname]

                        for i in range(1, len(d)):
                            d[i] = [2*x for x in d[i]]

                        times[fname] = d

                    elif fname in FA_env:
                        if isinstance(times[fname][0], list):
                            if times[fname][0][-1] > 47000:
                                val = times[fname][0]
                                times[fname][0][-3:] = [x -
                                                        7233 for x in val[-3:]]

                        elif isinstance(times[fname][0], int):
                            if times[fname][0] > 47000:
                                times[fname][0] -= 7233

            elif isinstance(times, tuple):

                # tuple = abf_sum, csv_sum -> force abf_sum = csv_sum
                if fname == '20903005':
                    return times[0]
                else:
                    return times[1]

            return times

        if df is not None:
            if fname in FA_env:
                N = int(df.shape[1]/2)

                dt = df.shape[0] - 47937
                new = df.iloc[40704:, N].copy()

                # plt.plot(df.iloc[:,N], lw=2, c='y')
                df.iloc[40704:(40704+dt), N] = new.iloc[7233:].values
                df.iloc[52766:, N] = df.iloc[-1, N]

                return df

    else:
        if times is not None:
            if isinstance(times, tuple):
                # abf_sum, csv_sum -> keep csv_sum
                return times[1]
            else:
                # don't change anything
                return times

        elif df is not None:
            return df
