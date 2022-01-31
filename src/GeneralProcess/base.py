# Copyright (c) 2021 Delbert Yip
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os, glob, logging, inspect 
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path

from typing import Dict, Any, List, Union, Tuple
from typing import Callable, TypeVar

import numpy.typing as npt

# ------------------------------- Custom types ------------------------------- #

TNumber = TypeVar('TNumber', int, float)

NDArrayBool = npt.NDArray[np.bool_]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float64]

FloatOrArray = Union[float, NDArrayFloat]

KwDict = Dict[str, Any]

# https://stackoverflow.com/questions/533048/how-to-log-source-file-name-and-line-number-in-python
LOG_FORMAT = logging.Formatter(
    r"%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------------------------------- #
#                                Logging methods                               #
# ---------------------------------------------------------------------------- #


class CallStackFormatter(logging.Formatter):
    """
    https://stackoverflow.com/questions/54747730/adding-stack-info-to-logging-format-in-python
    """

    def formatStack(self, _=None) -> str:
        stack = inspect.stack()[::-1]
        stack_names = (inspect.getmodulename(stack[0].filename),
                       *(frame.function for frame in stack[1:-9]))
        return '::'.join(stack_names)

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        record.stack_info = self.formatStack()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        s = self.formatMessage(record)
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        return s


def createLogger(
    log_path: str, overwrite=False,
    formatter=CallStackFormatter(), log_level="DEBUG"
) -> logging.Logger:
    
    if log_path is None or not os.path.isdir(log_path):
        logging.info(
            f"{log_path} is an invalid directory. Logger instantiated without an output file."
        )
        logger = logging.basicConfig(
            encoding='utf-8', level=log_level, format=formatter
        )
        return None

    if overwrite:
        file = log_path + "processing.log"
    else:
        n = len(glob.glob(log_path + "processing*.log"))
        file = log_path + f"processing_{n}.log"

    logging.basicConfig(format=formatter)
    plog = logging.getLogger("Processing")
    plog.setLevel(log_level)

    hndl = logging.FileHandler(file, encoding='utf-8')
    hndl.setLevel(log_level)
    hndl.setFormatter(formatter)
    plog.addHandler(hndl)

    plog.info(f"Log file created at time: {datetime.now()}")

    return plog

# -------------------------------- I/O helpers ------------------------------- #


def get_valid_paths(
    paths: Union[List[str], List[Path]],
    relative_dir: Union[str, Path] = None
) -> List[Path]:

    invalid_msg = f"{0} is an invalid directory or file."
    valid_paths = []

    # set relative directory to cwd if not given
    if relative_dir is None:
        relative_dir = Path.cwd()

    for p in paths:

        if isinstance(p, str):
            p = Path(p)

        if p.is_dir() or p.is_file():
            valid_paths.append(p)
            continue

        p = relative_dir.joinpath(p)
        if p.is_dir() or p.is_file():
            valid_paths.append(p)
        else:
            logging.info(invalid_msg.format(p))

    return valid_paths


# ----------------------------- Simple functions ----------------------------- #

def FallbackDecorator(
    func, fallback, 
    errors: Tuple[Exception] = (ValueError, RuntimeError, NotImplementedError)
):
    def wrapped(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except errors:
            fallback(*args, **kwargs)
    return wrapped


def multi_sort(zipped: List[list]):
    """
    Sort multiple lists in `zipped` according to first element 
    Returns sorted lists 
    """
    return map(list, zip(*sorted(zipped)))


def CleanlyDropNaNs(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda x: pd.Series(x.dropna().values))


def pprint_dict(
    d: Dict[str, Any], delim: str = "\n", func: Callable[[Any], Any] = None
) -> str:
    """Pretty printing for dictionaries"""
    if func is None:
        lst = [f"{k} = < {v} >" for k, v in d.items()]
    else:
        lst = [f"{k} = < {func(v)} >" for k, v in d.items()]

    return delim.join(lst)


def scalarTimesList(scalar: TNumber, lst: List[TNumber]) -> List[TNumber]:
    return [scalar*x for x in lst]


def extendListAsArray(lst: List[TNumber], dims: Tuple[int, ...]) -> Union[NDArrayFloat, NDArrayInt]:

    elemType = type(lst[0])
    ext = np.full((dims[0], len(lst)), lst, dtype=elemType)

    if ext.shape == dims:
        return ext

    extension = (0, dims[1] - ext.shape[1])
    ext = np.pad(ext, ((0, 0), extension),
                 mode='constant', constant_values=ext[-1])

    return ext


def exp1(
    t: FloatOrArray, dI: FloatOrArray, tau: FloatOrArray, I_ss: FloatOrArray
):
    """Single-exponential function

    :param t: time
    :type t: FloatOrArray
    :param dI: current
    :type dI: FloatOrArray
    :param tau: time constant
    :type tau: float
    :param I_ss: steady-state current
    :type I_ss: FloatOrArray
    :return: simulated exponential
    :rtype: NDArrayFloat
    """
    return dI*np.exp(-t/tau) + I_ss
