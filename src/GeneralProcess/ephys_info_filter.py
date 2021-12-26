# file selection, filtering, etc. of `ephys_info` for processing

import pandas as pd
import numpy as np
import math
import glob

import seaborn as sns
from scipy.stats import describe

ephys_info = r"C:\Users\delbe\Downloads\wut\wut\Post_grad\UBC\Research\lab\Github_repos\hcn-gating-kinetics\data\ephys_data_info.xlsx"

# clean up experimental params


def clean_up_params(df, unfiltered, return_paired_files=True, single=False):
    """
    `df` = dataframe containing filenames as index, and [R_pp...R_sr] as columns \\
    `unfiltered`= analogous to above, but from unfiltered original dataframe, used if `parent` file is not kept in filtering

    `return_paired_files` = whether to return a dictionary specifying files that were recorded from the same cell (at least >1) \\
    `single` = True -> ensures all dataframe elements are single numbers. \\
    `single` = False -> entries with multiple numbers separated by `/` are parsed as a list

    # Cateogry-specific rules for parsing entries when `single = True`:  \\
    # R_pp and R_sl  \\
    Either number or string corresponding to a filename.
    1. If the filename is not in the index, raise Error.
    2. Else, find the corresponding values of said file, and replace the entries accordingly.

    # Cm, Rm, and Rsr   \\
    If three values are given for Cm, Rm, or Rsr - this corresponds to values from:
    1. `membrane test` (MT) before,
    2. whole cell compensation (WCC, for Rsr and Cm) and seal test (for Rm), and
    3. MT after the recording

    If two values are given for Cm, Rm, or Rsr - this typically corresponds to MT before and after.

    For Rsr and Cm, take the WCC value when available, or the mean.

    For Rm,
    - if 2 values are available, take the mean
    - if 3 values are available, take the mean of the first and last
    """
    colnames = df.columns
    pp = ['R_pp (M)', 'R_sl (G)']               # seal parameters
    wc = ['C_m (pF)', 'R_m (M)', 'R_sr (M)']    # whole cell parameters

    # dictionary of paired protocols, i.e. recorded from the same cell
    # in a given {key : value}, the key is name of first recording, while values are names of subsequent
    paired = {}

    for idx in df.index.values:
        # idx = ith filename

        # iterate over columns (recording parameters)
        for j, col in enumerate(colnames):
            # value of (i,j)-th element in dataframe
            df_ij = df.at[idx, col]

            # convert to string; if filename, this is the filename of the `parent` file
            s = str(df_ij)

            # try converting value of current cell to a float;
            # this may succeed if the parent filename is fully numeric
            # if it is actually a parent file's name, we will change it to the parent's value below
            try:
                df.at[idx, col] = float(df_ij)

            # if conversion to a float fails,
            # we may have a filename (seal parameters) or multiple values (whole cell parameters)
            except:
                # if we have a whole cell parameter, split the cell by either \ or /
                # then convert each resulting value into a float
                # get a single value using procedure in docstring above

                # see if current column is a whole cell parameter
                if col in wc:

                    if df.at[idx, col] == "x":
                        print(
                            "Invalid value `x` encountered in whole-cell parameters. Replacing with NaN")
                        df.at[idx, col] = np.nan
                        break

                    # see if there is a slash in the current cell value
                    if "/" in df_ij:
                        x = df.at[idx, col].split("/")
                        x = [float(u) for u in x]
                    elif "\\" in df_ij:
                        x = df.at[idx, col].split("\\")
                        x = [float(u) for u in x]

                    # return a single value
                    if single:
                        if len(x) == 2:
                            df.at[idx, col] = sum(x)/2
                        elif len(x) == 3:
                            if col == wc[1]:
                                df.at[idx, col] = (x[0]+x[-1])/2
                            else:
                                df.at[idx, col] = x[1]

                    # return the list of values
                    else:
                        df.at[idx, col] = x

            # conversion to a float can succeed if filenames are numeric, so
            # we need to check seal parameters regardless (filenames aren't in whole-cell parameters)
            if col in pp:

                # check if (i,j)th cell is a filename using the following criteria
                #   1. should have 8 characters
                #   2. 18 < first two digits < 30, which represent years 2018 < year < 2030
                #   3. characters are alphanumeric
                if (len(s) == 8) and (18 < float(s[:2]) < 30) and s.isalnum():

                    # if filtering removed `s` from the index, query the unfiltered dataframe
                    # for simplicity, we query the unfiltered dataframe from the beginning
                    if s in unfiltered.index:
                        u = unfiltered.loc[s, :]
                    else:
                        raise Exception(
                            " Tried querying unfiltered `ephys_info` for filename %s, but failed." % s)

                    # if the referenced file `s` is the same as the current file `idx`, exit
                    if s in df.index:
                        if s == str(int(df.at[s, col])):
                            print(df)
                            raise Exception(
                                "\n The current cell value is its own filename at index `%s`. \n You may need to change the original .xlsx file. Exiting. \n" % s)

                    # set value of current cell to value of parent cell
                    df.at[idx, col] = u.loc[col]

                    # add to paired dictionary
                    # if parent is already in the dictionary, append it to the list of derivative recordings
                    if s in paired.keys():
                        if idx not in paired[s]:
                            paired[s].append(idx)

                    # else, add a new entry with value being list containing current filename
                    else:
                        paired.update({s: [idx]})

    if return_paired_files:
        return df, paired
    else:
        return df


def apply_inclusion_exclusion_criteria(df, col, criteria):
    """
    `df` is the dataframe to filter
    `col` = column name to filter
    `criteria` is a nested list containing [inclusion] and [exclusion] criteria

    Returns filtered dataframe as a copy, `out`
    """
    # copy dataframe to be filtered
    out = df.copy()

    # create mask by joining inclusion and exclusion criteria in their separate lists
    # masks[0] = inclusion criteria, masks[1] = exclusion criteria
    masks = ["|".join(c) for c in criteria]

    # inclusion
    if masks[0] != "all":
        out = out.loc[out[col].str.contains(masks[0], na=False)]

    # exclusion
    out = out.loc[~out[col].str.contains(masks[1], na=False)]

    return out


class EphysInfoFiltering():
    def __init__(self, criteria, EphysInfoPath=ephys_info):
        """
        `criteria` = Dictionary containing criteria to direct filtering of `ephys_info` dataframe
        `EphysInfoPath` = path to `ephys_info.xlsx`
        """
        # ephys data info
        ephys_info = pd.read_excel(EphysInfoPath, header=0, index_col=None)
        # convert Files column into string
        ephys_info = ephys_info.astype({'Files': 'str'})

        # columns that may have numeric values
        self.NumCols = ['DNA (ng)', 'Transfection \nreagent (ul)',
                        'OptiMEM (ul)', 'Time post-seed (h)',
                        'Time post-transfection (h)', 'R_pp (M)', 'R_sl (G)',
                        'C_m (pF)', 'R_m (M)', 'R_sr (M)', 'Tau, +20 (pA)', 'Leak (pA)']

        self.criteria = criteria
        self.ephys_info = ephys_info

    def criteria_msg(self, s):
        return "Criteria %s must be list of Strings, String, or list of [inclusion, exclusion] criteria." % s

    def FilterDates(self, EphysInfo, sel=None):
        """
        Apply filtering on dates
        `sel` = date criteria. If None, indexes `self.criteria["Dates"]`
        """

        if sel is None:
            dates = self.criteria["Dates"]
        else:
            dates = sel

        if len(dates) < 1:
            raise Exception(
                " No dates were selected for exclusion. To process all dates, use {'dates' : ['all']}")

        elif dates[0] == "all" and len(dates) == 1:
            print(" Selecting `all` dates.")

            # change type of protocol column to string to filter out nans
            EphysInfo["Protocol"] = EphysInfo["Protocol"].astype(str)

            # select every row in the ei spreadsheet if no NaNs are present
            EphysInfo = EphysInfo.loc[
                (EphysInfo["Files"] != 'nan') & (
                    EphysInfo["Protocol"] != 'nan')
            ].reset_index(drop=True)

        else:
            # row indices of files to keep
            to_keep = []

            # Each criteria value should be a List of Strings, or a nested List with two lists corresponding to [[inclusion], [exclusion]]
            if all(isinstance(d, list) for d in dates):
                if len(dates) == 2:
                    return apply_inclusion_exclusion_criteria(EphysInfo, "Files", dates)
                else:
                    raise Exception(self.criteria_msg("Dates"))

            for d in dates:
                # a `full` date is 8 characters long, regardless of month = YY M DD xxx, where M = {1-9, o, n, d}
                # any entry in `dates` that is < 8 characters is interpreted as an attempt to index a collection of files, if present

                if isinstance(d, str):
                    n = len(d)
                    if n < 8:
                        # find indices of filenames that match the date `d`
                        to_keep.extend([i for (i, x) in enumerate(
                            EphysInfo["Files"].values) if d in x])
                    else:
                        to_keep.extend([i for (i, x) in enumerate(
                            EphysInfo["Files"].values) if d == x])
                else:
                    raise Exception(
                        " Criteria can only be String or List of [Inclusion, Exclusion] criteria.")

            if len(to_keep) > 0:
                to_keep = list(set(to_keep))
                to_keep.sort()
            else:
                raise Exception(
                    " Number of files to keep based on dates is 0.")

            # select files based on dates (rows indices in `to_keep`)
            EphysInfo = EphysInfo.iloc[to_keep, :]
            # ensure that Protocols are not nan
            EphysInfo = EphysInfo.loc[EphysInfo["Protocol"]
                                      != 'nan'].reset_index(drop=True)

        return EphysInfo

    def FilterColumn(self, EphysInfo, col, sel=None):
        """
        Apply filtering on a column of `EphysInfo`
        `col` = name of column to filter, or key of `self.criteria`
        `sel` = Protocol criteria. If None, indexes `self.criteria["Protocol_Name"]`
        """

        # extract column to filter as `C`
        if col in self.ephys_info.columns:
            # extract the filenames and column of interest from unfiltered data
            C = self.ephys_info.loc[:, ["Files", col]]

            # ensure uniform type String
            C = C.astype(str).astype("string")
            # convert any string nans to floats
            C = C.replace('nan', np.nan).dropna()

            # if filenames are in values of `col`, replace filenames with parent values
            if any(C[col].isin(C["Files"])):

                # set filenames as index
                C.set_index("Files", drop=True, inplace=True)

                # convert to string type if not expected to have numeric values
                if col not in self.NumCols:
                    C[col] = C[col].astype(str).astype("string")
                    EphysInfo[col] = EphysInfo[col].astype(
                        str).astype("string")

                # replace filenames in column `col` of filtered df with parent value from unfiltered df
                # using the unfiltered dataframe avoids errors where the parent row was filtered out
                for i, s in enumerate(EphysInfo[col]):
                    if s in C.index:
                        EphysInfo[col].iat[i] = C.at[s, col]

            # `pd.Series` containing values in column `col` after replacement
            C = EphysInfo.loc[:, col]

        else:
            if col == "Dates":
                raise Exception(
                    "`FilterProtocol` was called to filter `Dates`, but `FilterDates` should be used instead.")
            elif col == "Protocol_Name":
                col = "Protocol"
                C = EphysInfo.loc[:, "Protocol"]
            else:
                raise Exception(
                    " Besides `Dates` and `Protocol_Name`, all keys in `filter_criteria` must be columns of `ephys_info.xlsx`.")

        # extract filtering criteria from `self.criteria`, unless the values to select `sel` are given
        if sel is None:
            if col in self.criteria.keys():
                v = self.criteria[col]
            else:
                if col == "Protocol":
                    v = self.criteria["Protocol_Name"]
                else:
                    raise Exception(
                        " Since criteria were not given, tried indexing `self.criteria` with `col` as a key, but failed. Try setting `col` as a key in `self.criteria`.")
        else:
            v = sel

        # select matching row values
        if isinstance(v, str):
            if v == "all":
                return EphysInfo
            else:
                return EphysInfo.loc[C.str.contains(v, case=False, na=False)]

        elif isinstance(v, list):
            if v[0] == "all":
                return EphysInfo
            else:
                # nested list of inclusion, exclusion criteria
                if all(isinstance(x, list) for x in v) and len(v) == 2:
                    return apply_inclusion_exclusion_criteria(EphysInfo, col, v)

                # list of strings
                elif all(isinstance(x, str) for x in v):
                    if len(v) > 1:
                        # join strings with 'OR' operator into boolean mask
                        mask = '|'.join(v)
                        return EphysInfo.loc[C.str.contains(mask, case=False, na=False)]
                    else:
                        return EphysInfo.loc[C.str.contains(v[0], case=False, na=False)]
                else:
                    raise Exception(self.criteria_msg(col))
        else:
            raise Exception(self.criteria_msg(col))

        raise Exception("   Filtering %s with criteria %s failed." % (v, col))

    def filter(self):
        """
        Apply criteria in `criteria` to filter `ephys_info`

        `criteria` = dictionary with usual keys "Dates", "Files_to_Skip" and "Protocol_Name"

        `criteria["Dates"]` is a List of Strings, or a List of Lists of Strings.
            * The length of each String must be no greater than 8 characters (maximum length of any recording's filename).
                - If the length is 8 characters, then it specifies a single recording to extract.
                - If the length is less than 8 characters, then we search for a set of recordings that match the sub-string.
                - Filenames have format `YY M DD xxx`, where `M` is month (1-9 for January-September, and 'o', 'n', and 'd' for October-December)
            * A nested List of Strings is treated similarly. We choose not to flatten a nested list to allow for some list elements to be Strings themselves.

        `criteria["Files_to_Skip"]` should be a List of Strings, each String being the full 8-character filenames to skip.

        `criteria["Protocol_Name"]` is a String or a List of Strings corresponding to protocols to select.
            * In either case, the String(s) may be full or partial names of protocols.
            * To avoid including undesired files (e.g. which satisfy a protocol in `Protocol_Name` as a [sub]-string, and other criteria above), consider specifying such files in `Files_To_Skip`

        Other keys in `criteria` must match the name of a column in `ephys_info.xlsx`. The values of these keys may be:
            1. A String or List of Strings, if the column `dtype` is String
            2. An Int/Float or List of Int/Float if the column `dtype` is Int/Float

        Returns `filenames` and `ei`
            * `filenames` = list of filenames from filtered dataframe
            * `ei` = filtered ephys_info dataframe
        """
        # read ephys_info dataframe
        ei = self.ephys_info.copy()

        if len(self.criteria.keys()) < 1:
            raise Exception(
                "Processing cannot proceed w/o filtering criteria. At least provide 'all' as the value for each key in the dictionary `criteria`.")

        for k, v in self.criteria.items():
            print("\n    Filtering... {k} = {v}".format(k=k, v=v))

            if k == "Dates":
                ei = self.FilterDates(ei, v)

            elif k == "Files_To_Skip":
                files_to_skip = self.criteria["Files_To_Skip"]
                # print(" Skipping files specified in 'Files_To_Skip' \n ", files_to_skip)

                if len(files_to_skip) > 0:
                    ei = ei.loc[~ei["Files"].isin(files_to_skip)]

            else:
                print(ei)
                print(k)
                print(v)
                ei = self.FilterColumn(ei, k, sel=v)

        if ei.shape[0] < 1:
            raise Exception("No files found.")

        else:
            print("\n %d files found for these criteria..." % ei.shape[0])
            ei.reset_index(drop=True, inplace=True)
            print(ei.loc[:, ['Files', 'Protocol']])

            # names of files
            filenames = ei['Files'].values.tolist()

            return filenames, ei

    def ExpParams(self, EphysInfo):
        """
        Isolate experimental parameters from `EphysInfo` \\
        Returns `exp_params`, a dataframe containing just floats for seal and whole-cell parameters, and `paired_files`, a dictionary that specifies recordings from the same cell, which has the following structure: `{ 'parent' filename : [file1, file2, file3, ..., [act1, act2, act3, ...]] }`.

        Structure of `paired_files`:
            `parent` is the first recording and contains the seal and experimental parameters in `ephys_info.xlsx`.
            `file1`, `file2`, etc. are subsequent files recorded with non-activation protocols, and only have their whole-cell values specified in `ephys_info.xlsx`
            `act1`, etc. are activation recordings from the same cell as above.
            If
        """

        # use original dataframe (unfiltered) and extract experimental parameters
        # retrieve experimental parameters with filenames as index
        unfiltered = self.ephys_info.loc[:, [
            'R_pp (M)', 'R_sl (G)', 'C_m (pF)', 'R_m (M)', 'R_sr (M)']]
        unfiltered.index = self.ephys_info['Files']

        # search for indices that are given in `EphysInfo` (filtered)
        exp_params = EphysInfo.loc[:, [
            'R_pp (M)', 'R_sl (G)', 'C_m (pF)', 'R_m (M)', 'R_sr (M)']]
        exp_params.index = EphysInfo["Files"]

        # make sure experimental params are numbers
        exp_params, paired_files = clean_up_params(
            exp_params, unfiltered, single=True)
        print("\n Corresponding experimental parameters...")
        print(exp_params)
        print(pd.concat(
            [exp_params.mean(axis=0), exp_params.sem(axis=0), exp_params.count(axis=0)],
            axis=0, keys=["Mean", "SEM", "Count"]
        ))

        if len(paired_files.keys()) < 1:
            print(
                " No paired files were found. The second return variable is an empty list.")
            return [], exp_params

        else:

            # find activation protocol in protocols from the same cell
            for key, val in paired_files.items():
                # use a copy to avoid modifying the actual dictionary values
                tmp = val.copy()
                # insert name of parent file at beginning of list of subsequent files
                tmp.insert(0, key)

                # find all protocols of filenames in `tmp`
                prox = EphysInfo.loc[EphysInfo["Files"].isin(
                    tmp), ["Files", "Protocol"]]
                # find the indices of activation protocols in the above
                act_ = np.where(prox["Protocol"].str.contains("_act"))
                # corresponding filenames to list
                act_ = prox["Files"].iloc[act_].values.tolist()

                # append activation filenames to the end of the subsequent filenames for current cell
                if len(act_) > 0:
                    paired_files[key].append(act_)

            print("\n Files recorded from the same cell...")
            # { parent filename : [subsequent filenames, [activation filenames]] }
            print(paired_files)

            return paired_files, exp_params

    def CreatePrefix(self, skip=True, exclusion=True):
        """
        Create file prefix from dictionary of filter criteria

        Returns `prefix`, where intra-criteria entries are separated by `-` and different criteria types (e.g. Dates, Protocol) are separated by `__`
        `exclusion` = include exclusion criteria
        `skip` = include skipped filenames
        """

        # Currently just assume every dict value is a List of Strings, but later add compatibility with nested List for exclusion criteria
        try:
            prefix = ["-".join(v) for k, v in self.criteria.items()
                      if k != "Files_To_Skip"]
        except:

            prefix = []
            for k, v in self.criteria.items():

                if not skip:
                    if k == "Files_To_Skip":
                        continue

                if isinstance(v, str):
                    prefix.append(v)
                elif all(isinstance(x, str) for x in v):
                    prefix.append("-".join(v))
                elif all(isinstance(x, list) for x in v) and len(v) == 2:
                    s = ""

                    if len(v[0]) > 1:
                        if v[0][0] == "all":
                            pass
                        else:
                            s += "-".join(v[0])

                    if exclusion:
                        if len(v[1]) > 1:
                            s += "-!"
                            s += "-".join(v[1])

                    if len(s) > 1:
                        prefix.append(s)

                else:
                    raise Exception(
                        "\n Tried creating prefix by joining each value in `criteria` with '-'. If this didn't work, one of the values might be a nested list that doesn't follow [inclusion, exclusion] format.")

        prefix = "__".join(prefix)
        return prefix

    def frequency_plots(self, df):
        """
        Use `pd.Series.value_counts` to create histogram for each column in `df` if > 4 unique values, otherwise use pie plot \\
        """
        N = int(math.ceil(df.shape[1]/3))
        fig, ax = plt.subplots(3, N, figsize=(14, 6))

        cols = df.columns

        k = 0
        for i in range(3):
            for j in range(N):
                # empty unused plots
                if k > df.shape[1]:
                    ax[i, j].axis("off")
                    continue

                # column name
                c = cols[k]

                # filter construct names to consider identity up to `pcDNA3`, and exclude `GFP`
                if c == "Construct":
                    s = pd.Series([x.split(" ")[:-1] for x in df.iloc[:, i]])
                    freq = s.value_counts()
                else:
                    freq = df.iloc[:, i].value_counts()

                if freq.shape[0] < 5:
                    ax[i, j].pie(
                        freq, labels=freq.index.values.tolist(), autopct="%.1f")
                    ax[i, j].set_frame_on(False)
                else:
                    nbins = int(math.ceil(freq.shape[0]/10))*5
                    ax[i, j].hist(freq, bins=nbins)

                    if freq.shape[0] <= 10:
                        ax[i, j].set_xticks(np.linspace(0, 1, freq.shape[0]))
                        ax[i, j].set_xticklabels(freq.index)
                    else:
                        ax[i, j].set_xticklabels(freq.index)

                k += 1

        plt.tight_layout()
        plt.show()
        plt.close()

    def FillDates(self, df):
        """
        Fill empty cells in "Date" column with previous non-empty cell's value using forward fill (`ffill`) method in `pd.DataFrame.fillna` \\
        Returns filled `Dates` as `pd.Series`
        """
        if "Date" in df.columns:
            dates = df.loc[:, "Date"]
            dates.fillna(method="ffill", inplace=True)
        else:
            raise Exception(
                "   No 'Date' column in `df`. This is needed for `FillDates`.")

        return pd.to_datetime(dates, format="%d-%m-%Y", errors='coerce')

    def do_stats(self, indep="all"):
        """
        Given criteria in initialization of class, visualize and/or run statistics on independent variable(s) in `indep`
        """
        if not isinstance(indep, list):
            raise Exception(
                " Independent variables in `indep` must be List of Strings.")

        # filenames of and filtered ephys_info
        fnames, filtered = self.filter()
        print(filtered)

        # get dates of the filtered dataframe by indexing filled dates of unfiltered dataframe with indices of filtered dataframe
        fdates = self.FillDates(self.ephys_info).iloc[filtered.index, :]

        # difference in rows
        print(" {x1} of {x2} rows ({x3}) remain after filtering `ephys_info.xlsx`.".format(
            x1=filtered.shape[0], x2=self.ephys_info.shape[0], x3=(filtered.shape[0]/self.ephys_info.shape[0]))
        )

        # `NumericState` tracks whether all (0), some (0.5), or none (1) of `indep` are numeric.
        # -1 is reserved for when a complete analysis of all variables is desired
        if indep == "all":
            NumericState = -1
        else:
            # check whether independent variable(s) are expected to have numeric values
            IsNumeric = [j in self.NumCols for j in indep]

            if all(isNumeric):
                print(" All independent variables are numeric.")
                NumericState = 0
            else:
                if any(isNumeric):
                    print(
                        " Independent variables include numeric and non-numeric data.")
                    NumericState = 0.5
                else:
                    print(" All independent variables are non-numeric.")
                    NumericState = 1

        if NumericState < 1:
            _, ExpParams = self.ExpParams(filtered)

            # pair plot of experimental parameters
            pp = sns.pairplot(ExpParams)

        # pie chart and histogram for categorical data
        # box plot and histogram for numerical data
        # time series for each

        plt.show()


class FindPairedFiles():
    def __init__(self, fname):
        """
        Convenience method to find paired files, given criteria in `fname`
        `fname` is expected to have the following structure:
            1. Dates separated by "-", followed by "__", then
            2. Protocol name(s), separated by "-", followed by "__", then
            3. "act_norm.csv"
        """
        self.fname = fname

    def ParseFName(self, f):
        """
        `f` is an `fname` described in `__init__`. The parsing function is defined separately so that it can be called repeatedly when `fname` is a list of filenames
        """
        # parse filename into Dates, Protocol, and "act_norm.csv"
        sub1 = f.split("__")
        if sub1[-1] != "act_norm.csv":
            raise Exception(
                "   `f` may not be formatted correctly. Expected 'act_norm.csv'. \n", sub1)

        if "-" in sub1[0]:
            dates = sub1[0].split("-")
        else:
            dates = [sub1[0]]

        if "-" in sub1[1]:
            protcls = sub1[1].split("-")
        else:
            protcls = [sub1[1]]

        return dates, protcls

    def Find(self):
        """
        Find paired files by first parsing `fname` into Dates and Protocols, then using these to filter `ephys_info.xlsx`
        """
        # parse `fname`
        if isinstance(self.fname, str):
            dates, protcls = self.ParseFName(self.fname)
        elif isinstance(self.fname, list):
            dates = []
            protcls = []

            # parse each filename in `fname` for Dates and Protocols, then append to respective lists without duplication
            for f in self.fname:
                d, p = self.ParseFName(f)

                if d not in dates:
                    dates.append(d)

                if p not in protcls:
                    protcls.append(p)
        else:
            raise Exception(
                "   Type of `self.fname` not understood. Expected a String or List of Strings. Given ", type(self.fname))

        # initialize class object with criteria given by `fname`
        F = EphysInfoFiltering({"Dates": dates, "Protocol_Name": protcls})
        # apply filtering over Dates and Protocols
        _, Filtered = F.filter()
        # find paired files (included in extraction of experimental parameters)
        paired_files, _ = F.ExpParams(Filtered)

        return paired_files
