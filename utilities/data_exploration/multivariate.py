import pandas as pd
import numpy as np
import matplotlib
import os

if os.name == "nt":
    matplotlib.use("Agg", warn=False)
# else:
# matplotlib.use("TkAgg")
from ..general_utilities import bucket_numerical


def two_way_tables(
    frame, var1, var2, var1_type="auto", var2_type="auto", num_groups=5, with_nan=True
):
    """
    Generate two-way table for Pandas (using cross-tab functionality), with capability to bucket numerical variables
    if requested (rather than using every single numerical value as a separate bucket.
    :param frame: pandas dataframe containing the variables for the two-way table
    :param var1: name of the first variable
    :param var2: name of the second variable
    :param var1_type: string that is either "auto", "num", "cat" with the following meaning
    "auto": numerical variables are bucketed, and other variables are treated as categorical
    "cat": treat as if it were categorical
    "num": treat as numerical (only allowed for numerical variables)
    :param var2_type: same description as var2_type
    :param num_groups: for numerical variables that are bucketed, how many buckets to use (NA bucket will be separate)
    :param with_nan: boolean for whether to count NaN or None values
    :return: pandas dataframe (generated using pd.crosstab)
    """

    dataset_slim = frame[[var1, var2]].copy()
    numerics = dataset_slim.select_dtypes(include=[np.number]).columns.values

    # Check variable types and treat accordingly
    if var1_type in ["auto", "num"]:
        if var1_type == "num" and var1 not in numerics:
            raise TypeError(
                "Variable var1 is not numerical and can therefore not be treated as a numerical"
            )
        elif var1 in numerics:
            quantile_cuts, *ignored = bucket_numerical(
                dataset_slim[var1], num_groups, with_inf=False
            )
            var1 = "{}_binned".format(var1)
            dataset_slim[var1] = quantile_cuts

    if var2_type in ["auto", "num"]:
        if var2_type == "num" and var2 not in numerics:
            raise TypeError(
                "Variable var2 is not numerical and can therefore not be treated as a numerical"
            )
        elif var2 in numerics:
            quantile_cuts, *ignored = bucket_numerical(
                dataset_slim[var2], num_groups, with_inf=False
            )
            var2 = "{}_binned".format(var2)
            dataset_slim[var2] = quantile_cuts

    # Perform cross-tabulation
    if with_nan:
        crosstab = pd.crosstab(
            dataset_slim[var1].apply(str),
            dataset_slim[var2].apply(str),
            margins=True,
            dropna=False,
        )
    else:
        crosstab = pd.crosstab(
            dataset_slim[var1], dataset_slim[var2], margins=True, dropna=False
        )

    return crosstab


def nway_freq(frame, var_names, treat_numerical="num", num_groups=5):
    """
    Generate nway frequency table.

    :param frame: pandas dataframe with the relevant variables
    :param var_names: list of variable names that will be used to group the data
    :param treat_numerical: whether to bucket numerical variables or treat as categorical (each unique value of the IV)
    :param num_groups: number of buckets to use for numerical variables that are not treated as categorical
    :return: pandas DataFrame with the frequency table
    """

    # Create dataframe to be summarized
    dataset_used = pd.DataFrame()

    # Identify all numerical fields
    numerics = frame[var_names].select_dtypes(include=[np.number]).columns.values

    for var_name in var_names:
        if var_name in numerics and treat_numerical == "num":
            temp_data, *ignored = bucket_numerical(
                frame[var_name], num_groups, with_inf=False
            )
            dataset_used["{}_bucketed".format(var_name)] = temp_data
        else:
            temp_data = frame[var_name].apply(str)
            dataset_used[var_name] = temp_data

    # Generate results
    output_table = pd.DataFrame(
        dataset_used.groupby(list(dataset_used.columns.values)).size()
    )
    output_table.columns = ["Count"]

    return output_table


def all_or_no_missing(frame, var_names):
    """
    Check if all values in an observation are missing at the same time, or non-missing at the same time. This is
    useful for variables that are conditional on eachother or a common other factor (e.g. costs associated with
    loan losses are only available after default).

    :param frame: pandas dataframe with the relevant variables
    :param var_names: list of variable names that should be checked
    :return: tuple, with the first element being a boolean indicating whether or not the conditions are satisfied (i.e. no concurrent missing and non-missing among the variables, and the second element being a boolean pd.Series with TRUE indicating a consistent observation, and FALSE indicating problematic observation
    """

    if len(var_names) < 2:
        raise ValueError("You need at least two variables to perform this analysis")

    result = frame[var_names].apply(
        lambda x: x.isnull().all() or (~x.isnull()).all(), axis=1
    )
    summary = result.all()

    return summary, result


def all_or_no_zero(frame, var_names, allow_NA=False):
    """
    Check if all values in an observation are zero at the same time, or non-zero the same time. This is
    useful for variables that are conditional on eachother or a common other factor.

    :param frame: pandas dataframe with the relevant variables
    :param var_names: list of variable names that should be checked. These must be numerical in nature.
    :param allow_NA: boolean indicating whether to consider
    :return: tuple, with the first element being a boolean indicating whether or not the conditions are satisfied (i.e. no concurrent zero and non-zero among the variables), and the second element being a boolean pd.Series with TRUE indicating a consistent observation, and FALSE indicating problematic observation
    """

    if len(var_names) < 2:
        raise ValueError("You need at least two variables to perform this analysis")

    if any(
        var_names
        not in frame[var_names].select_dtypes(include=[np.number]).columns.values
    ):
        raise TypeError("One or more of your variables are not numeric")

    if allow_NA:
        frame = frame[var_names].fillna(0)

    result = frame[var_names].apply(lambda x: (x == 0).all() or (x != 0).all(), axis=1)
    summary = result.all()

    return summary, result


def check_hierarchy(frame, var_names):
    """
    Check that the there is are unique one-directional mapping between variables, i.e. no many-to-many mapping.

    :param frame: pandas dataframe
    :param var_names: list of variables to check, with the least granular variable first (e.g. MSA, STATE, REGION)
    :return: tuple, with the first element being a boolean indicating whether or not the conditions are satisfied (i.e. no many-to-many mappings), and the second element a dataframe highlighting where there are issues (if any)
    """

    if len(var_names) < 2:
        raise ValueError("You need at least two variables to perform this analysis")

    small_frame = frame[var_names]
    problematic = []
    for stage in range(len(var_names) - 1):
        use_var_names = var_names[stage:]
        small_frame = small_frame[use_var_names]
        unique_table = small_frame.drop_duplicates()

        # Check if table indeed is unique
        duplicated_index = unique_table[var_names[stage]].duplicated(keep=False)

        if duplicated_index.any():
            unique_table["__PROBLEMATIC"] = duplicated_index
            problematic.append(unique_table)

    return len(problematic) > 0, problematic
