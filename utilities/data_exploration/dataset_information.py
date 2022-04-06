import pandas as pd
import numpy as np
import sys
from typing import List, Optional, Tuple
import matplotlib
import os

if os.name == "nt":
    matplotlib.use("Agg", warn=False)
import matplotlib.pyplot as plt


def sizeof_fmt(num, suffix="B"):
    """
    This is a utility function that formats data sizes with appropriate unit.
    From: https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size

    :param num: Number to convert
    :param suffix: Base suffix for the number to convert
    :return: String with more user-friendly units
    """
    for unit in [" ", " K", " M", " G", " T", " P", " E", " Z"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Y", suffix)


def get_basic_data_information(frame, print_results=False):
    """
    This function obtains summary metadata information for the data file as a whole. The results
    are both output as inline text within the Jupyter notebook as well as saved in the dictionary
    that is returned from the function. This is useful for a quick sense check on any files provided
    by clients or generated internally, especially when comparing with similar datasets.

    :param frame: Dataframe for which to obtain metadata information
    :param print_results: User option to display some metadata information inline
    :return: Dictionary containing number of rows, number of columns, memory usage, and total memory usage of the dataset.
    """

    # Check that data is a pandas dataframe
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(
            "The dataset is not a pandas dataframe, please convert it into a pandas dataframe"
        )

    # Get basic data information
    data_information = {
        "Rows": frame.shape[0],
        "Columns": frame.shape[1],
        # "Headers": list(frame.columns.values),
        # "Sorted Headers": list(np.sort(frame.columns.values)),
        "Memory Usage": sizeof_fmt(frame.memory_usage(deep=True).sum()),
        "Total Memory Usage": sizeof_fmt(sys.getsizeof(frame)),
    }

    temp_data = pd.DataFrame.from_dict(
        data=data_information, orient="index"
    ).reset_index()
    temp_data.columns = ["Item", "Value"]

    # print(temp_data) # Used for debugging

    # Obtain the data information
    data_rows = data_information["Rows"]
    data_columns = data_information["Columns"]
    data_mem_usage = data_information["Memory Usage"]
    data_total_usage = data_information["Total Memory Usage"]

    if print_results:
        # Display the data information
        print("Summary diagnostics for the uploaded dataset:")
        print("Number of rows: " + str(data_rows))
        print("Number of columns: " + str(data_columns))
        print("Memory Usage: " + str(data_mem_usage))
        print("Total Memory Usage: " + str(data_total_usage))
        print("")

    return data_information


def get_basic_variable_information(
    frame, key_vars: Optional[List[str]] = None, basic_output: Optional[bool] = False
):
    """
    This function obtains summary information for each variable / column in the dataframe. The results
    are saved as a dictionary. The table is displayed inline within the Jupyter notebook. This provides
    important diagnostic traits and checks, such as % nulls and blanks. Additional functionality and checks
    can be easily built by adding to the dictionary.

    :param frame: Dataframe to obtain metadata information on
    :param key_vars: variables for which to get information, use None to indicate all variables
    :param basic_output: Whether to only output basic results (data type, nulls, blanks, zeros, uniques, most common)
    :return: Dictionary containing data type, #/% nulls, #/% blanks, #/% zeros, sum, mean, min, 1%, 25%, 50%, 75%, 99%,
        max, and the most common result for each variable.
    """

    # Check that data is a pandas dataframe
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(
            "The dataset is not a pandas dataframe, please convert it into a pandas dataframe"
        )

    if key_vars is None:
        key_vars = frame.columns

    # Iterate through variables
    for idx in range(len(key_vars)):

        variable = key_vars[idx]

        this_variable = frame[variable]  # Retrieve data from pandas column
        this_type = this_variable.dtypes  # Retrieve data type
        this_nulls = (
            this_variable.isnull().sum()
        )  # Obtain nulls for non-numeric variables
        this_count_no_na = pd.Series(
            this_variable
        ).value_counts()  # Create series of values and value counts
        this_uniques = this_variable.nunique()

        # This check is built to filter out any columns that contain no data / all nulls
        if len(this_count_no_na) == 0:
            this_maxpos = None
            this_count_no_na = [0]
        else:
            this_maxpos = np.argmax(np.array(this_count_no_na))  # Count of mode
            this_count_no_na = this_count_no_na.tolist()

        this_rows = frame.shape[0]

        if not pd.api.types.is_numeric_dtype(this_type):
            this_blanks = np.sum(this_variable == "")
            this_dict_result = {
                "Variable": variable,
                "Type": this_type,
                "# Nulls": this_nulls,
                "% Nulls": this_nulls / this_rows * 100,
                "# Blanks": this_blanks,
                "% Blanks": this_blanks / this_rows * 100,
                "# Zeros": None,  # if a dict entry is None, this is because it does not apply
                "% Zeros": None,
                "# Unique": this_uniques,
                "Most Common (non NA)": this_maxpos,
                "Common Count": this_count_no_na[0],
                "Common Count %": this_count_no_na[0] / this_rows * 100,
            }

            if basic_output is False:
                this_dict_result.update(
                    {
                        "Sum": None,
                        "Mean": None,
                        "Min": None,
                        "1%": None,
                        "25%": None,
                        "50%": None,
                        "75%": None,
                        "99%": None,
                        "Max": None,
                    }
                )

        else:

            this_zeros = np.sum(this_variable == 0)
            this_dict_result = {
                "Variable": variable,
                "Type": this_type,
                "# Nulls": this_nulls,
                "% Nulls": this_nulls / this_rows * 100,
                "# Blanks": None,
                "% Blanks": None,
                "# Zeros": this_zeros,
                "% Zeros": this_zeros / this_rows * 100,
                "# Unique": this_uniques,
                "Most Common (non NA)": this_maxpos,
                "Common Count": this_count_no_na[0],
                "Common Count %": this_count_no_na[0] / this_rows * 100,
            }

            if basic_output is False:

                this_min = np.min(this_variable)
                this_max = np.max(this_variable)
                this_mean = np.mean(this_variable)
                this_sum = np.sum(this_variable)
                this_1pc = np.nanquantile(this_variable, 0.01)
                this_25pc = np.nanquantile(this_variable, 0.25)
                this_50pc = np.nanquantile(this_variable, 0.50)
                this_75pc = np.nanquantile(this_variable, 0.75)
                this_99pc = np.nanquantile(this_variable, 0.99)

                this_dict_result.update(
                    {
                        "Sum": this_sum,
                        "Mean": this_mean,
                        "Min": this_min,
                        "1%": this_1pc,
                        "25%": this_25pc,
                        "50%": this_50pc,
                        "75%": this_75pc,
                        "99%": this_99pc,
                        "Max": this_max,
                    }
                )

        this_result = pd.DataFrame(
            data=this_dict_result, columns=this_dict_result.keys(), index=[0]
        )

        if idx == 0:
            result = this_result
        else:
            result = pd.concat([result, this_result], axis=0)

    return result.reset_index(drop=True)


def null_column_checks(
    frame,
    name: Optional[str] = None,
    save_path: Optional[str] = None,
    save_graph: bool = False,
    null_threshold: int = 50,
    figsize: Optional[Tuple[float, float]] = None,
    print_warn=False,
):
    """
    If called, performs checks on the null values of the dataframe by columns.

    :param frame: Dataframe to perform null column checks on
    :param name: Name of dataframe, must be provided if save_graph = True
    :param save_path: Path to save the null column check figure in, must be provided if to_graph = True
    :param save_graph: User option to generate a figure visually representing the null count by column
    :param null_threshold: % to specify the threshold for nulls - beyond the % will throw a warning
    :param figsize: Specify the dimensions of the figure to be generated
    :param print_warn: Print message when variables have more than null_threshold number of Nulls
    :return: Tuple with List of variables breaching the null_threshold, and with separate series that contains just the
        null values, sorted by most nulls to least.
    """

    # Check arguments
    if save_graph:
        if save_path is None:
            raise ValueError(
                "Must specify the path to save the figure as the argument 'save_path'"
            )
        if name is None:
            raise ValueError(
                "Must specify the name of the dataset as the argument 'name'"
            )

    # Set up lists to be filled with null values to later graph
    null_list = []
    total_rows = frame.shape[0]

    null_vars = []

    if figsize is None:
        figsize = (10, 0.2 * len(frame.columns))

    # Checks nulls in each column, flagging high % null columns
    for variable in frame.columns:
        var_nulls = frame[variable].isnull().sum()
        null_pc = var_nulls / total_rows * 100
        if null_pc > null_threshold:
            null_vars.append(variable)
        null_list.append(var_nulls)

    if print_warn:
        if len(null_vars):
            print(
                "The following variables have more than {nulls}% nulls:".format(
                    nulls=null_threshold
                )
            )
            print("\n- ".join(str(null_vars)))
        else:
            print(
                "No variables with more than {nulls}% nulls".format(
                    nulls=null_threshold
                )
            )

    null_frame = pd.Series(null_list, index=frame.columns)
    # Sort the data by largest to least null count for ease of analysis
    sorted_null_frame = null_frame.sort_values(axis=0, ascending=True)

    # Plot and save the graph
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(sorted_null_frame))
    ax.barh(y_pos, sorted_null_frame, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_null_frame.index)
    if name is None:
        ax.set_title("Column nulls for dataset", fontsize=18)
    else:
        ax.set_title("Column nulls for {}".format(name), fontsize=18)
    ax.set_ylabel("Variable", fontsize=14)
    ax.set_xlabel("Null count", fontsize=14)
    ax.set_xlim(left=0)

    if save_graph:
        fig.savefig(save_path + "/null_cols_{}".format(name))

    return null_vars, sorted_null_frame, fig


def null_row_checks(
    frame,
    name,
    save_path: Optional[str] = None,
    save_graph: Optional[bool] = False,
    figsize: Tuple[float, float] = (10, 6),
    max_size: int = 100000,
):
    """
    If called, performs checks on the null values of the dataframe by rows.

    :param frame: Dataframe to perform null column checks on
    :param name: Name of dataframe
    :param save_path: Path to save the null column check figure in
    :param save_graph: User option to generate a figure visually representing the null count by column
    :param figsize: Specify the dimensions of the figure to be generated
    :param max_size: maximum number of observations to plot
    :return: Separate series that contains just the null values, sorted by most nulls to least.
    """

    if name is None:
        raise ValueError("Must specify the name of the dataset as the argument 'name'")
    if save_graph:
        if save_path is None:
            raise ValueError(
                "Must specify the path to save the figure as the argument 'save_path'"
            )

    # Create a separate version of the dataframe that has the null rows information
    frame_with_nulls = frame.copy()
    frame_with_nulls = frame_with_nulls.reset_index(drop=True)
    if frame_with_nulls.shape[0] > max_size:
        frame_with_nulls = frame_with_nulls.sample(max_size)

    frame_with_nulls["null_row_sum"] = pd.Series(frame_with_nulls.isnull().sum(axis=1))

    # Plot and save the graph
    sorted_null_frame = frame_with_nulls["null_row_sum"]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sorted_null_frame, sorted_null_frame.index.values, "bo-")
    ax.set_title("Row nulls for {}".format(name), fontsize=18)
    ax.set_ylabel("Row", fontsize=14)
    ax.set_xlabel("Null count", fontsize=14)
    ax.set_xlim(left=0)
    ax.invert_yaxis()
    if save_graph:
        fig.savefig(save_path + "/null_rows_{}".format(name))

    return frame_with_nulls["null_row_sum"], fig


def compare_datasets(
    df_1, df_1_name, df_2, df_2_name, union_cols: Optional[str] = None
):
    """
    Generates a dataframe that shows the values of the union columns (or specified columns by user).

    :param df_1: First dataframe to compare
    :param df_1_name: Name of first dataframe
    :param df_2: Second dataframe to compare
    :param df_2_name: Name of second dataframe
    :param union_cols: Manually entered columns to example. If not, by default does union of 2 dataframes' columns
    :return: The comparison_clean dataframe that shows the values of variables in both datasets.
    """

    # Check for which specified columns (if applicable) are in which dataset
    if union_cols:
        union_cols_1 = list(set(union_cols).intersection(df_1.columns.values.tolist()))
        union_cols_2 = list(set(union_cols).intersection(df_2.columns.values.tolist()))
    else:
        union_cols_1 = None
        union_cols_2 = None

    df_1_metadata = get_basic_data_information(df_1)
    df_2_metadata = get_basic_data_information(df_2)
    df_1_key_data = get_basic_variable_information(
        df_1, key_vars=union_cols_1, basic_output=False
    )
    df_2_key_data = get_basic_variable_information(
        df_2, key_vars=union_cols_2, basic_output=False
    )

    # Specify the key variables to be used, can be added to
    key_vars = ["Sum", "# Nulls", "% Nulls", "# Unique"]

    if union_cols:
        neither_list = list(
            set(union_cols).difference(
                df_1.columns.values.tolist(), df_2.columns.values.tolist()
            )
        )
        if len(neither_list) > 0:
            print(
                "{} are columns listed in union_cols, but are not in either dataframe".format(
                    neither_list
                )
            )

    # Generate the dataframes, add flag columns
    comparison_2 = (
        df_1_key_data[["Variable", "Type"] + key_vars]
        .set_index(["Variable", "Type"])
        .stack()
        .reset_index()
    )
    comparison_2["{}_flag".format(df_1_name)] = 1
    comparison_other = (
        df_2_key_data[["Variable", "Type"] + key_vars]
        .set_index(["Variable", "Type"])
        .stack()
        .reset_index()
    )
    comparison_other["{}_flag".format(df_2_name)] = 1

    # Rename columns in dataframes
    comparison_2 = comparison_2.rename(
        index=str, columns={"level_2": "Check", 0: "Value"}
    )
    comparison_other = comparison_other.rename(
        index=str, columns={"level_2": "Check", 0: "Value"}
    )

    # Merge
    comparison_merge = comparison_2.merge(
        comparison_other, how="outer", on=["Variable", "Type", "Check"]
    )
    comparison_clean = comparison_merge.rename(
        index=str,
        columns={
            "Value_x": "Value from {}".format(df_1_name),
            "Value_y": "Value from {}".format(df_2_name),
        },
    )

    # Generate metadata statistics
    meta_comparison = None

    meta_data = [
        "Rows",
        "Columns",
    ]  # TODO: Need to be able to get string values comparison

    for data in meta_data:
        dict_comparison = {
            "Variable": "Metadata",
            "Type": "int64",
            "Check": data,
            "Value from {}".format(df_1_name): df_1_metadata[data],
            "{}_flag".format(df_1_name): 1,
            "Value from {}".format(df_2_name): df_2_metadata[data],
            "{}_flag".format(df_2_name): 1,
        }

        this_comparison = pd.DataFrame(
            data=dict_comparison, columns=dict_comparison.keys(), index=[0]
        )

        if meta_comparison is None:
            meta_comparison = this_comparison
        else:
            meta_comparison = pd.concat([meta_comparison, this_comparison], axis=0)

    comparison_clean = comparison_clean.append(meta_comparison, ignore_index=True)

    # Add discrepancy columns
    comparison_clean["Discrepancy"] = (
        comparison_clean["Value from {}".format(df_2_name)]
        - comparison_clean["Value from {}".format(df_1_name)]
    )

    comparison_clean["% Discrepancy"] = None
    for i in comparison_clean.index:
        try:
            comparison_clean["% Discrepancy"].loc[i] = (
                comparison_clean["Discrepancy"].loc[i]
                / comparison_clean["Value from {}".format(df_1_name)].loc[i]
                * 100
            )
        except ZeroDivisionError:
            comparison_clean["% Discrepancy"].loc[i] = None

    comparison_clean_pd = pd.DataFrame(
        comparison_clean, columns=comparison_clean.columns, index=comparison_clean.index
    )

    return comparison_clean_pd
