import pandas as pd
import numpy as np
import re
from .univariate import univariate_summary_categorical
from sklearn.feature_extraction.text import CountVectorizer


def get_dataset_string_info(dataset):
    """
    Reads in a dataset and gets general dataset information for each column, and generates a list of potential
    string / categorical columns that are eligible for string-specific diagnostics.

    :param dataset: Dataset containing string/categorical columns
    :return: Information describing each column in input dataset
    :return: List of potential string / categorical columns
    """

    # Create list for non-numeric columns that are eligible for string/categorical diagnostics
    numeric_list = []
    string_cat_list = []
    for column in dataset.columns.tolist():
        try:
            _ = dataset[column].astype("float")
            numeric_list.append(column)
        except:
            string_cat_list.append(column)

    print(
        "--------------------------\nPotential numeric columns:\n--------------------------"
    )
    print(numeric_list)

    print(
        "\n-----------------------------------------\n"
        + "Potential string and categorical columns:\n-----------------------------------------"
    )
    print(string_cat_list)

    # Generate general dataset info before diving into one column
    dataset_all_cols = []
    dataset_len = len(dataset)

    for column in string_cat_list:
        diagnostics = {
            "column_name": column,
            "unique_values": len(dataset[column].drop_duplicates()),
            "datatype": dataset[column].dtype,
            "% null": (
                len(dataset[column].loc[dataset[column].isnull() == True])
                / dataset_len  # noqa: E712
            )
            * 100,
        }
        diagnostics = pd.DataFrame(
            data=diagnostics, columns=diagnostics.keys(), index=[0]
        )
        dataset_all_cols.append(diagnostics)

    dataset_all_cols = pd.concat(dataset_all_cols, sort=False)
    dataset_all_cols.reset_index(inplace=True)
    dataset_all_cols.drop(columns="index", inplace=True)

    return dataset_all_cols, string_cat_list


def break_out_value_types(dataset, col_to_use, special_char=r"\-.,\s"):
    """
    For a user-specified column, specifies whether each row is numeric, string, or a mix of both. Using a univariate
    summary function visuals and basic stats are also provided for each population (numeric, string, None, mix).

    :param dataset: Dataset containing the specified column
    :param col_to_use: User-specified string / categorical column
    :param special_char: Special characters that should be ignored when checking for pure strings
    :return: Pandas dataframe where the first column contains the 'col_to_use' data, and four additional columns with
        boolean flags for numeric, string, mix, and null
    :return: List of figures containing the univariate summary for numeric, string, and mixed data
    :return: Dictionary of univariate outputs for numeric, string, and mixed data
    """

    # Create flags for whether is numeric, is string, None, or has both elements
    unique_values_df = dataset[[col_to_use]].drop_duplicates()
    unique_values_df["is_numeric"] = (
        unique_values_df[col_to_use]
        .apply(lambda value: str(value).replace(".", "", 1).isdigit())
        .fillna(False)
    )

    unique_values_df["is_null"] = (
        unique_values_df[col_to_use].isin([np.nan, None]).fillna(False)
    )

    unique_values_df["is_alphabetic"] = unique_values_df[col_to_use].apply(
        lambda value: re.sub(r"[{}]".format(special_char), "", str(value)).isalpha()
    ).fillna(False) & (~unique_values_df["is_null"])

    unique_values_df["is_mixed"] = ~(
        unique_values_df["is_numeric"]
        | unique_values_df["is_alphabetic"]
        | unique_values_df["is_null"]
    )

    # Join the unique values back to full dataset
    unique_values_df_indexed = unique_values_df.set_index(col_to_use)
    temp_dataset = dataset[[col_to_use]].join(
        unique_values_df_indexed, on=col_to_use, how="left"
    )

    # Portion of dataset values that contain only numbers
    try:
        figure_numeric, output_dict_numeric = univariate_summary_categorical(
            temp_dataset[col_to_use].loc[temp_dataset["is_numeric"]],
            col_to_use + " (numeric)",
        )
    except IndexError:
        figure_numeric = None
        output_dict_numeric = None
        print("There are no numeric values in {}".format(col_to_use))

    # Portion of dataset values that contain only strings
    try:
        figure_string, output_dict_string = univariate_summary_categorical(
            temp_dataset[col_to_use].loc[temp_dataset["is_alphabetic"]].astype("str"),
            col_to_use + " (alphabetic)",
        )
    except IndexError:
        figure_string = None
        output_dict_string = None
        print("There are no values in {} that are only alphabetic".format(col_to_use))

    # Portion of dataset values that contain mix of numbers and strings
    figure_mix, output_dict_mix = univariate_summary_categorical(
        temp_dataset[col_to_use].loc[temp_dataset["is_mixed"]], col_to_use + " (mix)"
    )

    # Convert series of figure and output variables into a dictionary
    figure_list = {}
    figure_list["numeric"] = figure_numeric
    figure_list["alphabetic"] = figure_string
    figure_list["mix"] = figure_mix

    output_dict_list = {}
    output_dict_list["numeric"] = output_dict_numeric
    output_dict_list["alphabetic"] = output_dict_string
    output_dict_list["mix"] = output_dict_mix

    return temp_dataset, figure_list, output_dict_list


def consistent_use_of_case(dataset, col_to_use):
    """
    Checks that the user-specified column has a consistent use of cases (uppercase, lowercase).

    :param dataset: Dataset containing the specified (string) column
    :param col_to_use: User-specified string / categorical column
    :return: Table containing metrics and cleaning suggestions for the consistent use of case
    """

    if len(dataset[col_to_use].str.lower().drop_duplicates()) == len(
        dataset[col_to_use].drop_duplicates()
    ):
        print("No standardization of case is required for {}.\n".format(col_to_use))
        cleaning = 0
        metric = "No cleaning required"
    else:
        print(
            "There are strings with the same text but different capitalization. "
            + "Standardization of case is required for {}.\n".format(col_to_use)
        )
        cleaning = 1
        metric = "Refer to outputs of view_inconsistent_cases"

    info_df = {
        "column": col_to_use,
        "check_type": "(In)consistent use of cases",
        "needs_cleaning": cleaning,
        "metric": metric,
    }
    info_df = pd.DataFrame(info_df, columns=info_df.keys(), index=[0])

    return info_df


def trailing_leading_spaces(dataset, col_to_use, dataset_len):
    """
    Checks whether the user-specified column has trailing and/or leading spaces.

    :param dataset: Dataset containing the specified (string) column
    :param col_to_use: User-specified string / categorical column
    :param dataset_len: row count of dataset input
    :return: Table containing metrics and cleaning suggestions for any trailing / leading spaces
    """

    if any(dataset[col_to_use].str.strip() != dataset[col_to_use]):
        # get a count of rows that have leading/trailing spaces
        dataset["has_trailing_leading_spaces"] = np.where(
            (dataset[col_to_use].str.strip() != dataset[col_to_use]), 1, 0
        )
        trailing_space_count = len(
            dataset.loc[dataset["has_trailing_leading_spaces"] == 1]
        )
        trailing_space_pc = "%.3f" % (trailing_space_count / dataset_len * 100)
        print(
            "{}% (count {}) of rows in {} have leading/trailing spaces which may need to be removed.\n".format(
                trailing_space_pc, trailing_space_count, col_to_use
            )
        )
        cleaning = 1
        metric = "{}% (count {}) have leading/trailing spaces".format(
            trailing_space_pc, trailing_space_count, col_to_use
        )
    else:
        print(
            "There are no leading or trailing spaces for values in {}.\n".format(
                col_to_use
            )
        )
        cleaning = 0
        metric = "No cleaning required"

    info_df = {
        "column": col_to_use,
        "check_type": "Trailing/leading spaces",
        "needs_cleaning": cleaning,
        "metric": metric,
    }
    info_df = pd.DataFrame(info_df, columns=info_df.keys(), index=[0])

    return info_df


def check_double_spaces(dataset, col_to_use, dataset_len):
    """
    Checks whether the user-specified column has double spaces.

    :param dataset: Dataset containing the specified (string) column
    :param col_to_use: User-specified string / categorical column
    :param dataset_len: row count of dataset input
    :return: Table containing metics and cleaning suggestions for any double spaces
    """

    if dataset[col_to_use].str.replace("  ", "check").equals(dataset[col_to_use]):
        print("There are no double spaces in {}.\n".format(col_to_use))
        cleaning = 0
        metric = "No cleaning required"
    else:
        # get a count of rows that have double spaces
        number_of_double_space = dataset[col_to_use].str.contains("  ").sum()

        double_space_pc = "%.3f" % (number_of_double_space / dataset_len * 100)
        print(
            "{}% (count {}) of rows in {} have double spaces which may need to be removed.\n".format(
                double_space_pc, number_of_double_space, col_to_use
            )
        )
        cleaning = 1
        metric = "{}% (count {}) have double spaces".format(
            double_space_pc, number_of_double_space, col_to_use
        )

    info_df = {
        "column": col_to_use,
        "check_type": "Double spaces",
        "needs_cleaning": cleaning,
        "metric": metric,
    }
    info_df = pd.DataFrame(info_df, columns=info_df.keys(), index=[0])

    return info_df


def concat_key_check(dataset, col_to_use, concat_key):
    """
    Checks that for the specified concatenation key, a) the column is consistently concatenated and
    b) provides the count of the concat key for each value in the column.

    :param dataset: Dataset containing the specified (string) column
    :param col_to_use: User-specified string / categorical column
    :param concat_key: User-specified special character / string that is checked as a delimiter for col_to_use
    :return: Table containing metrics and cleaning suggestions related to the concat key and the column values
    """

    # create column that has count of concat key in each row for col_to_use
    concat_key_count = (
        dataset[col_to_use].astype("str").apply(lambda value: value.count(concat_key))
    )

    # check that the count is consistent, otherwise throw error
    count_uniques = len(concat_key_count.unique())

    if count_uniques == 1:

        if concat_key_count[0] == 0:
            print(
                'There are no occurrences of the concat key "{}".\n'.format(concat_key)
            )
            metric = "No concatenation detected"
        else:
            print(
                'Each value is properly concatenated with {} of the concat key "{}" specified.\n'.format(
                    concat_key_count[0], concat_key
                )
            )
            metric = "Option to break out the strings in this one column into multiple columns (separated by delimiter)"
        cleaning = 0

    else:
        print(
            'There are {} different counts of the concat key "{}" specified. Cleaning may be required.\n'.format(
                count_uniques, concat_key
            )
        )
        cleaning = 1
        metric = '{} different counts of the concat key "{}"'.format(
            count_uniques, concat_key
        )

    info_df = {
        "column": col_to_use,
        "check_type": "Proper concatenation",
        "needs_cleaning": cleaning,
        "metric": metric,
    }
    info_df = pd.DataFrame(info_df, columns=info_df.keys(), index=[0])

    return info_df


def check_string_col_traits(dataset, col_to_use, concat_key):
    """
    For a user-specified column, checks for inconsistent use of case, trailing and leading spaces, double spaces,
    and the presence / consistency of a concat key for a column with multiple values.

    :param dataset: Dataset containing the specified (string) column
    :param col_to_use: User-specified string / categorical column
    :param concat_key: User-specified special character / string that is checked as a delimiter for col_to_use
    :return: Table containing information via columns on the checks that were performed for col_to_use
    """

    # Get row count of full dataset
    dataset_len = len(dataset[col_to_use])

    # Perform checks on the specified string / categorical column
    case_df = consistent_use_of_case(dataset, col_to_use)
    trailing_leading_df = trailing_leading_spaces(dataset, col_to_use, dataset_len)
    double_space_df = check_double_spaces(dataset, col_to_use, dataset_len)
    concat_key_df = concat_key_check(dataset, col_to_use, concat_key)

    # Combine each check into a single table containing information on col_to_use
    col_to_use_traits_df = pd.concat(
        (case_df, trailing_leading_df, double_space_df, concat_key_df), sort=False
    )

    return col_to_use_traits_df


def view_inconsistent_cases(dataset, col_to_use, get_count=False):
    """
    If the user-specified col_to_use contains inconsistent case use as defined by function _consistent_use_of_case,
    this function allows the user to investigate all string values that when standardized are the same.

    :param dataset: Dataset containing the specified (string) column
    :param col_to_use: User-specified string / categorical column
    :param get_count: Boolean indicating whether to include a column containing the frequency of the value (which can
        be used to find the most common occurence of a value with inconsistent case)
    :return: Table containing the col_to_use variations where there are multiple capitalizations of the same value
    """

    # filter down original dataset to just key column and add column that standardizes case (uppercase)
    dataset_filtered = dataset[[col_to_use]].copy()
    dataset_filtered["uppercased_{}".format(col_to_use)] = (
        dataset[col_to_use].astype("str").apply(lambda value: value.upper())
    )

    if get_count:
        dataset_filtered["occurrences"] = dataset_filtered.groupby(col_to_use)[
            "uppercased_{}".format(col_to_use)
        ].transform("count")

    dataset_filtered.drop_duplicates(inplace=True)

    # get counts of the uppercased column values
    uppercased_df = dataset_filtered[["uppercased_{}".format(col_to_use)]].copy()
    uppercased_df["count"] = 1
    uppercased_df = (
        uppercased_df[["uppercased_{}".format(col_to_use), "count"]]
        .groupby("uppercased_{}".format(col_to_use))
        .agg("sum")
        .reset_index()
    )
    uppercased_df_indexed = uppercased_df.set_index("uppercased_{}".format(col_to_use))

    # join uppercase counts back to original filtered dataset
    dataset_filtered = dataset_filtered.join(
        uppercased_df_indexed, on=["uppercased_{}".format(col_to_use)], how="left"
    )

    dataset_filtered.sort_values(by="uppercased_{}".format(col_to_use), inplace=True)

    # add flag where there are duplicates
    dataset_filtered = dataset_filtered.loc[
        dataset_filtered["count"] != 1
    ].drop_duplicates()
    dataset_filtered.drop(columns="count", inplace=True)

    if get_count:
        dataset_filtered["occurrences"] = dataset_filtered["occurrences"].astype("int")

    return dataset_filtered


def string_col_diagnostics(dataset, string_cat_list, col_to_use, concat_key):
    """
    Performs string / categorical diagnostics on a single user-specified column.

    :param dataset: Dataset containing the user-specified string / categorical column
    :param string_cat_list: List of columns that have been identified with the dataset as string / categorical
    :param col_to_use: User-specified string / categorical column
    :param concat_key: User-specified special character / string that is checked as a delimiter for col_to_use
    :return: Same output as the function: break_out_value_types
    :return: Same output as the function check_string_col_traits
    :return: If there is inconsistent use of case, then it returns a pandas dataframe containing a mapping table of
        strings that map to the same upper-case string
    """

    # Check that specified column exists and is eligible for string based diagnostics
    assert (
        col_to_use in dataset.columns.tolist()
    ), "Specified column is not within uploaded dataset"
    assert (
        col_to_use in string_cat_list
    ), "Specified column is not a string/categorical column"

    # Label the distinct values in the specified column as numeric, string, or a mix of both
    temp_dataset, figure_list, output_dict_list = break_out_value_types(
        dataset, col_to_use
    )

    # Perform various checks (inconsistent use of case, trailing/leading spaces, double spaces, concat key usage)
    col_to_use_traits_df = check_string_col_traits(temp_dataset, col_to_use, concat_key)

    # If there is inconsistent case usage, use the view inconsistent usage function
    if (
        col_to_use_traits_df["needs_cleaning"]
        .loc[
            (col_to_use_traits_df["column"] == col_to_use)
            & (col_to_use_traits_df["check_type"] == "(In)consistent use of cases")
        ]
        .item()
        == 1
    ):
        inconsistent_case_df = view_inconsistent_cases(temp_dataset, col_to_use)
    else:
        inconsistent_case_df = None

    data_types = temp_dataset, figure_list, output_dict_list

    return data_types, col_to_use_traits_df, inconsistent_case_df


def explore_ngrams(data_series, min_n=1, max_n=2, min_df=1, max_terms=None, **kwargs):
    """
    This function makes use of sklearn's CountVectorizer to identify all or common combinations of n-words from a
    provided data series (list, pd.Series), and provides information about their frequencies. It also returns the
    trained vectorizer and the matrix with ngram counts by row.

    :param data_series: list or pd.Series containing the data for analysis
    :param min_n: minimum number of words in the n-gram. With max_n, makes up the ngrams_range argument of the
        CountVectorizer
    :param max_n: maximum number of words in the n-gram
    :param min_df: minimal document frequency of (# of cells with) the n-grams kept
    :param max_terms: Only track the top 'max' number of n-grams in terms of term frequency

    :return: pandas dataframe containing the frequency counts (Document count and Term count)
    :return: trained CountVectorizer object
    :return: matrix containing the transformed output (i.e. counts of the terms by document)
    """

    # instantiate the CountVectorizer with the options
    vectorizer = CountVectorizer(
        ngram_range=(min_n, max_n), min_df=min_df, max_features=max_terms, **kwargs
    )
    X = vectorizer.fit_transform(data_series.apply(lambda x: str(x)))

    ngram_summary = pd.DataFrame({"n-grams": vectorizer.get_feature_names()})
    ngram_summary["N"] = ngram_summary["n-grams"].apply(lambda x: len(str(x).split()))
    ngram_summary["Document Frequency"] = (X >= 1).sum(axis=0).tolist()[0]
    ngram_summary["Term Frequency"] = X.sum(axis=0).tolist()[0]

    return (
        ngram_summary.sort_values(by="Document Frequency", ascending=False),
        vectorizer,
        X,
    )
