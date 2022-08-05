import pandas as pd
import numpy as np
import re
from .univariate import univariate_summary_categorical
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Optional, Tuple, Literal, Dict
import matplotlib.pyplot as plt


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


def check_string_col(dataset, col_to_use):
    """
    For a user-specified column, checks for inconsistent use of case, trailing and leading spaces, double spaces,
    and the presence / consistency of a concat key for a column with multiple values.

    :param dataset: Dataset containing the specified (string) column
    :param col_to_use: User-specified string / categorical column
    :return: Table containing information via columns on the checks that were performed for col_to_use
    """

    # Get row count of full dataset
    dataset_len = len(dataset[col_to_use])

    # Perform checks on the specified string / categorical column
    case_df = consistent_use_of_case(dataset, col_to_use)
    trailing_leading_df = trailing_leading_spaces(dataset, col_to_use, dataset_len)
    double_space_df = check_double_spaces(dataset, col_to_use, dataset_len)

    # Combine each check into a single table containing information on col_to_use
    col_to_use_traits_df = pd.concat(
        [case_df, trailing_leading_df, double_space_df], sort=False
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

    ngram_summary = pd.DataFrame({"n-grams": vectorizer.get_feature_names_out()})
    ngram_summary["N"] = ngram_summary["n-grams"].apply(lambda x: len(str(x).split()))
    ngram_summary["Document Frequency"] = (X >= 1).sum(axis=0).tolist()[0]
    ngram_summary["Term Frequency"] = X.sum(axis=0).tolist()[0]

    return (
        ngram_summary.sort_values(by="Document Frequency", ascending=False),
        vectorizer,
        X,
    )


def two_way_proportions(input_data, x_var,y_var,normalize="row",dropna=False,
        figsize: Tuple[float, float] = (10, 6),
        cmap=None,
        annot: bool = True,
        fmt: str = ".1f"
    ):
    """
    :param data: Pandas dataframe
    :param x: Variable to populate the columns
    :param y: Variable to populate the rows
    :param normalize: Should "row" or "column" sum up to 1? 
    :param figsize: Figure size for matplotlib
    :param cmap: Colormap to use (if not default)
    :param annot: Whether to annotate heatmap with correlation values
    :param fmt: Format specifier for annotated correlation values
    """
    
    if normalize == "row":
        table = pd.crosstab(input_data[y_var], input_data[x_var], normalize='index')
    elif normalize == "columns":
        table = pd.crosstab(input_data[y_var], input_data[x_var], normalize='column')
    else:
        raise ValueError("The 'normalize' argument must be either 'row' or 'column'")
        
    title = "Two-way frequency table (proportion by {})".format(normalize)
    
    fig = _plot_table(table, figsize, title, cmap, annot, fmt)
    
    return fig, table
    
    
def _plot_table(table, figsize, title, cmap, annot, fmt):
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    img = ax.imshow(table, interpolation='none', aspect='auto', vmax=1, vmin=-1, cmap=cmap)
    if annot:
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                if not np.isnan(table.iloc[i, j]):
                    ax.text(j, i, f"{100*table.iloc[i, j]:{fmt}}%", ha="center", va="center")
    
    ax.set_xticks(np.arange(-.5, table.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, table.shape[0], 1), minor=True)
    ax.grid(which='minor', color='b', linestyle='-', linewidth=2)
    plt.colorbar(img, ax=ax)

    x_label_list = table.columns.values
    y_label_list = table.index.values
    
    ax.set_xticks(range(len(x_label_list)))
    ax.set_xticklabels(x_label_list, rotation=45, ha="right")
    ax.set_yticks(range(len(y_label_list)))
    ax.set_yticklabels(y_label_list)
    ax.set_title(title)
    
    return fig
