import pandas as pd
import numpy as np
from typing import Optional


def get_join_statistics(
    left_dataset, right_dataset, join_type, join_col, output_join: Optional[bool] = True
):
    """
    This function calculates key statistics on the output of a join between two datasets. This join is specified
    by the user using the parameter join_type. If desired, the user can output the joined table. A few asserts are
    put in place to ensure that necessary conditions for the join are met. For more information on the pandas join
    function, please refer to link here:
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html

    :param left_dataset: dataframe that at least contains the join_col
    :param right_dataset: dataframe with no overlapping columns with left_dataset except for join_col
    :param join_type: Acceptable values are {'left', 'right', 'outer', 'inner'}. Refer to link for info
    :param join_col: column that exists in both columns and is used as the key to join the two datasets together
    :param output_join: binary True/False for whether or not the joined dataset is output

    :return: Table containing statistics about the join between the two datasets
    :return: If output_join is True, returns the merged dataset
    """

    # If requiring only the statistics, then other columns are not relevant
    if output_join is False:
        if isinstance(join_col, list):
            left_dataset = left_dataset[join_col].copy()
            right_dataset = right_dataset[join_col].copy()
        else:
            left_dataset = left_dataset[[join_col]].copy()
            right_dataset = right_dataset[[join_col]].copy()
    else:
        left_dataset = left_dataset.copy()
        right_dataset = right_dataset.copy()

    ### Check that a "many to many" join is not attempted; if so, stop the code before the join
    # Get frequency count of left dataset
    left_dataset_freq = (
        left_dataset.groupby(join_col)
        .size()
        .reset_index()
        .rename(columns={0: "left_count"})
    )

    # Get frequency count of right dataset
    right_dataset_freq = (
        right_dataset.groupby(join_col)
        .size()
        .reset_index()
        .rename(columns={0: "right_count"})
    )

    # Outer join the two together
    both_freq = left_dataset_freq.merge(right_dataset_freq, how="outer")

    # Create flag to check whether there is a circumstance where left_count and right_count are both >1
    both_freq["many_to_many"] = np.where(
        ((both_freq["left_count"] > 1) & (both_freq["right_count"] > 1)), 1, 0
    )

    # Create assert that will stop the code if there is a circumstance where left_count and right_count are both >1
    assert (
        both_freq["many_to_many"] == 0
    ).all(), "Many to many check failed. Join attempt would create cartesian product of columns"

    ### Perform initial join to gt key join statistics as an output and return the df if output_join==True
    # Add left and right dataset flags for statistics
    # Add left and right dataset flags for statistics
    left_dataset["_left_flag"] = 1
    right_dataset["_right_flag"] = 1

    # Perform the join
    merged_dataset = left_dataset.merge(right_dataset, how=join_type)

    # Get overall values to use for statistics (from the merged dataset)
    merged_dataset_len = len(merged_dataset)
    statistics_dict = []

    # Get left only statistics
    left_only_count = len(
        merged_dataset.loc[
            (merged_dataset["_left_flag"] == 1)
            & (merged_dataset["_right_flag"].isnull())
        ]
    )
    left_only_pc = round((left_only_count / merged_dataset_len * 100), 2)
    original_left = left_dataset.shape[0]
    original_left_unique = left_dataset[join_col].drop_duplicates().shape[0]
    left_only_unique = len(
        merged_dataset[join_col]
        .loc[
            (merged_dataset["_left_flag"] == 1)
            & (merged_dataset["_right_flag"].isnull())
        ]
        .drop_duplicates()
    )

    # Convert left only into dictionary entries for overall statistics dataframe
    left_only_dict = {
        "statistic": "Left only rows",
        "number_obs": left_only_count,
        "percent_of_merged": left_only_pc,
        "percent_of_original": round(100 * left_only_count / original_left, 2),
    }
    left_only_dict = pd.DataFrame(
        left_only_dict, columns=left_only_dict.keys(), index=[0]
    )
    statistics_dict.append(left_only_dict)

    left_only_unique_dict = {
        "statistic": "Left only unique keys",
        "number_obs": left_only_unique,
        "percent_of_merged": "N/A",
        "percent_of_original": round(100 * left_only_unique / original_left_unique, 2),
    }
    left_only_unique_dict = pd.DataFrame(
        left_only_unique_dict, columns=left_only_unique_dict.keys(), index=[0]
    )
    statistics_dict.append(left_only_unique_dict)

    # Get matched statistics
    matched_count = len(
        merged_dataset.loc[
            (merged_dataset["_left_flag"] == 1) & (merged_dataset["_right_flag"] == 1)
        ]
    )
    matched_pc = round((matched_count / merged_dataset_len * 100), 2)
    matched_unique = len(
        merged_dataset[join_col]
        .loc[(merged_dataset["_left_flag"] == 1) & (merged_dataset["_right_flag"] == 1)]
        .drop_duplicates()
    )

    # Convert matched statistics into dictionary entries for overall statistics dataframe
    matched_dict = {
        "statistic": "Matched rows",
        "number_obs": matched_count,
        "percent_of_merged": matched_pc,
        "percent_of_original": "N/A",
    }
    matched_dict = pd.DataFrame(matched_dict, columns=matched_dict.keys(), index=[0])
    statistics_dict.append(matched_dict)

    matched_unique_dict = {
        "statistic": "Matched unique keys",
        "number_obs": matched_unique,
        "percent_of_merged": "N/A",
        "percent_of_original": "N/A",
    }
    matched_unique_dict = pd.DataFrame(
        matched_unique_dict, columns=matched_unique_dict.keys(), index=[0]
    )
    statistics_dict.append(matched_unique_dict)

    # Get right only statistics
    right_only_count = len(
        merged_dataset.loc[
            (merged_dataset["_left_flag"].isnull())
            & (merged_dataset["_right_flag"] == 1)
        ]
    )
    right_only_pc = round((right_only_count / merged_dataset_len * 100), 2)
    original_right = right_dataset.shape[0]
    original_right_unique = right_dataset[join_col].drop_duplicates().shape[0]
    right_only_unique = len(
        merged_dataset.loc[
            (merged_dataset["_left_flag"].isnull())
            & (merged_dataset["_right_flag"] == 1)
        ].drop_duplicates()
    )

    # Convert right only statistics into dictionary entries for overall statistics dataframe
    right_only_dict = {
        "statistic": "Right only rows",
        "number_obs": right_only_count,
        "percent_of_merged": right_only_pc,
        "percent_of_original": round(100 * right_only_count / original_right, 2),
    }
    right_only_dict = pd.DataFrame(
        right_only_dict, columns=right_only_dict.keys(), index=[0]
    )
    statistics_dict.append(right_only_dict)

    right_unique_dict = {
        "statistic": "Right only unique keys",
        "number_obs": right_only_unique,
        "percent_of_merged": "N/A",
        "percent_of_original": round(
            100 * right_only_unique / original_right_unique, 2
        ),
    }
    right_unique_dict = pd.DataFrame(
        right_unique_dict, columns=right_unique_dict.keys(), index=[0]
    )
    statistics_dict.append(right_unique_dict)

    # Remove added columns
    merged_dataset = merged_dataset.drop(columns=["_left_flag", "_right_flag"])

    # Create and print overall statistics
    statistics_df = pd.concat(statistics_dict, sort=False).reset_index(drop=True)

    # Check whether dataset should be returned
    if output_join:
        pass
    else:
        merged_dataset = None

    return statistics_df, merged_dataset
