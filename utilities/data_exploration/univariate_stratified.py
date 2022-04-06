import pandas as pd
import numpy as np
import matplotlib  # https://stackoverflow.com/questions/38090455/what-is-the-simplest-way-to-make-matplotlib-in-osx-work-in-a-virtual-environment
import os

if os.name == "nt":
    matplotlib.use("Agg", warn=False)
import matplotlib.pyplot as plt
import seaborn as sns
import re
from ..general_utilities import diff_month_pd


def calculate_mean_by_strata(dataset_used, var_names, strata, prefix=None):
    """
    Calculates any requested percentiles by unique values of a strata.

    :param dataset_used: a single Pandas DataFrame containing the date identifier, and relevant variables to sum
    :param var_names: a list of strings containing the names of the variables for which to calculate means
    :param strata: a single string with the name of the strata identifier variable
    :param prefix: Optional prefix (default: mean) for the new variables to be created: {prefix}_{variable}
    :return: new Pandas DataFrame with columns representing mean by strata
    """

    if prefix is None:
        prefix = "mean"
    results = dataset_used.groupby([strata])[var_names].mean()
    available_varnames = results.columns.values[:]
    results = results.rename(
        columns=dict(
            zip(
                available_varnames,
                [
                    "{}_{}".format(prefix, available_varname)
                    for available_varname in available_varnames
                ],
            )
        )
    )

    return results


def plot_mean_by_strata(dataset_used, var_names, strata):
    """
    Plots specific quantiles of a variable by strata (e.g. time).

    :param dataset_used: a single Pandas DataFrame containing the date identifier and relevant variables
    :param var_names: a list of strings containing the names of the variables for which to plot distribution
    :param strata: a single string with the name of the by variable to be summarized by
    :return: Pandas DataFrame with quantiles for each variable by date. Also plots the quantiles by date
    """

    # Request quantiles from helper function
    results = calculate_mean_by_strata(dataset_used, var_names, strata).reset_index()

    # Sort the series
    results = results.sort_values(by=[strata])

    plot_num = 0
    fig_list = []
    for var_name in var_names:
        fig = plt.figure(figsize=(10, 10))

        plot_num = plot_num + 1
        plt.subplot(len(var_names), 1, plot_num)
        plt.plot(
            range(results.shape[0]),
            results["mean_{}".format(var_name)],
            label="mean_{}".format(var_name),
        )
        x_axis_intervals = [
            int(val)
            for val in np.arange(
                0, results.shape[0] - 1, np.ceil(results.shape[0] / 10)
            )
        ]
        plt.xticks(
            x_axis_intervals,
            [results[strata][val] for val in x_axis_intervals],
            size="small",
        )
        plt.ylabel("Value")
        plt.title("Average {} over time".format(var_name))

        fig_list.append({"variable": var_name, "figure": fig})

    return results, fig_list


def calculate_quantile_by_strata(
    dataset_used, var_names, strata, probability, prefix=None
):
    """
    Calculates any requested percentiles by unique values of a strata.

    :param dataset_used: a single Pandas DataFrame containing the date identifier, and relevant variables to sum
    :param var_names: a list of strings containing the names of the variables for which to calculate quantiles
    :param strata: a single string with the name of the strata identifier variable
    :param probability: a number between 0 and 1 containing the percentile (e.g. 0.1 means 10%)
    :param prefix: Optional prefix (default: perc_[prob]) for the new variables to be created: {prefix}{variable}
    :return: new Pandas DataFrame with columns representing quantile by strata
    """
    if prefix is None:
        prefix = "perc_{}_".format(str(round(100 * probability)))

    results = dataset_used.groupby([strata])[var_names].quantile(q=probability)
    available_varnames = results.columns.values[:]
    results = results.rename(
        columns=dict(
            zip(
                available_varnames,
                [
                    "{}{}".format(prefix, available_varname)
                    for available_varname in available_varnames
                ],
            )
        )
    )

    return results


def plot_quantiles_by_strata(dataset_used, var_names, strata, probs=None):
    """
    Plots specific quantiles of a variable by strata (e.g. time).

    :param dataset_used: a single Pandas DataFrame containing the date identifier and relevant variables
    :param var_names: a list of strings containing the names of the variables for which to plot distribution
    :param strata: a single string with the name of the by variable to be summarized by
    :param probs: a list of doubles between 0 and 1 containing the percentiles. Default: [0.05, 0.25, 0.50, 0.75, 0.95]
    :return: Pandas DataFrame with quantiles for each variable by date. Also plots the quantiles by date
    """

    if probs is None:
        probs = [0.05, 0.25, 0.50, 0.75, 0.95]

    # Request quantiles from helper function
    results = pd.concat(
        [
            calculate_quantile_by_strata(dataset_used, var_names, strata, prob)
            for prob in probs
        ],
        axis=1,
    ).reset_index()
    # Sort the series
    results = results.sort_values(by=[strata])

    plot_num = 0
    fig_list = []
    for var_name in var_names:
        fig = plt.figure(figsize=(10, 10))

        plot_num = plot_num + 1
        plt.subplot(len(var_names), 1, plot_num)
        for prob in probs:
            column_name = "perc_{}_{}".format(str(round(100 * prob)), var_name)
            plt.plot(
                range(results.shape[0]),
                results[column_name],
                label=column_name,
                color=(
                    max(prob, 1 - prob) / 1.5,
                    max(prob, 1 - prob) / 1.5,
                    max(prob, 1 - prob) / 1.5,
                ),
            )
        x_axis_intervals = [
            int(val)
            for val in np.arange(
                0, results.shape[0] - 1, np.ceil(results.shape[0] / 10)
            )
        ]
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.xticks(
            x_axis_intervals,
            [results[strata][val] for val in x_axis_intervals],
            size="small",
        )
        plt.ylabel("Value")
        plt.title("{} over time".format(var_name))

        fig_list.append({"variable": var_name, "figure": fig})

    return results, fig_list


def count_distinct_by_strata(dataset_used, var_names, strata, prefix=None):
    """
    Calculates the number of distinct values of a variable by strata.

    :param dataset_used: a single Pandas DataFrame containing the subject identifier, and relevant
        variables to identify number of distinct values per strata
    :param var_names: a list of strings containing the names of the variables for which to count distincts
    :param strata: a single string with the name of the strata variable
    :param prefix: Optional prefix (default: "num_distinct") for the new variables to be created: {prefix}_{variable}
    :return: new Pandas DataFrame with columns representing the number of distinct values by strata
    """

    if prefix is None:
        prefix = "num_distinct"

    results = (
        dataset_used.groupby([strata])[var_names]
        .nunique()
        .rename(
            columns=dict(
                zip(
                    var_names,
                    ["{}_{}".format(prefix, var_name) for var_name in var_names],
                )
            )
        )
    )

    # Old Pandas code
    # results = dataset_used.groupby([strata])[var_names]. \
    #     filter(lambda g: (g.apply(pd.Series.nunique) > 1).any()). \
    #     rename(columns=dict(zip(var_names, ["{}{}".format(prefix, var_name) for var_name in var_names])))

    return results


def distinct_values(dataset_used, var_names, strata, summary_only=True):
    """
    Checks whether variables only attain a single value for each strata, and if not, how often it attains more than a
    single value and how many values it attains. This functionality is primarily used to check whether presumably
    constant variables for a strata (e.g. loan) actually are constant.

    :param dataset_used: a single pandas DataFrame containing the strata variable and relevant variables to check
    :param var_names: a list of strings containing the names of the variables for which to check for constancy
    :param strata: a single string with the name of the subject identifier variable
    :param summary_only: a single boolean indicating whether the function should return only the summary results,
        or return the results per strata identifier
    :return: a tuple with a small Pandas DataFrame providing a summary of the number of strata with unique, non-unique
        and only NaN value for each variable, and detailed dataframe with results by strata
    """

    # Count the number of uniques
    number_of_unique = dataset_used.groupby([strata])[var_names].nunique()

    # Old Python code
    # number_of_unique = dataset_used.groupby([strata])[var_names].\
    #     filter(lambda g: (g.apply(pd.Series.nunique) > 1).any())

    # Summarize the number of subjects with NaN, Unique value, or Non-Unique value
    number_non_unique = (number_of_unique > 1).sum()
    number_unique = (number_of_unique == 1).sum()
    number_total = len(number_of_unique.index)
    number_NaN = (number_of_unique == 0).sum()
    number_max = number_of_unique.max()

    # For the non-unique, what is the average number of occurences?
    non_unique_mean = np.ma.average(
        number_of_unique, weights=(number_of_unique > 1), axis=0
    )

    summary = pd.DataFrame(
        {
            "number_non_unique": number_non_unique,
            "number_unique": number_unique,
            "number_total": number_total,
            "number_max": number_max,
            "number_NaN": number_NaN,
            "non_unique_mean": non_unique_mean,
        }
    )

    if not summary_only:
        # Edit the column headers for the number of unique dataframe
        number_of_unique.columns = [
            "num_unique__{}".format(s) for s in number_of_unique.columns.values
        ]
        # In addition to the number of unique values, also count the total number of unique values
        total_number = dataset_used.groupby([strata])[var_names].count()
        total_number.columns = [
            "total_obs__{}".format(s) for s in total_number.columns.values
        ]
        detailed = pd.concat([number_of_unique, total_number], axis=1)
    else:
        detailed = None

    return summary, detailed


def calculate_miss_by_strata_individual(
    dataset_used, var_names, strata, ratio=False, prefix=None
):
    """
    Calculates number of missing observations by strata (e.g. date) and generates a list of figures of line plots.

    :param dataset_used: a single Pandas DataFrame containing the strata, and relevant variables for analysis
    :param var_names: a list of string containing the name of the variables for which to count NaNs
    :param strata: a single string with the name of the strata variable
    :param ratio: boolean for whether to generate ratio instead of absolute counts
    :param prefix: Optional prefix (default: miss) for the new variables to be created: {prefix}_{variable}
    :return: new Pandas DataFrame summarizing the number of missings for each variable by strata
    """

    # Calculate the number of missings for a variable in a strata
    if ratio:
        summary = (
            dataset_used.groupby([strata])[var_names]
            .apply(lambda x: x.isnull().sum() / x.shape[0])
            .reset_index()
            .rename(
                columns=dict(
                    zip(
                        var_names,
                        ["{}_{}".format(prefix, var_name) for var_name in var_names],
                    )
                )
            )
        )
        title = "% missing for {} over {}"
    else:
        summary = (
            dataset_used.groupby([strata])[var_names]
            .apply(lambda x: x.isnull().sum())
            .reset_index()
            .rename(
                columns=dict(
                    zip(
                        var_names,
                        ["{}_{}".format(prefix, var_name) for var_name in var_names],
                    )
                )
            )
        )
        title = "# missing for {} over {}"

    figure_list = []
    for var_name in var_names:
        fig = plt.figure()
        fig.axes[0] = summary["{}_{}".format(prefix, var_name)].plot(
            figsize=(10, 5), linewidth=4, title=title.format(var_name, strata), rot=90
        )
        fig.axes[0].set_ylim(ymin=0)

        figure_list.append(fig)

    return figure_list


def calculate_non_miss_by_strata_individual(
    dataset_used, var_names, strata, ratio=False, prefix=None
):
    """
    Calculates number of non-missing observations by strata (e.g. date) and generates a list of figures of line plots.

    :param dataset_used: a single Pandas DataFrame containing the strata, and relevant variables for analysis
    :param var_names: a list of string containing the name of the variables for which to count NaNs
    :param strata: a single string with the name of the strata variable
    :param ratio: boolean for whether to generate ratio instead of absolute counts
    :param prefix: Optional prefix (default: miss) for the new variables to be created: {prefix}_{variable}
    :return: new Pandas DataFrame summarizing the number of non-missings for each variable by strata
    """

    # Calculate the number of missings for a variable in a strata
    if ratio:
        summary = (
            dataset_used.groupby([strata])[var_names]
            .apply(lambda x: x.count() / x.shape[0])
            .reset_index()
            .rename(
                columns=dict(
                    zip(
                        var_names,
                        ["{}_{}".format(prefix, var_name) for var_name in var_names],
                    )
                )
            )
        )
        title = "% non-missing for {} over {}"
    else:
        summary = (
            dataset_used.groupby([strata])[var_names]
            .apply(lambda x: x.count())
            .reset_index()
            .rename(
                columns=dict(
                    zip(
                        var_names,
                        ["{}_{}".format(prefix, var_name) for var_name in var_names],
                    )
                )
            )
        )
        title = "# non-missing for {} over {}"

    figure_list = []
    for var_name in var_names:
        fig = plt.figure()
        fig.axes[0] = summary["{}_{}".format(prefix, var_name)].plot(
            figsize=(10, 5), linewidth=4, title=title.format(var_name, strata), rot=90
        )
        fig.axes[0].set_ylim(ymin=0)

        figure_list.append(fig)

    return figure_list


def calculate_miss_ratio_by_strata_individual(dataset_used, var_name, strata):
    """
    Generate a table with missings, non-missings, missing ratio for a variable.

    :param dataset_used: a single Pandas DataFrame containing the strata variable and the variable for
        which to identify missings
    :param var_name: name of variable for which to count NaNs
    :param strata: a single string with the name of the strata variable
    :return: new Pandas DataFrame summarizing the number of missings for each variable by strata
    """

    # Calculate the number of missings for each variable
    summary = (
        dataset_used.groupby([strata])
        .apply(
            lambda x: pd.Series(
                [
                    x[var_name].isnull().sum(),
                    x[var_name].count(),
                    x[var_name].shape[0],
                    x[var_name].isnull().sum() / x.shape[0],
                ]
            )
        )
        .rename(columns={0: "miss", 1: "non_miss", 2: "all", 3: "ratio"})
    )

    return summary


def calculate_miss_by_strata(dataset_used, var_names, strata, prefix=None):
    """
    Calculates number of missing observations by strata and plots a heatmap.

    :param dataset_used: a single Pandas DataFrame containing the date identifier, and relevant variables to
        identify number of missings
    :param var_names: a list of strings containing the names of the variables for which to count leading NaNs
    :param strata: a single string with the name of the date identifier variable
    :param prefix: Optional prefix (default: "miss") for the new variables to be created: {prefix}_{variable}
    :return: new Pandas DataFrame summarizing the number of missings for each variable by date
    """
    if prefix is None:
        prefix = "miss"

    # Calculate the number of missings for each variable
    summary = (
        dataset_used.groupby([strata])[var_names]
        .apply(lambda x: x.isnull().sum() / x.shape[0])
        .reset_index()
        .rename(
            columns=dict(
                zip(
                    var_names,
                    ["{}_{}".format(prefix, var_name) for var_name in var_names],
                )
            )
        )
    )

    tick_labels = summary[strata].tolist()

    # Plot data using seaborn
    fig = plt.figure(figsize=(15, 0.15 * len(tick_labels)))
    plt.tick_params(axis="both", which="major", labelsize=8)
    sns.heatmap(
        summary[["miss_{}".format(var_name) for var_name in var_names]],
        cmap="Oranges",
        vmin=0,
        vmax=1,
        yticklabels=tick_labels,
    )
    plt.close()
    return fig, summary


def calculate_mean_std_by_strata(dataset_used, var_names, strata, std=None):
    """
    Calculates and plots the mean and standard deviation of variables over strata (e.g. time).

    :param dataset_used: a single Pandas DataFrame with the date identifier, and relevant variables to summarize
    :param var_names: list of strings containing the names of the variables for which to summarize
    :param strata a single string with the name of the by variable to be summarized by (e.g. time)
    :param std: List of doubles containing the multiples of standard deviation to be plotted
    :return: Pandas DataFrame with new columns containing the mean and standard deviation for each variable by strata
    """

    if std is None:
        std = [1, 2]

    summary = pd.concat(
        [
            dataset_used.groupby([strata])[var_names]
            .mean()
            .rename(
                columns=dict(
                    zip(
                        var_names,
                        ["mean_{}".format(var_name) for var_name in var_names],
                    )
                )
            ),
            dataset_used.groupby([strata])[var_names]
            .std()
            .rename(
                columns=dict(
                    zip(
                        var_names,
                        ["stddev_{}".format(var_name) for var_name in var_names],
                    )
                )
            ),
        ],
        axis=1,
    ).reset_index()

    # Calculate the mean +/ stdevs
    date_ids = summary[strata].tolist()
    stdevs = [-1 * s for s in std] + [0] + std

    plot_num = 0
    figure_list = []
    for var_name in var_names:
        fig = plt.figure(figsize=(10, 10))
        plot_num = plot_num + 1
        plt.subplot(len(var_names), 1, plot_num)
        for stdev in stdevs:
            new_name = "{}_{}".format(
                var_name,
                re.sub(
                    "plus_-",
                    "minus_",
                    re.sub(r"\.", "_", "plus_{}_sd".format(str(stdev))),
                ),
            )
            summary[new_name] = (
                summary["mean_{}".format(var_name)]
                + summary["stddev_{}".format(var_name)] * stdev
            )
            plt.plot(
                range(len(date_ids)),
                summary[[new_name]],
                label=new_name,
                color=(
                    abs(stdev) / (1.5 * max(stdevs)),
                    abs(stdev) / (1.5 * max(stdevs)),
                    abs(stdev) / (1.5 * max(stdevs)),
                ),
            )
        x_axis_intervals = [
            int(val)
            for val in np.arange(0, len(date_ids) - 1, np.ceil(len(date_ids) / 10))
        ]
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.xticks(
            x_axis_intervals, [date_ids[val] for val in x_axis_intervals], size="small"
        )
        plt.ylabel("Value")
        plt.title("{} over time".format(var_name))
        figure_list.append(fig)
        plt.close()
    return figure_list, summary


def continuous_dist_by_strata(
    dataset_used, var_name, strata, num_buckets=5, normalize=False
):
    """
    Plot the distribution for a continuous variable over time (relative to long-term distribution). The dataset is
    bucketed by quantile buckets, following which the number of observations by strata are calculated and plotted as a
    stacked bar chart.

    :param dataset_used: a single Pandas DataFrame containing the date identifier and relevant variable
    :param var_name: a strings containing the name of the variable for which to plot their distribution
    :param strata: a single string with the name of the by variable to be summarized by (e.g. time)
    :param num_buckets: a double representing the number of buckets to split the distribution
    :param normalize: a single boolean indicating whether the stacked bar charts should be normalized to 100%
    :return: tuple with figure and plotted data
    """

    if num_buckets < 2:
        raise ValueError("num_buckets must be at least 2")
    # Get quantiles
    probs = np.linspace(0.0, 1.0, 1 + int(num_buckets)).tolist()
    # For each variable, merge in the quantiles
    dataset_slim = dataset_used[[strata, var_name]].copy()
    dataset_slim["quantile_cuts"] = pd.qcut(
        dataset_slim[var_name], q=probs, duplicates="raise"
    )

    # Summarize the results
    summary = (
        dataset_slim.groupby([strata, "quantile_cuts"])[var_name]
        .count()
        .unstack("quantile_cuts")
        .fillna(0)
    )

    if normalize:
        row_sum = summary.sum(axis=1)
        summary = summary.apply(lambda x: x / row_sum, axis=0)

    summary.plot(
        kind="barh",
        stacked=True,
        figsize=(30 / 2.54, summary.shape[0] / (2 * 2.54)),
        width=0.8,
    )
    ax = plt.subplot(111)
    plt.title("{} distribution over time".format(var_name))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig = ax.get_figure()
    plt.close()
    return fig, summary


def categorical_dist_by_strata(dataset_used, var_name, strata, normalize=False):
    """
    Plot the distribution for a categorical variable over time (relative to long-term distribution).
    The dataset is bucketed by unique values of the categorical variable, following which the number of observations
    by strata are calculated and plotted as a stacked bar chart.

    :param dataset_used: a single PySpark DataFrame containing the date identifier and relevant variable
    :param var_name: a strings containing the name of the variable for which to plot their distribution
    :param strata: a single string with the name of the by variable to be summarized by (e.g. time)
    :param normalize: a single boolean indicating whether the stacked bar charts should be normalized to 100%
    :return: tuple with figure and plotted data
    """

    summary = dataset_used.groupby([strata, var_name]).size().reset_index(name="count")
    summary = summary.sort_values([strata])
    summary = summary.pivot(index=strata, columns=var_name, values="count").fillna(0)

    if normalize:
        column_variables = summary.columns.values
        summary = summary.assign(__SUM__=summary.sum(axis=1))
        old_summary = summary
        summary = pd.concat(
            [
                old_summary[col_var] / old_summary["__SUM__"]
                for col_var in column_variables
            ],
            axis=1,
        )
        summary.columns = column_variables

    date_ids = summary.index.values

    summary_array = summary.T.as_matrix()
    bottom = np.cumsum(summary_array, axis=0)

    fig = plt.figure(figsize=(30, len(date_ids) / 2))
    ind = np.arange(len(date_ids))
    plt.barh(ind, summary_array[0])
    for j in range(1, len(summary.columns)):
        plt.barh(ind, summary_array[j], left=bottom[j - 1])
        plt.gca().invert_yaxis()
    y_axis_intervals = list(np.arange(0, len(date_ids)))
    plt.yticks(
        y_axis_intervals, [date_ids[val] for val in y_axis_intervals], size="small"
    )
    plt.title("{} distribution over time".format(var_name))
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.gca().legend(summary.columns, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.close()

    return fig, summary


def add_cumulative_sum_by_subject(
    frame, var_names, strata, prefix="cumsum", na_fill=None
):
    """
    Calculates cumulative sum of variables by strata as new columns in a Pandas DataFrame.

    :param frame: a single sorted Pandas DataFrame containing the subject identifier, and relevant variables to
        calculate function sum. Within each strata, this needs to be sorted by time
        (for the cumulative sum to work properly)
    :param strata: a single string with the name of the strata variable
    :param var_names: a list of strings containing the names of the variables for which to calculate the cumulative sum
    :param prefix: Prefix (default: cumsum) for the new variables to be created: {prefix}_{variable}
    :return: Pandas DataFrame with new columns containing the cumulative sum with updated names
    """

    # Check existence
    for var_name in var_names:
        assert var_name in frame.columns, "{feat} not in data frame".format(
            feat=var_name
        )

    if na_fill is None:
        cumsums = frame.groupby(strata)[var_names].cumsum()
    else:
        cumsums = frame.groupby(strata)[var_names].fillna(na_fill).cumsum()

    new_names = ["{pf}_{var}".format(pf=prefix, var=var_name) for var_name in var_names]
    cumsums.rename(columns=dict(zip(var_names, new_names)), inplace=True)

    return pd.concat([frame, cumsums], axis=1)


def add_cumulative_prod_by_subject(
    frame, var_names, strata, prefix="cumprod", na_fill=None
):
    """
    Calculates cumulative product of variables as new columns in a Pandas DataFrame.

    :param frame: a single sorted Pandas DataFrame containing the subject identifier, and relevant variables to
        calculate function product. Within each strata, this needs to be sorted by time
        (for the cumulative product to work properly)
    :param strata: a single string with the name of the subject identifier variable
    :param var_names: a list of strings containing the names of the variables for which to calculate the cumulative sum
    :param prefix: Prefix (default: cumprod) for the new variables to be created: {prefix}_{variable}
    :return: Pandas DataFrame with new columns containing the cumulative prod with updated names
    """

    # Check existence
    for var_name in var_names:
        assert var_name in frame.columns, "{feat} not in data frame".format(
            feat=var_name
        )

    if na_fill is None:
        cumprods = frame.groupby(strata)[var_names].cumprod()
    else:
        cumprods = frame.groupby(strata)[var_names].fillna(na_fill).cumprod()

    new_names = ["{pf}_{var}".format(pf=prefix, var=var_name) for var_name in var_names]
    cumprods.rename(columns=dict(zip(var_names, new_names)), inplace=True)

    return pd.concat([frame, cumprods], axis=1)


def leading_nan(dataset_used, var_name, strata):
    """
    Checks the number of leading NaN for a specific variable. It is called by the function variable_leading_nan
    once per variable, if the analysis is needed for multiple variables.

    :param dataset_used: a single pandas DataFrame containing the strata variable and relevant variable to check.
        Within each strata, this needs to be sorted by time (for the cumulative product to work properly)
    :param var_name: the name of the variable for which to check for leading NaNs.
    :param strata: a single string with the name of the subject identifier variable
    :return: A pandas dataframe containing the number of leading NaNs for each subject
    """

    dataset_slim = dataset_used[[strata, var_name]].copy()
    dataset_slim = dataset_slim.assign(not_null=lambda x: ~np.isnan(x[var_name]) * 1)
    dataset_slim = dataset_slim.assign(
        not_null_cumsum=dataset_slim.groupby([strata])["not_null"].cumsum()
    )
    dataset_slim = dataset_slim.assign(
        initial_null=1 * (dataset_slim["not_null_cumsum"] == 0)
    )

    temp_results = (
        dataset_slim.groupby([strata])["initial_null"]
        .sum()
        .reset_index()
        .set_index(strata)
    )
    temp_results = temp_results.rename(columns={"initial_null": var_name})

    return temp_results


def variable_leading_nan(dataset_used, var_names, strata):
    """
    Checks the number of leading NaN that variables have by strata (e.g. loan).

    :param dataset_used: a single pandas DataFrame containing the strata variable and relevant variables to check.
        Within each strata, this needs to be sorted by time (for the cumulative product to work properly)
    :param var_names: a list of strings containing the names of the variables for which to check for leading NaNs
    :param strata: a single string with the name of the strata variable
    :return: a tuple containing a single pandas DataFrame summarizing the number of leading NaN for each variable, and
        a detailed dataframe containing the count of leading NaN for each variable by each subject
    """

    dataset_slim = dataset_used[[strata] + var_names].copy()
    results = pd.concat(
        [leading_nan(dataset_slim, var_name, strata) for var_name in var_names], axis=1
    )

    # Add in the length
    results = results.assign(length=dataset_slim.groupby([strata]).size())

    # Summarize
    summary = results.mean()

    return summary, results


def large_jumps(dataset_used, var_names, strata, jump_type, jump_size):
    """
    Checks for the presence of large jumps in value of variables by strata over time (e.g. loan).

    :param dataset_used: a single pandas DataFrame containing the strata variable and relevant variables to check.
        Within each strata, this needs to be sorted by time (for the cumulative product to work properly)
    :param var_names: a list of strings containing the names of the variables for which to check for jumps
    :param strata: a single string with the name of the strata variable
    :param jump_type: a string indicating the type of jump to be looking for, which can be 'perc', 'abs_perc',
        'diff', 'abs_diff', or 'sign'
    :param jump_size: minimum size of the jump to be identified as a 'large' jump
    :return: a tuple with a summary containing a single pandas DataFrame summarizing the number of large jumps for
        each variable, and a detailed DataFrame containing the count of number of jumps for each variable by subject
    """

    all_variables = [strata] + var_names
    dataset_slim = dataset_used[all_variables].copy()

    if jump_type == "perc":
        if jump_size > 0:
            check_dataset = 1 * (
                dataset_slim.groupby([strata])[var_names].pct_change(periods=1, axis=0)
                > jump_size
            )
        else:
            check_dataset = 1 * (
                dataset_slim.groupby([strata])[var_names].pct_change(periods=1, axis=0)
                < jump_size
            )
    elif jump_type == "abs_perc":
        check_dataset = 1 * (
            dataset_slim.groupby([strata])[var_names]
            .pct_change(periods=1, axis=0)
            .abs()
            > jump_size
        )
    elif jump_type == "diff":
        if jump_size > 0:
            check_dataset = 1 * (
                dataset_slim.groupby([strata])[var_names].diff(periods=1, axis=0)
                > jump_size
            )
        else:
            check_dataset = 1 * (
                dataset_slim.groupby([strata])[var_names].diff(periods=1, axis=0)
                < jump_size
            )
    elif jump_type == "abs_diff":
        check_dataset = 1 * (
            dataset_slim.groupby([strata])[var_names].diff(periods=1, axis=0).abs()
            > jump_size
        )
    elif jump_type == "sign":
        check_dataset = 1 * (
            dataset_slim.groupby([strata])[var_names].pct_change(periods=1, axis=0) < 0
        )
    else:
        raise Exception("Invalid jump type")

    # Add the logicals back to the data
    results = pd.concat(
        [dataset_slim[strata].reset_index(drop=True), check_dataset], axis=1
    )

    # Perform aggregations
    summary_by_loan = results.groupby([strata])[var_names].sum()
    number_non_null = results.fillna(0).groupby([strata])[var_names].sum()
    number_non_null.columns = ["NonNull_" + s for s in number_non_null.columns.values]
    summary_by_loan = pd.concat([summary_by_loan, number_non_null], axis=1)

    # Add in the length
    summary_by_loan = summary_by_loan.assign(
        length=dataset_slim.groupby([strata]).size()
    )
    summary = summary_by_loan.mean()

    return summary, summary_by_loan


def summarize_conditional(dataset_used, var_names, cutoff=0):
    """
    Calculates the count and conditional average for variables in a dataset

    :param dataset_used: a single Pandas DataFrame containing variables that should be summarized
    :param var_names: a list of strings containing the names of the variables for which to summarize
    :param cutoff: a double that contains the cutoff for which to use in the conditions (<= X, and >X)
    :return: Small Pandas DataFrame with the results
    """

    results = pd.concat(
        [
            (dataset_used[var_names] <= cutoff).sum(),
            (dataset_used[var_names] * ((dataset_used[var_names]) <= cutoff)).sum(),
            (dataset_used[var_names] > cutoff).sum(),
            (dataset_used[var_names] * ((dataset_used[var_names]) > cutoff)).sum(),
        ],
        axis=1,
    )

    results.columns = ["count_leq", "sum_leq", "count_gt", "sum_gt"]

    results = results.assign(ave_leq=results["sum_leq"] / results["count_leq"]).assign(
        ave_ge=results["sum_gt"] / results["count_gt"]
    )

    return results


def box_plot_by_strata(dataset_used, var_name, strata, vert=False):
    """
    Generates a box and whiskers plot by strata.

    :param dataset_used: a single Pandas DataFrame containing variables that should be summarized
    :param var_name: the name of the variable for which to generate box and whiskers plots
    :param strata: a single string with the name of the strata variable
    :param vert: boolean indicating whether the strata should be plotted on the y-axis
    :return: figure containing the box plot
    """

    num_items = dataset_used.groupby(strata).size()
    fig = plt.figure()
    if vert:
        fig.add_axes([0, 0, (len(num_items) / 20), 3])
    else:
        fig.add_axes([0, 0, 1, (len(num_items) / 20)])
    dataset_used.boxplot(
        column=var_name,
        by=strata,
        vert=vert,
        boxprops=dict(color="b", linewidth=3),
        medianprops=dict(color="b", linewidth=3),
        whiskerprops=dict(color="b", linewidth=3),
        showmeans=True,
        ax=fig.axes[0],
    )
    fig.suptitle("")
    plt.title("Boxplot of {}, by {}".format(var_name, strata))
    x_axis = fig.axes[0].axes.get_xaxis()
    x_axis.set_label_text("foo")
    x_label = x_axis.get_label()
    x_label.set_visible(False)
    plt.close()

    return fig


def periods_since_last_non_missing(
    dataset_used, var_names, strata, treat_zero_blank_as_missing=False, details=True
):
    """
    Calculates the number of periods (rows of data) to the closest non-missing observation for one or several variables.
    This currently only handles 'closest previous' non-missing.

    :param dataset_used: a single Pandas DataFrame containing variables that should be summarized
    :param var_names: list of variable names for which to calculate periods since last missing
    :param strata: a single string with the name of the strata variable
    :param treat_zero_blank_as_missing: Boolean whether to treat zero (numerical), blanks (for categorical) as missings
    :param details: option to output details in addition to summary
    :return: If details is requested, outputs a tuple with a summary of the average number of periods for a variable,
        and a detailed table with observation-level values for periods_since_last_non_missing (which can be merged back
        into the original dataset). If details is not requested, only a summary table is provided
    """

    if strata is None:
        all_variables = var_names
        dataset_slim = dataset_used[all_variables].copy()
        dataset_slim["__identity__"] = 1
        strata = "__identity__"

    else:
        all_variables = [strata] + var_names
        dataset_slim = dataset_used[all_variables].copy()

    if treat_zero_blank_as_missing:
        # Identify whether the variable is numeric or categorical
        numerics = (
            dataset_slim[var_names].select_dtypes(include=[np.number]).columns.values
        )
        categorical = list(set(var_names) - set(numerics))

        if len(numerics):
            dataset_slim[numerics] = np.where(
                dataset_slim[numerics] == 0, np.nan, dataset_slim[numerics]
            )
        if len(categorical):
            dataset_slim[categorical] = np.where(
                dataset_slim[categorical] == "", None, dataset_slim[categorical]
            )

    # Generate a counter by strata
    __count__ = dataset_slim.groupby([strata]).cumcount()

    # Auxiliary calculations
    new_names = []
    for var_name in var_names:
        true_index = ~dataset_slim[var_name].isnull()
        dataset_slim.loc[true_index, "__TRUE_{}".format(var_name)] = __count__[
            true_index
        ]
        new_names.append("__TRUE_{}".format(var_name))

    temp = dataset_slim.groupby([strata])[new_names].fillna(method="pad")

    # Generate detailed result
    detailed = pd.DataFrame()
    for var_name in var_names:
        detailed["{}_obs_since_nonNA".format(var_name)] = (
            __count__ - temp["__TRUE_{}".format(var_name)]
        )

    # Summarize the results
    null_table = detailed.isnull().sum()
    non_null_table = (1 * (~detailed.isnull())).sum()
    mean_table = detailed.mean()
    non_zero_count_table = (1 * (detailed > 0)).sum()
    mean_non_zero_table = detailed[detailed > 0].mean()

    summary = pd.DataFrame(
        {
            "Nulls": null_table,
            "Non Nulls": non_null_table,
            "Mean": mean_table,
            "Non Zeros": non_zero_count_table,
            "Mean Non Zeros": mean_non_zero_table,
        },
        columns=["Nulls", "Non Nulls", "Mean", "Non Zeros", "Mean Non Zeros"],
    )

    if details:
        return summary, detailed
    else:
        return summary


def calendar_since_last_non_missing(
    dataset_used,
    var_names,
    strata,
    date,
    diff_type="day",
    treat_zero_blank_as_missing=False,
    details=True,
):
    """
    Calculates the number of periods (rows of data) to the closest non-missing observation for one or several variables.
    This currently only handles 'closest previous' non-missing.

    :param dataset_used: a single Pandas DataFrame containing variables that should be summarized
    :param var_names: list of variable names for which to calculate periods since last missing
    :param strata: a single string with the name of the strata variable
    :param date: variable containing date, where date is a proper date variable (where subtractions are possible)
    :param diff_type: designates the type of date difference. Can be "day", "week", "month", "year"
    :param treat_zero_blank_as_missing: Boolean whether to treat zero (numerical), blanks (for categorical) as missings
    :param details: option to output details in addition to summary
    :return: If details is requested, outputs a tuple with a summary of the average number of periods for a variable,
        and a detailed table with observation-level values for periods_since_last_non_missing (which can be merged back
        into the original dataset). If details is not requested, only a summary table is provided
    """

    if strata is None:
        all_variables = [date] + var_names
        dataset_slim = dataset_used[all_variables].copy()
        dataset_slim["__identity__"] = 1
        strata = "__identity__"

    else:
        all_variables = [strata, date] + var_names
        dataset_slim = dataset_used[all_variables].copy()

    if treat_zero_blank_as_missing:
        # Identify whether the variable is numeric or categorical
        numerics = (
            dataset_slim[var_names].select_dtypes(include=[np.number]).columns.values
        )
        categorical = list(set(var_names) - set(numerics))

        if len(numerics):
            dataset_slim[numerics] = np.where(
                dataset_slim[numerics] == 0, np.nan, dataset_slim[numerics]
            )
        if len(categorical):
            dataset_slim[categorical] = np.where(
                dataset_slim[categorical] == "", np.nan, dataset_slim[categorical]
            )

    # Auxiliary calculations
    new_names = []
    for var_name in var_names:
        true_index = ~dataset_slim[var_name].isnull()
        dataset_slim.loc[true_index, "__TRUE_{}".format(var_name)] = dataset_slim.loc[
            true_index, date
        ]
        new_names.append("__TRUE_{}".format(var_name))

    temp = dataset_slim.groupby([strata])[new_names].fillna(method="pad")

    # Generate detailed result
    detailed = pd.DataFrame()
    for var_name in var_names:
        if diff_type == "day":
            diff_amount = (
                dataset_slim[date] - temp["__TRUE_{}".format(var_name)]
            ).dt.days
        elif diff_type == "week":
            diff_amount = (
                dataset_slim[date] - temp["__TRUE_{}".format(var_name)]
            ).dt.days / 7
        elif diff_type == "month":
            diff_amount = diff_month_pd(
                dataset_slim[date], temp["__TRUE_{}".format(var_name)]
            )
        elif diff_type == "year":
            diff_amount = (
                dataset_slim[date].dt.year - temp["__TRUE_{}".format(var_name)].dt.year
            )
        else:
            raise AttributeError(
                "Argument 'diff_type' must be one of 'day', 'week', 'month', or 'year'"
            )

        detailed["{}_obs_since_nonNA".format(var_name)] = diff_amount

    # Summarize the results
    null_table = detailed.isnull().sum()
    non_null_table = (1 * (~detailed.isnull())).sum()
    mean_table = detailed.mean()
    non_zero_count_table = (1 * (detailed > 0)).sum()
    mean_non_zero_table = detailed[detailed > 0].mean()

    summary = pd.DataFrame(
        {
            "Nulls": null_table,
            "Non Nulls": non_null_table,
            "Mean": mean_table,
            "Non Zeros": non_zero_count_table,
            "Mean Non Zeros": mean_non_zero_table,
        },
        columns=["Nulls", "Non Nulls", "Mean", "Non Zeros", "Mean Non Zeros"],
    )

    if details:
        return summary, detailed
    else:
        return summary


def intermittent_nulls(
    dataset_used, var_names, strata, zero=True, blank=True, missing=True, details=True
):
    """
    Identifies the presence of 0 (if numeric), blank (if categorical) or missing values between other values for
    variables, with option for doing so by strata. Between the types of values to detect, no distinction is made.

    :param dataset_used: a single Pandas DataFrame containing variables that should be summarized
    :param var_names: list of variable names for which to calculate periods since last missing
    :param strata: a single string with the name of the strata variable
    :param zero: boolean for whether to detect zero values
    :param blank: boolean for whether to detect blank values
    :param missing: boolean for whether to detect missing values
    :param details: option to output details in addition to summary
    :return: If details is requested, outputs a tuple with a summary of the average number of intermittent nulls for a
        variable per strata, and a detailed table with observation-level indicators of where these intermittent nulls
        exist (which can be merged back into the original dataset). If details is not requested, only a summary table
        is provided
    """

    if not any([zero, blank, missing]):
        raise AttributeError(
            "At least one of 'zero', 'blank' and 'missing' must be provided"
        )

    if strata is None:
        all_variables = var_names
        dataset_slim = dataset_used[all_variables].copy()
        dataset_slim["__identity__"] = 1
        strata = "__identity__"

    else:
        all_variables = [strata] + var_names
        dataset_slim = dataset_used[all_variables].copy()

    # Identify whether the variable is numeric or categorical
    numerics = dataset_slim[var_names].select_dtypes(include=[np.number]).columns.values
    categorical = list(set(var_names) - set(numerics))

    if zero:
        if len(numerics):
            dataset_slim[numerics] = np.where(
                dataset_slim[numerics] == 0, np.nan, dataset_slim[numerics]
            )
    if blank:
        if len(categorical):
            dataset_slim[categorical] = np.where(
                dataset_slim[categorical] == "", np.nan, dataset_slim[categorical]
            )

    # Fill up and down
    backfill = dataset_slim.groupby([strata])[var_names].fillna(method="backfill")
    forwardfill = dataset_slim.groupby([strata])[var_names].fillna(method="pad")

    # Identify the key observations
    detailed = pd.DataFrame()
    for var_name in var_names:
        if var_name in numerics:
            if zero and missing:
                temp = 1 * (
                    (~backfill[var_name].isnull())
                    & (~forwardfill[var_name].isnull())
                    & (dataset_slim[var_name].isnull())
                )
            elif zero and not missing:
                temp = 1 * (
                    (~backfill[var_name].isnull())
                    & (~forwardfill[var_name].isnull())
                    & (dataset_used[var_name] == 0)
                )
            else:
                temp = (
                    0 * dataset_used[var_name].isnull()
                )  # Should just be 0 all the way through

        else:
            if blank and missing:
                temp = 1 * (
                    (~backfill[var_name].isnull())
                    & (~forwardfill[var_name].isnull())
                    & (dataset_slim[var_name].isnull())
                )
            elif blank and not missing:
                temp = 1 * (
                    (~backfill[var_name].isnull())
                    & (~forwardfill[var_name].isnull())
                    & (dataset_used[var_name] == "")
                )
            else:
                temp = (
                    0 * dataset_used[var_name].isnull()
                )  # Should just be 0 all the way through

        detailed[var_name] = temp

    # Summarize the results
    mean_table = detailed.mean()
    non_zero_count_table = (1 * (detailed > 0)).sum()
    mean_non_zero_table = detailed[detailed > 0].mean()

    summary = pd.DataFrame(
        {
            "Mean": mean_table,
            "Non Zeros": non_zero_count_table,
            "Mean Non Zeros": mean_non_zero_table,
        },
        columns=["Mean", "Non Zeros", "Mean Non Zeros"],
    )

    if details:
        return summary, detailed
    else:
        return summary
