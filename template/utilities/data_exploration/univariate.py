import pandas as pd
import numpy as np
import scipy
import scipy.stats
import matplotlib
import os
from math import log10, floor

import matplotlib.pyplot as plt
from collections import Counter


def plot_histogram_advanced(data, variable_name):
    """
    Plots 8 different types of histograms using various x and y scaling techniques.

    :param data: numpy or pd.Series containing the variable for plotting
    :param variable_name: name of variable being plotted, for use in chart names
    :return: fig with histograms
    """

    # Allocate space for 2 x 4 charts
    fig = plt.figure(figsize=(12, 20))

    # First plot is a standard histogram
    plt.subplot(421)
    plt.hist(data)
    plt.title("{}".format(variable_name))

    # Second plot scales the y-axis via log-transform
    plt.subplot(422)
    plt.hist(data, log=True)
    plt.title("{} w/ log-frequency".format(variable_name))

    # Third plot scales the x-axis using a robust log transform
    plt.subplot(423)
    plt.hist(np.sign(data) * np.log10(1 + np.abs(data)))
    plt.title("{} with symlog axis".format(variable_name))

    # Fourth plot scales the x-axis using a robust log transform, and y-axis using log transform
    plt.subplot(424)
    plt.hist(np.sign(data) * np.log10(1 + np.abs(data)), log=True)
    plt.title(
        "{} with symlog axis w/ log-frequency".format(
            variable_name
        )
    )

    # Fifth plot uses Doane binning for x-axis
    plt.subplot(425)
    plt.hist(data, bins="doane")
    plt.title("{}".format(variable_name))

    # Sixth plot uses Doane binning for x-axis, and scales y-axis using log-transform
    plt.subplot(426)
    plt.hist(data, bins="doane", log=True)
    plt.title("{} w/ log-frequency".format(variable_name))

    # Seventh plot uses bespoke binning for the x-axis
    # Eighth plot uses bespoke binning for the x-axis, and log-transform for the y-axis
    magnitude = np.floor(np.log10(0.00001+abs(data)))
    magnitude = magnitude[~np.isinf(magnitude)]
    max_m = int(max(magnitude))
    min_m = int(min(magnitude))
    magnitude_span = max_m - min_m + 1
    if magnitude_span > 4:
        if sum(magnitude < (max_m - 3)) < len(magnitude) / 5:
            plt.subplot(427)
            plt.hist(data, bins="auto")
            plt.title("{}".format(variable_name))
            plt.subplot(428)
            plt.hist(data, bins="auto", log=True)
            plt.title("{} w/ log-frequency".format(variable_name))
        else:
            breaks = [0]
            for m in range(min_m, max_m + 1):
                if sum((magnitude >= m) & (magnitude < m + 1)) > len(magnitude) / 10:
                    breaks.extend(np.arange(1, 9.5, 0.5) * (10 ** m))

            all_max = max(data)
            breaks_pos = [b for b in breaks if b < all_max]

            all_min = min(data)
            breaks_neg = [-b for b in breaks if -b > all_min]

            breaks = np.sort(
                np.unique(np.array([all_min] + breaks_neg + breaks_pos + [all_max]))
            )
            vals = np.bincount(np.digitize(data, breaks))
            break_names = [
                "{}_to_{}".format(breaks[i], breaks[i + 1])
                for i in range(0, len(breaks) - 1)
            ]
            plt.subplot(427)
            plt.bar(range(0, len(break_names)), vals[range(1, len(vals) - 1)])
            plt.xticks(range(0, len(break_names)), break_names, rotation=90)
            plt.title("{}".format(variable_name))
            ax = plt.subplot(428)
            plt.bar(range(0, len(break_names)), vals[range(1, len(vals) - 1)])
            ax.set_yscale("log")
            plt.xticks(range(0, len(break_names)), break_names, rotation=90)
            plt.title("{}".format(variable_name))

    return fig


def plot_histogram_body(
    data, variable_name, min_val=None, max_val=None, prob_exclude = 0.05, winsorize=False
):
    """
    This generates one or more histogram plot on the body of a variable (not the full distribution).

    :param data: numpy or pd.Series containing the variable for plotting
    :param variable_name: name of variable being plotted, for use in chart names
    :param probs_exclude: list with the percentiles to exclude from the data (0.05 means keep 5th to 95th perc). Each
        number will result in one separate histogram. A value of 0 in the list means that the entire dataset is used
    :param winsorize: Boolean indicating whether to winsorize or drop values beyond the bound
    :param save_file: filepath (including filename but without file extension) for saving png files. They will have a
        suffix given by the probs_exclude
    :return: figure with histogram
    """

    if (min_val is not None) or (max_val is not None):
        lower_threshold = min_val
        upper_threshold = max_val
        
        title = "Histogram of {} from {} to {}".format(variable_name, min_val, max_val)
    
    elif prob_exclude is not None:

        lower_prob = min(prob_exclude, 1 - prob_exclude)
        if lower_prob < 0:
            raise AttributeError(
                "You specified a negative percentile"
            )
        upper_prob = 1 - lower_prob

        lower_threshold = np.percentile(data, lower_prob * 100)
        upper_threshold = np.percentile(data, upper_prob * 100)
        
        title = "Histogram of {} from {} to {}".format(variable_name, lower_prob, upper_prob)
    else:
        raise AttributeError(
            "You did not specify any thresholds for the histogram"
        )
        
    if winsorize:
        temp_data = data
        if lower_threshold is not None:
            temp_data[data < lower_threshold] = lower_threshold
        
        if upper_threshold is not None:
            temp_data[data > upper_threshold] = upper_threshold
            
    else:
        temp_data = data[(data >= lower_threshold) & (data <= upper_threshold)]

    fig, ax = plt.subplots()
    ax.hist(temp_data)
    ax.set_title(title)
    

    return fig


def export_univariate_summary_numeric(
    dataset,
    variable_names,
    output_folder = "sample_output",
    moments=True,
    stats=True,
    quantiles=True,
    extreme=True,
    hist=True,
    weights=None,
):
    
    # Check output
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    for variable in variable_names:
        print("Processing {}".format(variable))
        figure, output = univariate_summary_numeric(
            data = dataset[variable],
            data_name = variable,
            moments=moments,
            stats=stats,
            quantiles=quantiles,
            extreme=extreme,
            hist=hist,
            weights=weights,
        )
        
        plt.savefig("{}/{}.png".format(output_folder, variable))
        plt.close()
        
    return "Completed"
    

def univariate_summary_numeric(
    data,
    data_name,
    moments=True,
    stats=True,
    quantiles=True,
    extreme=True,
    hist=True,
    weights=None,
):
    """
    Python version of SAS' proc univariate summary statistics
    (see e.g. https://commons.wikimedia.org/wiki/File:Example_of_PROC_Univariate_Output_from_SAS.jpg)

    :param data: numpy or pd.Series containing the variable for exploration
    :param data_name: name of variable provided, for use in labeling the output
    :param moments: boolean for whether to generate moments. Note that variance and standard deviation are sample
        measures
    :param stats: boolean for whether to generate basic stats
    :param quantiles: boolean for whether to generate quantiles
    :param extreme: boolean for whether to generate extreme values
    :param hist: boolean for whether to generate histogram plot
    :param weights: array with observation weights (Not yet implemented)
    :return: Tuple with figure with all information requested, and dictionary with the results
    """

    # Round to significant figures
    def round_sig(x, sig=2, digit=True):
    
        if np.log10(0.00001+abs(x)) > 2:
            rounded = round(x, 0)
        else:
            try:
                rounded = round(x, sig - int(floor(log10(abs(x)))) - 1)
            except:
                rounded = np.nan
        return rounded

    if isinstance(data, pd.Series):
        data = data.values

    if weights is not None:
        raise AttributeError("Weights have not been implemented yet")

    num_requested = 0
    heights = []
    if moments:
        num_requested += 1
        heights.append(6)
    if stats:
        num_requested += 1
        heights.append(4)
    if quantiles:
        num_requested += 1
        heights.append(11)
    if extreme:
        num_requested += 1
        heights.append(6)
    if hist:
        num_requested += 1
        heights.append(20)

    if num_requested == 0:
        raise AttributeError("You must request at least one output")

    clean_data = data[np.isfinite(data)]

    # Initialize figure with the requested number of axes
    fig, axs = plt.subplots(
        num_requested,
        1,
        gridspec_kw={"height_ratios": heights, "hspace": 0.2},
        figsize=(10, sum(heights) * 0.3),
    )
    plt.suptitle("Univariate information for {}".format(data_name), fontsize=16)

    # Initialize output
    output_dict = {}

    # Basic description
    description = scipy.stats.describe(clean_data)

    # Plot counter
    plot_counter = 0

    # Generate table with moments
    if moments:
        N = description[0]
        uncorr_SS = np.sum(clean_data ** 2)
        corr_SS = np.sum((clean_data - np.mean(clean_data)) ** 2)
        variance = description[3]
        stdev = np.sqrt(variance)
        mean = description[2]
        skewness = description[4]
        kurtosis = description[5]

        moments_table = np.transpose(
            [
                [
                    "N",
                    "Mean",
                    "Std Deviation",
                    "Skewness",
                    "Uncorrected SS",
                    "Coeff Variation",
                ],
                [
                    N,
                    round_sig(mean, 3),
                    round_sig(stdev, 3),
                    round_sig(skewness, 3),
                    round_sig(uncorr_SS, 3),
                    round_sig(stdev / mean, 3),
                ],
                [
                    "Sum of Weights",
                    "Sum of Observations",
                    "Variance",
                    "Kurtosis",
                    "Corrected SS",
                    "Std Error Mean",
                ],
                [
                    N,
                    round_sig(np.sum(clean_data), 3),
                    round_sig(variance, 3),
                    round_sig(kurtosis, 3),
                    round_sig(corr_SS, 3),
                    round_sig(stdev / np.sqrt(N), 3),
                ],
            ]
        )

        # Add to plot
        axs[plot_counter].axis("tight")
        axs[plot_counter].axis("off")
        axs[plot_counter].table(cellText=moments_table, loc="center")
        axs[plot_counter].set_title("Moments")

        plot_counter += 1

        # Add to output
        output_dict["moments"] = pd.DataFrame(moments_table)

    # Generate table with stats
    if stats:
        variance = description[3]
        stdev = np.sqrt(variance)
        mean = description[2]
        median = np.median(clean_data)
        mode = scipy.stats.mode(clean_data)[0][0]
        missing = len(data) - len(clean_data)
        ranges = description[1][1] - description[1][0]
        iqr = scipy.stats.iqr(clean_data)

        stats_table = np.transpose(
            [
                ["Mean", "Median", "Mode", "# Missing"],
                [round_sig(mean, 3), round_sig(median, 3), round_sig(mode, 3), missing],
                ["Std Deviation", "Variance", "Range", "Interquartile Range"],
                [
                    round_sig(stdev, 3),
                    round_sig(variance, 3),
                    round_sig(ranges, 3),
                    round_sig(iqr, 3),
                ],
            ]
        )

        # Add to plot
        axs[plot_counter].axis("tight")
        axs[plot_counter].axis("off")
        axs[plot_counter].table(
            cellText=stats_table,
            colLabels=["Location", "", "Variability", ""],
            loc="center",
        )
        axs[plot_counter].set_title("Basic Statistical Measures")

        plot_counter += 1

        # Add to output
        output_dict["stats"] = pd.DataFrame(stats_table)

    if quantiles:
        quantile_list = np.percentile(
            clean_data, q=[100, 99, 95, 90, 75, 50, 25, 10, 5, 1, 0]
        )

        quantile_table = np.transpose(
            [
                [
                    "100% Max",
                    "99%",
                    "95%",
                    "90%",
                    "75% Q3",
                    "50% Median",
                    "25% Q1",
                    "10%",
                    "5%",
                    "1%",
                    "0% Min",
                ],
                [round_sig(x, 3) for x in list(quantile_list)],
            ]
        )

        # Add to plot
        axs[plot_counter].axis("tight")
        axs[plot_counter].axis("off")
        axs[plot_counter].table(
            cellText=quantile_table, colLabels=["Level", "Quantile"], loc="center"
        )
        axs[plot_counter].set_title("Quantiles")

        plot_counter += 1

        # Add to output
        output_dict["quantile"] = pd.DataFrame(quantile_table)

    if extreme:
        nobs = min(5, description[0])  # Top 5 max and min
        missing = len(data) - sum(np.isfinite(data))
        max_ind = np.argpartition(data, -(nobs + missing))[
            -(nobs + missing) :  # noqa: E203
        ]
        if missing > 0:
            max_ind = max_ind[np.argsort(data[max_ind])][:-missing][::-1]
        else:
            max_ind = max_ind[np.argsort(data[max_ind])][::-1]

        min_ind = np.argpartition(data, nobs)[:nobs]
        min_ind = min_ind[np.argsort(data[min_ind])]

        extreme_table = np.transpose(
            [
                ["Value"] + [str(round_sig(data[idx], 3)) for idx in min_ind],
                ["Obs"] + [str(val) for val in min_ind],
                ["Value"] + [str(round_sig(data[idx], 3)) for idx in max_ind],
                ["Obs"] + [str(val) for val in max_ind],
            ]
        )

        # Add to plot
        axs[plot_counter].axis("tight")
        axs[plot_counter].axis("off")
        axs[plot_counter].table(
            cellText=extreme_table,
            colLabels=["Lowest", "", "Highest", ""],
            loc="center",
        )
        axs[plot_counter].set_title("Extreme Observations")

        plot_counter += 1

        # Add to output
        output_dict["extreme"] = pd.DataFrame(extreme_table)

    if hist:
        axs[plot_counter].hist(clean_data)
        axs[plot_counter].set_title("Histogram plot")

        plot_counter += 1
    plt.close(fig)
    return fig, output_dict


def univariate_summary_categorical(
    data, data_name, stats=True, freq=True, freq_plot=True, max_num=10
):
    """
    Generate basic information about categorical variables.

    :param data: numpy or pd.Series containing the categorical variable data
    :param data_name: name of variable provided, for use in labeling the output
    :param stats: generate basic statistics about the variable
    :param freq: generate frequency table for the most common values (and the least common)
    :param freq_plot: generate frequency plot (barplot)
    :param max_num: maximum number of top occurrences to use in frequency count
    :return: Tuple with figure with all information requested, and dictionary with the results
    """

    # Round to significant figures
    def round_sig(x, sig=2):
        try:
            rounded = round(x, sig - int(floor(log10(abs(x)))) - 1)
        except:
            rounded = np.nan
        return rounded

    if isinstance(data, pd.Series):
        data = data.values

    num_requested = 0
    heights = []

    clean_data = data[~np.equal(data, None)]

    if stats:
        num_requested += 1
        heights.append(6 * 0.9)
    if freq:
        num_requested += 1
        num_unique = len(np.unique(clean_data))
        if num_unique > max_num + 1:
            heights.append(3 + max_num * 0.9)
        else:
            heights.append(1 + num_unique * 0.9)

    if freq_plot:
        num_requested += 1
        heights.append(20)

    if num_requested == 0:
        raise AttributeError("You must request at least one output")

    # Initialize figure with the requested number of axes
    fig, axs = plt.subplots(
        num_requested,
        1,
        gridspec_kw={"height_ratios": heights, "hspace": 0.2},
        figsize=(10, sum(heights) * 0.3),
    )
    plt.suptitle("Univariate information for {}".format(data_name), fontsize=16)

    # Initialize output
    output_dict = {}

    # Plot counter
    plot_counter = 0

    # Generate frequencies
    freq_counter = Counter(clean_data)

    # Generate table with stats
    if stats:
        N = len(clean_data)
        missing = len(data) - len(clean_data)
        most_common = "{} ({})".format(
            freq_counter.most_common()[0][0], freq_counter.most_common()[0][1]
        )
        least_common = "{} ({})".format(
            freq_counter.most_common()[::-1][0][0],
            freq_counter.most_common()[::-1][0][1],
        )
        num_unique = len(freq_counter)
        all_counts = [val[1] for val in freq_counter.most_common()]
        total_counts = sum(all_counts)
        HHI = round_sig(sum([(100 * val / total_counts) ** 2 for val in all_counts]), 3)

        stats_table = np.transpose(
            [
                [
                    "# Obs",
                    "# Missing",
                    "# Unique",
                    "Most Common",
                    "Least Common",
                    "Herfindahl-Hirchman Index",
                ],
                [N, missing, num_unique, most_common, least_common, HHI],
            ]
        )

        # Add to plot
        axs[plot_counter].axis("tight")
        axs[plot_counter].axis("off")
        axs[plot_counter].table(cellText=stats_table, loc="center")
        axs[plot_counter].set_title("Basic Stats")

        plot_counter += 1

        # Add to output
        output_dict["stats"] = pd.DataFrame(stats_table)

    if freq or freq_plot:
        # Create the frequency table
        if len(freq_counter) > max_num + 1:
            top_list = pd.DataFrame(list(freq_counter.most_common()[:max_num]))
            middle_list = pd.DataFrame([("...Others...", np.nan)])
            bottom_list = pd.DataFrame(list([freq_counter.most_common()[-1]]))
            freq_table = pd.concat(
                [top_list, middle_list, bottom_list], axis=0
            ).reset_index(drop=True)

        else:
            freq_table = pd.DataFrame(list(freq_counter.most_common()))

        # Add to plot
        if freq:
            axs[plot_counter].axis("tight")
            axs[plot_counter].axis("off")
            axs[plot_counter].table(
                cellText=freq_table.values, colLabels=["Value", "Count"], loc="center"
            )
            axs[plot_counter].set_title("Frequency table")

            plot_counter += 1

            # Add to output
            output_dict["freq"] = freq_table

        if freq_plot:
            axs[plot_counter].bar(
                range(freq_table.shape[0]),
                freq_table[1].values,
                tick_label=freq_table[0].values,
            )
            axs[plot_counter].set_title("Frequency plot")
            plt.setp(axs[plot_counter].get_xticklabels(), rotation=90)

            plot_counter += 1
    plt.close(fig)
    return fig, output_dict
