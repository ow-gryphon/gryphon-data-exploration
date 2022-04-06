import pandas as pd
import numpy as np

# https://stackoverflow.com/questions/38090455/what-is-the-simplest-way-to-make-matplotlib-in-osx-work-in-a-virtual-environment
import matplotlib
import os

if os.name == "nt":
    matplotlib.use("Agg", warn=False)
# else:
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance


def get_univariate_outliers(data, perc=0.01, z_score=2, IQR_score=1.5, boxplot=True):
    """
    Generates indices identifying outliers beyond percentile, standard deviation, IQR thresholds,
    and generates a boxplot.

    :param data: pd.Series with the data to be assessed
    :param perc: extreme percentiles to be highlighted, use 0.01 or 0.99 for 1% and 99%. Use None to deactivate
    :param z_score: observations that are at least this many standard deviations away from the mean. None to deactivate
    :param IQR_score: observations that are at least this many multiples of IQR away from the mean. None to deactivate
    :param boxplot: boolean indicating whether a boxplot should be generated
    :return: tuple containing:

        - As first element, a dataframe with -1,0,1 values, where -1 indicates an outlier in the lower direction,
            1 indicates an outlier in the higher direction, and 0 indicates no outlier. Each column in this dataframe
                is related to the outlier criteria used.
        - As second element, a dataframe summarizing the lower and upper bounds
        - As the third element, a boxplot (or None) if not requested

    """

    bounds_frame = pd.DataFrame()
    outlier_indicator = pd.DataFrame()
    if perc is not None:
        if perc > 0 and perc < 1:
            lower_p = np.min([perc, 1 - perc])
            bounds = np.quantile(data, q=[lower_p, 1 - lower_p])
            lower_bound = bounds[0]
            upper_bound = bounds[1]

            outlier_indicator["perc_{}".format(perc)] = np.where(
                data < lower_bound, -1, np.where(data > upper_bound, 1, 0)
            )

            bounds_frame["perc_{}".format(perc)] = [lower_bound, upper_bound]

        else:
            raise ValueError(
                "Argument perc must be either None or between 0 and 1 exclusively"
            )

    if z_score is not None:
        if z_score > 0:
            stdev = np.std(data)
            mean = np.mean(data)

            lower_bound = mean - z_score * stdev
            upper_bound = mean + z_score * stdev

            outlier_indicator["zscore_{}".format(z_score)] = np.where(
                data < lower_bound, -1, np.where(data > upper_bound, 1, 0)
            )

            bounds_frame["zscore_{}".format(z_score)] = [lower_bound, upper_bound]

        else:
            raise ValueError("Argument z_score must be either None or above 0")

    if IQR_score is not None:
        if IQR_score > 0:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            mean = np.mean(data)

            lower_bound = mean - IQR_score * IQR
            upper_bound = mean + IQR_score * IQR

            outlier_indicator["IQRscore_{}".format(IQR_score)] = np.where(
                data < lower_bound, -1, np.where(data > upper_bound, 1, 0)
            )

            bounds_frame["IQRscore_{}".format(IQR_score)] = [lower_bound, upper_bound]

        else:
            raise ValueError("Argument IQR_score must be either None or above 0")

    if bounds_frame.shape[1] > 0:
        bounds_frame.index = ["Lower", "Upper"]

    if boxplot:
        fig = plt.figure()
        fig.add_axes([0, 0, 1, 0.5])
        sns.boxplot(x=data, ax=fig.axes[0])
        plt.close()
    else:
        fig = None

    return outlier_indicator, bounds_frame, fig


def distance_from_mean(frame, var_names, standardize=True, power=1):
    """
    Calculate Minkowski distance from mean to identify observations that are furthest from the mean.

    :param frame: pandas DataFrame
    :param var_names: list of numerical variables that should be used to calculate the distance
    :param standardize: boolean whether to standardize the dataset
    :param power: float for the power used in Minkowski distance, e.g.:

        - 1 = sum of individual distances
        - 2 = euclidean distance

    :return: pd.Series with distances in the same order as the observations in the dataset
    """

    # Standardize variables
    if standardize:
        scaler = StandardScaler()
        data_noNaN = frame[var_names].dropna()
        transformed_data_noNAN = scaler.fit_transform(data_noNaN)
        transformed_data = frame[var_names].copy()
        transformed_data[var_names] = np.nan
        transformed_data.loc[data_noNaN.index, var_names] = transformed_data_noNAN

    else:
        transformed_data = frame[var_names]

    # Calculate the mean
    mean_vector = transformed_data.mean().values
    distances = transformed_data.apply(
        lambda x: distance.minkowski(mean_vector, x, power)
        if all(~x.isnull())
        else np.nan,
        axis=1,
    )

    return distances


def isolation_forest(
    frame, var_names, random_state=100000, n_estimators=100, max_samples="auto"
):
    """
    Perform isolation forest to identify outliers and inliers.
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

    :param frame: Pandas dataframe with the variables for use in the analysis
    :param var_names: list of variable names to be used in analysis
    :param random_state: random seed
    :param n_estimators: number of estimators (trees)
    :param max_samples: the number of samples to draw from X to train each base estimator
    :return: pandas DataFrame with two columns, one indicating inliers (1) vs. outliers (-1), and another with
        the anomaly score
    """

    isolation_forest = IsolationForest(
        n_estimators=n_estimators, max_samples=max_samples, random_state=random_state
    )

    transformed_data = frame[var_names].dropna()
    xdata = transformed_data.values

    isolation_forest.fit(xdata)
    anomaly_score = isolation_forest.decision_function(xdata)
    prediction = isolation_forest.predict(xdata)

    # Put the results into a dataframe
    new_frame = pd.DataFrame(
        frame[var_names]
    )  # This is to create a dataframe with the right dimensions and index
    new_frame.loc[transformed_data.index.values, "is_inlier"] = prediction
    new_frame.loc[transformed_data.index.values, "anomaly_score"] = anomaly_score

    return new_frame
