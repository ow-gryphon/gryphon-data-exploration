import pandas as pd
import numpy as np
import scipy as sc
from typing import List, Optional, Tuple, Literal, Dict
from IPython import display
import matplotlib.pyplot as plt
from collections import OrderedDict
from pandas.api.types import is_string_dtype, is_numeric_dtype


def get_numerical_info(
    frame: pd.DataFrame,
    print_frame: Optional[bool] = True
):
    """
    Get information for numerical variables in a pandas dataframe.
    
    :param frame: pandas Dataframe to examine. Any non-numerical variables will be ignored
    :param print_frame: Whether to print the results (which is contained within pandas DataFrame)
    
    """

    numeric_frame = frame.select_dtypes(include=np.number)
    
    # Check there are columns available
    if len(numeric_frame.columns) == 0:
        print("No data within")
        return None
    
    numerical_summary = pd.concat([
        pd.DataFrame([
                  numeric_frame.dtypes,
                  numeric_frame.count(),
                  (~numeric_frame.isnull()).sum(),
                  (numeric_frame.isnull()).sum(),
                  (numeric_frame==0).sum(),
                  numeric_frame.nunique(),
                  numeric_frame.sum(),
                  numeric_frame.mean(),
                  numeric_frame.std(),
                  numeric_frame.skew(),
                  numeric_frame.mode().loc[0],
                  ],
                 index=["data type","count", "# non-missing", "# missing", "# of 0s", "# unique values", "sum", "mean", "standard deviation", "skewness", "mode (first)"]),
        numeric_frame.describe().loc[['min', '25%', '50%', '75%', 'max']]])
        
    if print_frame:
        display.display(numerical_summary)

    return numerical_summary


def get_non_numerical_info(frame: pd.DataFrame,
    print_frame: Optional[bool] = True
):
    """
    Get information for numerical variables in a pandas dataframe.
    
    :param frame: pandas Dataframe to examine. Any numerical variables will be ignored
    :param print_frame: Whether to print the results (which is contained within pandas DataFrame)
    
    """

    non_numerical_frame = frame.select_dtypes(exclude=np.number)
        
    # Check there are columns available
    if len(non_numerical_frame.columns) == 0:
        print("No data within")
    #     return None

    non_numerical_summary = pd.DataFrame([
                  non_numerical_frame.dtypes,
                  non_numerical_frame.count(),
                  (~non_numerical_frame.isnull()).sum(),
                  (non_numerical_frame.isnull()).sum(),
                  (non_numerical_frame=="").sum(),
                  non_numerical_frame.nunique(),
                  non_numerical_frame.mode().loc[0],
                  non_numerical_frame.apply(lambda x: np.nan if all(x.isnull()) else np.mean([len(str(item)) for item in x if item is not None]), axis=0),
                  non_numerical_frame.apply(lambda x: _check_if_numeric(x), axis=0),
                  non_numerical_frame.apply(lambda x: _check_if_datetime(x), axis=0),
                  ],
                 index=["data type","count", "# non-missing", "# missing", "# of blanks", "# unique values", "mode (first)", 
                        "average length", "potentially numeric", "potentially datetime"])
            
    if print_frame:
        display.display(non_numerical_summary)

    return non_numerical_summary
    
    
def null_histogram(
    frame,
    blanks_as_null: bool = True,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Distribution of null values",
    **kwargs: Optional[Dict]
):
    
    """
    Get information for numerical variables in a pandas dataframe.
    
    :param frame: pandas Dataframe to examine
    :param blanks_as_null: Whether to count blank strings as Null
    :param figsize: Figure size for matplotlib
    :param title: Title of the figure
    :param **kwargs: Other named arguments for use with plt.hist function
    """
    
    if blanks_as_null:
        missings = (frame.isnull() | (frame == "")).sum(axis=1)
    else:
        missings = frame.isnull().sum(axis=1)

    fig = plt.figure(figsize=figsize)
    
    # If the number of missing values is small, we can have each integer number of missing as a separate bar
    if (missings.max() < 20) and "bins" not in kwargs:
        n, bins, patches = plt.hist(x = missings, bins=np.arange(missings.max()+2)-0.5,**kwargs, ec="k");
    else:
        n, bins, patches = plt.hist(x = missings, **kwargs);
    
    plt.title(title)
    plt.xlabel("# Missings")
    plt.ylabel("Frequency")
    plt.xlim(bins[0],max(bins)*1.1)
    
    return fig, (n, bins, patches)


def null_line_chart(
    frame, 
    dimension: str = None,
    blanks_as_null: bool = True,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Number of null values"
):

    
    if blanks_as_null:
        missings = (frame.isnull() | (frame == "")).sum(axis=1)
    else:
        missings = frame.isnull().sum(axis=1)

    missing_data = pd.DataFrame({
        "dimension": frame.index.values if dimension is None else frame[dimension],
        "missings": missings
    })

    if any(missing_data['dimension'].isnull()):
        null_missings = missings[missing_data['dimension'].isnull()].mean().round(2)
        print("Warning: You have {} null values in the dimension variable, and the dataset has {} missing cells in each such row (on average).".format(missing_data['dimension'].isnull().sum(), null_missings))
    
    missing_summary = missing_data.groupby(by="dimension")['missings'].agg([np.mean, np.median, np.std, min, max, _q25, _q75]).sort_index().reset_index()
    missing_summary['std'] = missing_summary['std'].fillna(0)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(missing_summary['dimension'],missing_summary['q25'], label='25th percentile # missings', 
             color="green", linestyle='dashed')
    plt.plot(missing_summary['dimension'],missing_summary['q75'], label='75th percentile # missings', 
             color="orange", linestyle='dashed')
    plt.plot(missing_summary['dimension'],missing_summary['median'], label='Median # missings', color = 'black')
    plt.title(title)
    plt.xlabel('row index' if dimension is None else dimension)
    plt.ylabel("# of missings in each row")
    plt.legend(loc="best")

    if not is_numeric_dtype(missing_data['dimension']):
        if missing_summary.shape[0] > 20:
            print("Too many x-labels, only showing some labels")
            xloc = plt.MaxNLocator(20)
            ax.xaxis.set_major_locator(xloc)
        plt.xticks(rotation=45)
        
    return fig, missing_summary


def null_by_bin_chart(
    frame, 
    dimension: str,
    grouping: Literal["Equal Count", "Equal Width"] = "Equal Count",
    num_groups: int = 10,
    blanks_as_null: bool = True,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Number of null values",
    **kwargs: Optional[Dict]
):

    if blanks_as_null:
        missings = (frame.isnull() | (frame == "")).sum(axis=1)
    else:
        missings = frame.isnull().sum(axis=1)

    missing_data = pd.DataFrame({
        "dimension": frame.index.values if dimension is None else frame[dimension],
        "missings": missings
    })
    
    # Create the bin for the variable
    if not _check_if_numeric(missing_data['dimension']):
        raise ValueError("Variable {} selected as dimension cannot be interpreted as numeric".format(dimension))
        
    missing_data['dimension'] =  pd.to_numeric(missing_data['dimension'], errors='coerce')
    
    if grouping == "Equal Count":
        missing_data = missing_data.assign(bins=pd.qcut(missing_data['dimension'], int(num_groups), duplicates="drop"))
    else:
        missing_data = missing_data.assign(bins=pd.cut(missing_data['dimension'], np.linspace(min(missing_data['dimension']),max(missing_data['dimension']), 1+int(num_groups)),include_lowest = True))
    
    missing_data_grouped = missing_data.drop(columns='dimension').groupby(by="bins")
    
    N = missing_data['bins'].nunique()
    group_names = []
    fig = plt.figure()
    n = 0
    for id, grp in missing_data_grouped:
        n=n+1
        plt.boxplot(x='missings', data=grp, positions=[n]);
        group_names.append(str(id))
    plt.xticks([1+x for x in range(N)], group_names, rotation=45)
    
    plt.title(title)
    plt.xlabel(dimension)
    plt.ylabel("# of missings in each row")
        
    return fig, missing_data_grouped.agg([np.mean, np.std, np.median,  min, _q25, _q75, max]).sort_index().reset_index()
    

def _check_if_numeric(x):
    
    if all(x.isnull()):
        return "No"
    
    num_missing_before = x.isnull().sum()
    
    try:
        x_numeric = pd.to_numeric(x, errors='coerce')
    except:
        return "No"
    
    num_missing_after = x_numeric.isnull().sum()
    
    if num_missing_after > num_missing_before:
        return "No"
    
    else:
        return "Yes"        
        
        
def _check_if_datetime(x):
    
    if all(x.isnull()):
        return "No"
    
    num_missing_before = x.isnull().sum()
    
    try:
        x_datetime = pd.to_datetime(x)
    except:
        return "No"
    
    num_missing_after = x_datetime.isnull().sum()
    
    if num_missing_after > num_missing_before:
        return "No"
    
    else:
        return "Yes"


def _q25(x):
    return x.dropna().quantile(0.25)
    
    
def _q75(x):
    return x.dropna().quantile(0.75)
