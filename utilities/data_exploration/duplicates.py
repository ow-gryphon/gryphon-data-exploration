import pandas as pd
from itertools import compress


def row_duplicates(frame, returned="number", print_info=False):
    """
    Wrapper for pandas 'duplicates' function, returning the row number, row index or a slice of the pandas dataset
    with the duplicates. The first occurrence is not classified as duplicate, use the partial_row_duplicates if you
    want to include it as a duplicate.

    :param frame: Dataframe to check for duplicates
    :param axis: 0 for duplicated rows, 1 for duplicated columns (identical variables)
    :param returned:

        - "index" to return the row / column index that contain duplicated data
        - "number" to return the row / column number that contain duplicated data
        - "boolean" to return True / False for whether a row is duplicated or not
        - "frame" to return the pandas dataframe that contain the duplicated data

    :param print_info: Boolean indicating whether to print the number of rows that are duplicated
    :return: Information describing the duplicated data, as determined by the value provided for argument 'returned'
    """

    # Check that the dataset is a pandas dataframe
    if not isinstance(frame, pd.DataFrame):
        raise AttributeError("The object entered as 'frame' is not a pandas dataframe")

    duplicated_boolean = frame.duplicated()

    # Convert the TRUE / FALSE list to identify where it is TRUE
    which_boolean = list(compress(range(len(duplicated_boolean)), duplicated_boolean))

    if print_info:
        print("There are {} row duplicates in the dataset".format(len(which_boolean)))

    if returned == "boolean":
        return_value = duplicated_boolean
    elif returned == "index":

        if len(which_boolean) == 0:
            return_value = []
        else:
            return_value = frame.index[which_boolean]

            # Check that there are no duplicates in the index, and if so, throw an error
            if any(return_value.duplicated()):
                raise ValueError(
                    "You have duplicated data that have the same index. Select a different return method"
                )

    elif returned == "number":
        return_value = which_boolean

    elif returned == "frame":
        return_value = frame.iloc[which_boolean, :]
        # This is the same as frame.loc[duplicated_boolean,:]

    return return_value


def column_duplicates(frame, returned="number", print_info=False):
    """
    Checks a pandas dataset for complete column duplicates, returning the column number, column name or a slice of the
    pandas dataset with the duplicates.

    :param frame: Dataframe to check for duplicates
    :param axis: 0 for duplicated rows, 1 for duplicated columns (identical variables)
    :param returned:

        - "index" to return the row / column index that contain duplicated data|
        - "number" to return the row / column number that contain duplicated data|
        - "boolean" to return True / False for whether a row is duplicated or not|
        - "frame" to return the pandas dataframe that contain the duplicated data

    :param print_info: Boolean indicating whether to print the number of columns that are duplicated
    :return: Information describing the duplicated data, as determined by the value provided for argument 'returned'
    """

    # Check that the dataset is a pandas dataframe
    if not isinstance(frame, pd.DataFrame):
        raise AttributeError("The object entered as 'frame' is not a pandas dataframe")

    frame_T = frame.T
    duplicated_boolean = frame_T.duplicated()

    # Convert the TRUE / FALSE list to identify where it is TRUE
    which_boolean = list(compress(range(len(duplicated_boolean)), duplicated_boolean))

    if print_info:
        print(
            "There are {} column duplicates in the dataset".format(len(which_boolean))
        )

    if returned == "boolean":
        return_value = duplicated_boolean

    elif returned == "index":

        if len(which_boolean) == 0:
            return_value = []
        else:
            return_value = frame.index[which_boolean]

            # Check that there are no duplicates in the index, and if so, throw an error
            if any(return_value.duplicated()):
                raise ValueError(
                    "You have duplicated data that have the same index. Select a different return method"
                )

    elif returned == "number":
        return_value = which_boolean

    elif returned == "frame":
        return_value = frame.iloc[:, which_boolean]

    return return_value


def partial_row_duplicates(
    frame, cols_for_row_check=None, returned="number", output="basic", print_info=False
):
    """
    Checks dataframe for partial row duplicates (duplicated rows based on subset of columns), and generates either:

        - Row numbers that are duplicates, including the first occurrence
        - A collection of row-number-sets that have the same values in the selected columns

    Note that other columns not included may be different, so additional scrutiny into these duplicates is recommended.
    If you do not care about these other columns and just want to remove any duplicates in these columns, you can subset
    the columns and run 'row_duplicates' to identify the rows to eliminate.

    :param frame: Dataframe to check for duplicates
    :param cols_for_row_check: List of specific column names to check jointly for duplicates. Default is all columns
    :param returned:

        - "index" to return the row index label that contain duplicated data
        - "number" to return the row number that contain duplicated data
    :param output:
        - "basic" to output list of all row duplicates represented by row #, but not associating the duplicate rows with eachother,
        - "detailed" to output a list of lists containing row numbers where the column values are identical
    :param print_info: Boolean indicating whether to print the number of rows that are duplicated
    :return: See the description for the argument 'output'
    """

    # Generate flag for partial row duplicates
    if cols_for_row_check is None:
        cols_for_row_check = list(frame.columns.values)

    new_frame = frame.copy()

    new_frame["__row_number__"] = list(range(new_frame.shape[0]))
    new_frame["__row_duplicate_flag__"] = new_frame.duplicated(
        subset=cols_for_row_check, keep=False
    )

    if output == "basic":
        duplicated_rows = new_frame.loc[
            new_frame["__row_duplicate_flag__"] == True, "__row_number__"  # noqa: E712
        ]
    elif output == "detailed":
        all_columns = cols_for_row_check.copy()
        all_columns.extend(["__row_number__", "__row_duplicate_flag__"])

        duplicated_frame = new_frame.loc[
            new_frame["__row_duplicate_flag__"], all_columns
        ]

        # Sort by variables
        duplicated_frame = duplicated_frame.sort_values(by=cols_for_row_check, axis=0)

        # Identify duplicates excluding first occurrence
        original_row_number = duplicated_frame["__row_number__"]
        actual_duplicate = duplicated_frame.duplicated(subset=cols_for_row_check)

        duplicated_rows = []
        temp_list = []
        for idx in range(len(original_row_number)):
            if not actual_duplicate.values[idx]:
                if idx > 0:
                    # Reached the next new value, so put the previous set of duplicates into the list
                    duplicated_rows.append(temp_list)
                if returned == "number":
                    temp_list = [original_row_number.values[idx]]
                elif returned == "index":
                    temp_list = [duplicated_frame.index.values[idx]]
            else:
                if returned == "number":
                    temp_list.append(original_row_number.values[idx])
                elif returned == "index":
                    temp_list.append(duplicated_frame.index.values[idx])
                if idx == len(original_row_number) - 1:
                    # Reached the last item in the dataframe, add this group to the list
                    duplicated_rows.append(temp_list)

    else:
        raise AttributeError(
            "The argument 'output' must be either 'basic' or 'detailed'"
        )

    if print_info:
        print(
            "There are {} row duplicates in the dataset, including the first occurrence".format(
                sum(new_frame["__row_duplicate_flag__"])
            )
        )

    return duplicated_rows


def partial_column_duplicates(
    frame,
    rows_for_column_check=None,
    returned="number",
    output="basic",
    print_info=False,
):
    """
    Checks dataframe for partial column duplicates (duplicated oclumns based on subset of rows), and generates either:

        - Column numbers that are duplicates, including the first occurrence
        - A collection of column-number-sets that have the same values in the selected rows

    Note that other rows not included may be different, so additional scrutiny into these duplicates is recommended.
    If you do not care about these other rows and just want to remove any duplicates in these columns, you can subset
    the rows and run 'column_duplicates' to identify the columns to eliminate.

    :param frame: Dataframe to check for duplicates
    :param rows_for_column_check: List of row numbers to check jointly for duplicates. Default is all rows
    :param returned:

        - "index" to return the column label that contain duplicated data
        - "number" to return the column number that contain duplicated data
    :param output:

        - "basic" to output list of all column duplicates represented by column #, but not associating the duplicate columns with eachother
        - "detailed" to output a list of lists containing column numbers where the column values are identical
    :param print_info: Boolean indicating whether to print the number of columns that are duplicated
    :return: See the description for the argument 'output'
    """

    # Reset index and transpose (reset index to avoid duplicated index)
    frame_T = frame.copy().reset_index(drop=True).T

    # Generate flag for partial row duplicates
    if rows_for_column_check is None:
        cols_for_row_check = list(frame_T.columns.values)
    else:
        cols_for_row_check = rows_for_column_check

    frame_T["__row_number__"] = list(range(frame_T.shape[0]))
    frame_T["__row_duplicate_flag__"] = frame_T.duplicated(
        subset=cols_for_row_check, keep=False
    )

    if output == "basic":
        duplicated_rows = frame_T.loc[
            frame_T["__row_duplicate_flag__"] == True, "__row_number__"  # noqa: E712
        ]
    elif output == "detailed":
        all_columns = cols_for_row_check.copy()
        all_columns.extend(["__row_number__", "__row_duplicate_flag__"])

        duplicated_frame = frame_T.loc[frame_T["__row_duplicate_flag__"], all_columns]

        # Sort by variables
        duplicated_frame = duplicated_frame.sort_values(by=cols_for_row_check, axis=0)

        # Identify duplicates excluding first occurrence
        original_row_number = duplicated_frame["__row_number__"]
        actual_duplicate = duplicated_frame.duplicated(subset=cols_for_row_check)

        duplicated_rows = []
        temp_list = []
        for idx in range(len(original_row_number)):
            if not actual_duplicate.values[idx]:
                if idx > 0:
                    # Reached the next new value, so put the previous set of duplicates into the list
                    duplicated_rows.append(temp_list)
                if returned == "number":
                    temp_list = [original_row_number.values[idx]]
                elif returned == "index":
                    temp_list = [duplicated_frame.index.values[idx]]
            else:
                if returned == "number":
                    temp_list.append(original_row_number.values[idx])
                elif returned == "index":
                    temp_list.append(duplicated_frame.index.values[idx])
                if idx == len(original_row_number) - 1:
                    # Reached the last item in the dataframe, add this group to the list
                    duplicated_rows.append(temp_list)

    else:
        raise AttributeError(
            "The argument 'output' must be either 'basic' or 'detailed'"
        )

    if print_info:
        print(
            "There are {} row duplicates in the dataset, including the first occurrence".format(
                sum(frame_T["__row_duplicate_flag__"])
            )
        )

    return duplicated_rows


def check_if_unique(frame, variables_to_check=None):
    """
    Checks the dataframe for unique keys based on specified list of column names.

    :param frame: Dataframe to check / create unique keys in
    :param variables_to_check: Set of columns that exist in the dataset to be checked for whether they are unique.
        A string can be used for a single variable, otherwise a list for multiple variables, or None for all variables
    :return: List of variables that are unique
    """

    key_col_list = []

    if variables_to_check is None:
        variables_to_check = frame.columns

    # Unique key detection
    for col in variables_to_check:
        if frame[col].nunique() == frame.shape[0]:
            key_col_list.append(col)

    return key_col_list


def generate_key(frame, var_concat, sep="|"):
    """
    Attempt to generate a unique key by concatenating variables in the dataframe.

    :param frame: Pandas dataframe for which to concatenate variables
    :param var_concat: List of variable names (in the frame) for which to concatenate
    :param sep: Separator to use for the values being concatenated
    :return: tuple containing a list of concatenated values, and a boolean indicating whether it is unique or not
    """

    # Check if any variable has na
    if len(var_concat) < 2:
        print("Must have at least 2 variables to concatenate")

    small_frame = frame[var_concat]

    has_null = small_frame.isnull().sum()
    if any(has_null):
        NA_variables = [
            small_frame.columns.values[idx]
            for idx in range(small_frame.shape[1])
            if small_frame.isnull().sum()[idx] > 0
        ]
        raise ValueError(
            "The following variables have Null values in them: {}".format(
                ", ".join(NA_variables)
            )
        )

    # Allow for generation of key based on concatenation of two columns
    new_variable = small_frame.apply(
        lambda x: sep.join([str(x_i) for x_i in x]), axis=1
    )

    is_unique = len(new_variable.unique()) == len(new_variable)

    # Check if this variable has unique values only
    return new_variable, is_unique
