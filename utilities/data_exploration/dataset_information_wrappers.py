import pandas as pd
import os
import datetime
from ..data_checks.dataset_information import (
    get_basic_data_information,
    get_basic_variable_information,
    compare_datasets,
)
from ..general_utilities import create_folder
from ..report_generation import report_router
from ..report_generation.basic_css import get_default_css


def get_basic_data_information_wrapper(
    data,
    data_name,
    print_results=False,
    save_results=False,
    save_report=False,
    save_report_type="",
    save_folder="",
):
    """
    This is a wrapper for the get_basic_data_information function

    :param data: Dataframe to obtain metadata information on
    :param data_name: Name of the dataset
    :param print_results: User option to display some metadata information inline
    :param save_results: boolean indicating whether results should be saved to CSV
    :param save_report: boolean indicating whether report should be generated
    :param save_report_type: string (either Excel, Word, or HTML) for the type of report
    :param save_folder: filepath to the folder that results should be saved in
    :return: Tuple with dictionary containing results, and meta data
    """

    # Check for existence of folder, and if it doesn't exist, create it.
    if save_results or save_report:
        if not create_folder(save_folder):
            raise ValueError("Unable to create the output folder")

    # Run the code
    results = get_basic_data_information(frame=data, print_results=print_results)
    formatted_results = pd.concat(
        [
            pd.DataFrame({key: results[key]})
            if isinstance(results[key], list)
            else pd.DataFrame({key: [results[key]]})
            for key in results.keys()
        ],
        axis=1,
    )

    # Time stamp
    time_string = str(datetime.datetime.now()).split(".")[0]

    # Result export
    if save_results:
        formatted_results.to_csv(os.path.join(save_folder, "Basic Information.csv"))
        print("Saved the results into CSV called 'Basic Information.csv'")

    temp_report = [
        {
            "type": "report pandas table",
            "report_header": "Information about the loaded dataset",
            "report_text": "The dataset {data_to_run} has {ncols} fields and {nrows} rows, "
            + "taking up {memory} of memory when loaded into Python".format(
                data_to_run=data_name,
                ncols=results["Columns"],
                nrows=results["Rows"],
                memory=results["Total Memory Usage"],
            ),
            "content": pd.DataFrame(data=results, index=[0]),
            "tab_name": "Dataset Info",
        }
    ]

    if save_report:
        report_router.export_report(
            temp_report,
            save_report_type,
            save_folder,
            "Basic Dataset Information",
            chart_folder="NO CHARTS",
            style_string=get_default_css(),
        )

        print("Saved the Report into 'Basic Dataset Information'")

    temp_documentation = [
        {
            "type": "text",
            "style": "Heading 2",
            "content": "Information about loaded dataset",
        },
        {
            "type": "text",
            "content": "The dataset {data_to_run} has {ncols} fields and {nrows} rows,"
            + " taking up {memory} of memory when loaded into Python".format(
                data_to_run=data_name,
                ncols=results["Columns"],
                nrows=results["Rows"],
                memory=results["Total Memory Usage"],
            ),
        },
        {
            "type": "pandas table",
            "content": pd.DataFrame(data=results, index=[0]),
            "caption": True,
            "caption text": "Basic information about the dataset",
        },
    ]

    meta_data = {
        "export": True,
        "time": time_string,
        "item": "basic_data_information",
        "header": "Basic data information",
        "base_code": {
            "imports": [
                "from mercury.data_checks.dataset_information import get_basic_data_information"
            ],
            "code": """
results = get_basic_data_information(frame={data_name}, print_results=print_results)
formatted_results = pd.concat([pd.DataFrame({{key: results[key]}}) if isinstance(results[key],list) else pd.DataFrame(
{{key: [results[key]]}}) for key in results.keys()], axis=1)""".format(
                data_name=data_name
            ),
        },
        "full_code": {
            "imports": [
                "from mercury.data_checks.dataset_information_wrappers import get_basic_variable_information_wrapper"
            ],
            "code": """
results, meta = get_basic_variable_information_wrapper({data_name}, '{data_name}',
                                       print_results={print_results},
                                       save_results={save_results},
                                       save_report={save_report},
                                       save_report_type='{save_report_type}',
                                       save_folder='{save_folder}')
results_list.append(meta)""".format(
                data_name=data_name,
                print_results=print_results,
                save_results=save_results,
                save_report=save_report,
                save_report_type=save_report_type,
                save_folder=save_folder,
            ),
        },
        "output": [results],
        "results": {"Basic Info": formatted_results},
        "warnings": "",
        "description": "Generated basic information for the dataset {data_to_run}".format(
            data_to_run=data_name
        ),
        "documentation": temp_documentation,
        "report": temp_report,
    }

    return formatted_results, meta_data


def get_basic_variable_information_wrapper(
    data,
    data_name,
    print_results=False,
    save_results=False,
    save_report=False,
    save_report_type="",
    save_folder="",
):
    """
    This is a wrapper for the get_basic_variable_information function

    :param data: Dataframe to obtain metadata information on
    :param data_name: Name of the dataset
    :param print_results: User option to display some metadata information inline
    :param save_results: boolean indicating whether results should be saved to CSV
    :param save_report: boolean indicating whether report should be generated
    :param save_report_type: string (either Excel, Word, or HTML) for the type of report
    :param save_folder: filepath to the folder that results should be saved in
    :return: Tuple with dictionary containing results, and meta data
    """

    # Check for existence of folder, and if it doesn't exist, create it.
    if save_results or save_report:
        if not create_folder(save_folder):
            raise ValueError("Unable to create the output folder")

    # Run the code
    results = get_basic_variable_information(
        frame=data, key_vars=None, basic_output=False
    )
    results["Type"] = results["Type"].astype(str)

    # Time stamp
    time_string = str(datetime.datetime.now()).split(".")[0]

    # Result export
    if save_results:
        results.to_csv(os.path.join(save_folder, "Basic Variable Information.csv"))
        print("Saved the results into CSV called 'Basic Variable Information.csv'")

    temp_report = [
        {
            "type": "report pandas table",
            "report_header": "Information about variables in the loaded dataset",
            "content": pd.DataFrame(data=results),
            "tab_name": "Variable Info",
        }
    ]

    if save_report:
        report_router.export_report(
            temp_report,
            save_report_type,
            save_folder,
            "Basic Variable Information",
            chart_folder="NO CHARTS",
            style_string=get_default_css(),
        )

        print("Saved the Report into 'Basic Variable Information'")

    temp_documentation = [
        {
            "type": "text",
            "style": "Heading 2",
            "content": "Information about variables in loaded dataset",
        },
        {
            "type": "pandas table",
            "content": pd.DataFrame(data=results),
            "caption": True,
            "caption text": "Basic information about each variable in the dataset",
        },
    ]

    meta_data = {
        "export": True,
        "time": time_string,
        "item": "basic_var_information",
        "header": "Basic variable information",
        "base_code": {
            "imports": [
                "from mercury.data_checks.dataset_information import get_basic_variable_information"
            ],
            "code": """
results = get_basic_variable_information(frame={data_name}, key_vars=None, basic_output=False)
results["Type"] = results["Type"].astype(str)""".format(
                data_name=data_name
            ),
        },
        "full_code": {
            "imports": [
                "from mercury.data_checks.dataset_information_wrappers import get_basic_variable_information_wrapper"
            ],
            "code": """
results, meta = get_basic_variable_information_wrapper({data_name}, '{data_name}',
                                       print_results={print_results},
                                       save_results={save_results},
                                       save_report={save_report},
                                       save_report_type='{save_report_type}',
                                       save_folder='{save_folder}''
results_list.append(meta)""".format(
                data_name=data_name,
                print_results=print_results,
                save_results=save_results,
                save_report=save_report,
                save_report_type=save_report_type,
                save_folder=save_folder,
            ),
        },
        "output": [results],
        "results": {"Basic Info": results},
        "warnings": "",
        "description": "Generated basic variable information for the dataset {data_to_run}".format(
            data_to_run=data_name
        ),
        "documentation": temp_documentation,
        "report": temp_report,
    }

    return results, meta_data


def compare_datasets_wrapper(df_1, df_1_name, df_2, df_2_name, union_cols=None):
    """
    This is a wrapper for the compare_datasets function

    :param df1: First pandas dataframe
    :param df_1_name: Name of the first dataset
    :param df2: First pandas dataframe
    :param df_2_name: Name of the first dataset
    :param union_cols: Manually entered columns to example. If not, by default does union of 2 dataframes' columns
    :return: Tuple with DataFrame containing results, and meta data
    """

    # Run the code
    results = compare_datasets(df_1, df_1_name, df_2, df_2_name, union_cols)

    # Time stamp
    time_string = str(datetime.datetime.now()).split(".")[0]

    meta_data = {
        "export": True,
        "time": time_string,
        "item": "data_for_comparison",
        "header": "Dataframe generated containing comparisons",
        "base_code": {
            "imports": [
                "from mercury.data_checks.dataset_information import compare_datasets"
            ],
            "code": """
results = compare_datasets({df_1_name}, '{df_1_name}', {df_2_name}, '{df_2_name}', {union_cols})
""".format(
                df_1_name=df_1_name, df_2_name=df_2_name, union_cols=union_cols
            ),
        },
        "full_code": {
            "imports": [
                "from mercury.data_checks.dataset_information_wrappers import compare_datasets_wrapper"
            ],
            "code": """
results, meta = compare_datasets_wrapper({df_1_name}, '{df_1_name}', {df_2_name}, '{df_2_name}', {union_cols})
""".format(
                df_1_name=df_1_name, df_2_name=df_2_name, union_cols=union_cols
            ),
        },
        "output": [results],
        "results": {"sample data": results.head()},
        "warnings": "",
        "description": "Contains comparison of {} and {}".format(df_1_name, df_2_name),
        "documentation": [
            {
                "type": "text",
                "style": "Heading 2",
                "content": "Comparison of datasets {} and {}".format(
                    df_1_name, df_2_name
                ),
                "formatting": None,
                "bookmark": None,
            },
            {
                "type": "text",
                "content": "The dataframe contains metadata and variable comparisons.It has {nrow} rows and {ncol} columns".format(
                    nrow=results.shape[0], ncol=results.shape[1]
                ),
                "formatting": None,
                "bookmark": None,
            },
        ],
        "report": [
            {
                "type": "text",
                "style": "Heading 2",
                "content": "Generation of dataset for comparison between {} and {}".format(
                    df_1_name, df_2_name
                ),
                "formatting": None,
                "bookmark": None,
            },
            {
                "type": "report pandas table",
                "report_header": "Data for comparison",
                "content": pd.DataFrame(data=results),
                "tab_name": "Comparison",
            },
        ],
    }

    return results, meta_data
