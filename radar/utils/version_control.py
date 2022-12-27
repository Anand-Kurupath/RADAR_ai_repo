"""Module contains functions for reading and writing last version datasets."""
import glob
import json

from typing import Any

import os
import time as time

import pandas as pd


def create_file_path(file_path: str):
    """
    Helper function to check if directory exists and if not, create the directory.

    Args:
        file_path: str
            Path to file
    """
    if not os.path.isdir(file_path):
        os.mkdir(file_path)


def find_latest(
        path: str,
        file_format: str
) -> str:
    """
    Helper function to find the latest written file in a directory.

    Args:
        path: str
            The folder to search for.

        file_format: str
            The file format to search for.

    Returns:
        The latest written file.
    """
    
    list_of_files = glob.glob(f"{path}*.{file_format}")
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file


def write_file(
        data_file: Any,
        file_path: str,
        file_name: str,
        file_format: str
):
    """
    Helper function to write dataframes, models and models metadata to files.

    Args:

        data_file: Any
            The data file to write.

        file_path: str
            The folder name that will contain the data.

        file_name: str
            The file name of the data.

        file_format: str
            Format of the data.
    """
    create_file_path(file_path=file_path)
    time_version = time.strftime("%Y%m%d-%H%M%S")

    if file_format == "csv":
        data_file.to_csv(
            path_or_buf=os.path.join(file_path, f"{file_name}_{time_version}.{file_format}"),
            index=False,
        )

    elif file_format == "json":
        with open(os.path.join(file_path, f"{file_name}_{time_version}.{file_format}"),
                  "w", encoding="utf-8") as f_obj:
            json.dump(data_file, f_obj, indent=4)

    elif file_format == "ubj":
        data_file.save_model(os.path.join(file_path, f"{file_name}_{time_version}.{file_format}"))

    else:
        raise Exception(
            f"File format {file_format} is not defined \
              to be written using write_df for pandas dataframes."
        )


def read_file(
        file_path: str,
        file_name: str,
        file_format: str,
        return_path: bool = False
) -> Any:
    """
    Helper function to read a csv/json/ubj file.

    Args:
        file_path: str
            Folder name that will contain the data.

        file_name: str
            File name of the data.

        file_format: str
            File format of the data.

        return_path: default=False
            If True return the latest file path.

    Returns: Any
        The latest version of data file by provided path.
    """
    path=os.path.join(file_path, file_name)
    file=f"{path}.{file_format}"

    if file_format == "csv":
        data_file = pd.read_csv(file, header=0, infer_datetime_format=True, encoding= 'unicode_escape')

    elif file_format == "json":
        with open(file, "r", encoding="utf-8") as f_obj:
            data_file = json.load(f_obj)

    else:
        raise Exception(
            f"File format {file_format} is not defined \
              to be read using read_file."
        )

    if return_path:
        return data_file, file
    else:
        return data_file
