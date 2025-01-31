import json
import os
from typing import Union

import pandas as pd
import pyreadr
import yaml

from src.utils.utilfuncs import NestedDict


class DataLoader:
    """
    A utility class to load datasets in various formats (CSV, RData, pickle, JSON).

    Furthermore, it loads the metrics and coefficients from the processed output directory structure
    for further processing

    Attributes:
        nrows (int): Number of rows to read when loading CSV files. If None, all rows are loaded.
    """

    def __init__(self, nrows: int = None) -> None:
        """
        Initializes the DataLoader with the specified number of rows to load (for CSV files).

        Args:
            nrows: Number of rows to read (currently only implemented for CSV files).
        """
        self.nrows = nrows

    def read_csv(self, path_to_dataset: str) -> dict[str, pd.DataFrame]:
        """
        Loads all CSV files in the specified directory and returns them as pandas DataFrames.

        - Each CSV file is loaded into a DataFrame.
        - Drops any unnecessary "Unnamed: 0" column from the DataFrames.

        Args:
            path_to_dataset: Path to the directory containing the CSV files.

        Returns:
            dict[str, pd.DataFrame]: A dictionary with:
                - Keys: File names without extensions.
                - Values: Corresponding DataFrames.

        Raises:
            FileNotFoundError: If no CSV files are found in the specified directory.
        """
        files = os.listdir(path_to_dataset)
        csv_files = [
            file
            for file in files
            if file.endswith(".csv")
            and os.path.isfile(os.path.join(path_to_dataset, file))
        ]

        if not csv_files:
            raise FileNotFoundError(f"No CSV datasets found in {path_to_dataset}")

        df_dct = {
            file[:-4]: pd.read_csv(
                os.path.join(path_to_dataset, file), encoding="latin", nrows=self.nrows
            )
            for file in csv_files
        }

        for key, df in df_dct.items():
            if "Unnamed: 0" in df.columns:
                df_dct[key] = df.drop(["Unnamed: 0"], axis=1)

        return df_dct

    @staticmethod
    def read_r(path_to_dataset: str) -> dict[str, pd.DataFrame]:
        """
        Loads all R data files (.RData or .rds) in the specified directory and returns them as pandas DataFrames.

        Args:
            path_to_dataset: Path to the directory containing the R files.

        Returns:
            dict[str, pd.DataFrame]: A dictionary with:
                - Keys: File names without extensions.
                - Values: Corresponding DataFrames.

        Raises:
            FileNotFoundError: If no R files are found in the specified directory.
        """
        files = os.listdir(path_to_dataset)
        r_files = [
            file
            for file in files
            if file.endswith((".RData", ".rds"))
            and os.path.isfile(os.path.join(path_to_dataset, file))
        ]

        if not r_files:
            raise FileNotFoundError(f"No R datasets found in {path_to_dataset}")

        df_dct = {}
        for r_file in r_files:
            full_path = os.path.join(path_to_dataset, r_file)
            result = pyreadr.read_r(full_path)

            df_dct[r_file[:-4]] = next(
                iter(result.values())
            )  # Assuming one DataFrame per R file

        return df_dct

    @staticmethod
    def read_pkl(path_to_dataset: str) -> pd.DataFrame:
        """
        Loads a pickle file from the specified path and returns it as a DataFrame.

        Args:
            path_to_dataset: Path to the pickle file.

        Returns:
            pd.DataFrame: DataFrame loaded from the pickle file.
        """
        df = pd.read_pickle(path_to_dataset)
        return df

    @staticmethod
    def read_json(path_to_dataset: str) -> dict:
        """
        This method loads a JSON file from a given directory and returns a dataframe.

        Args:
            path_to_dataset: The path to the JSON file to be loaded.

        Returns:
            dict: Dictionary containing the JSON data.
        """
        with open(path_to_dataset, "r") as f:
            data = json.load(f)
        return data

    @staticmethod
    def read_txt(file_path: str) -> list[str]:
        """
        Loads a text file and returns its contents as a list of strings,
        where each line is stripped of leading/trailing whitespace.

        Args:
            file_path: The path to the input text file.

        Returns:
            list[str]: A list of strings, where each string corresponds to a line in the file.
        """
        with open(file_path, "r") as file:
            feature_lst = [line.strip() for line in file]
        return feature_lst

    @staticmethod
    def read_yaml(path_to_config: str) -> Union[dict, NestedDict]:
        """
        Loads a YAML configuration file from a given path and returns its content as a dictionary.

        Args:
            path_to_config: The path to the YAML file to be loaded.

        Returns:
            dict: A dictionary containing the parsed YAML data.
        """
        with open(path_to_config, "r") as f:
            data = yaml.safe_load(f)

        return data
