import json
import os
import pickle
from typing import Union

import pandas as pd
from matplotlib.figure import Figure

from src.utils.utilfuncs import NestedDict


class DataSaver:
    """
    A utility class to store datasets in various formats (CSV, RData, pickle, JSON, png, pdf, etc).

    """

    @staticmethod
    def save_csv(df: pd.DataFrame, output_path: str) -> None:
        """
        Saves a DataFrame as a CSV file.

        Args:
            df: DataFrame to save.
            output_path: Path to save the CSV file (including filename and `.csv` extension).
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"CSV saved to {output_path}")

    @staticmethod
    def save_pickle(data: Union[pd.DataFrame, dict], output_path: str) -> None:
        """
        Saves an object as a pickle file.

        Args:
            data: Data to save (e.g., a DataFrame, dictionary, etc.).
            output_path: Path to save the pickle file (including filename and `.pkl` extension).
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Pickle file saved to {output_path}")

    @staticmethod
    def save_json(
        data: Union[dict, NestedDict], output_path: str, indent: int = 4
    ) -> None:
        """
        Saves a dictionary as a JSON file.

        Args:
            data: Dictionary to save.
            output_path: Path to save the JSON file (including filename and `.json` extension).
            indent: Number of spaces for JSON indentation. Defaults to 4.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=indent)

        print(f"File saved to {output_path}")

    @staticmethod
    def save_png(fig: Figure, output_path: str) -> None:
        """
        Saves a matplotlib figure as a PNG file.

        Args:
            fig: Matplotlib figure to save.
            output_path: Path to save the PNG file (including filename and `.png` extension).
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, format="png", bbox_inches="tight")

        print(f"PNG saved to {output_path}")

    @staticmethod
    def save_pdf(fig: Figure, output_path: str) -> None:
        """
        Saves a matplotlib figure as a PDF file.

        Args:
            fig: Matplotlib figure to save.
            output_path: Path to save the PDF file (including filename and `.pdf` extension).
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

        print(f"PDF saved to {output_path}")

    @staticmethod
    def save_excel(
        df: pd.DataFrame,
        output_path: str,
        merge_cells: bool = True,
        index: bool = True,
    ) -> None:
        """
        Saves a DataFrame as an Excel file.

        Args:
            df: DataFrame to save.
            output_path: Path to save the Excel file (including filename and `.xlsx` extension).
            merge_cells: If True, merge cells in the Excel file. Defaults to True.
            index:
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_excel(
            output_path, index=index, merge_cells=merge_cells, engine="openpyxl"
        )

        print(f"Excel file saved to {output_path}")

    @staticmethod
    def save_txt(output_file: str, data: list[str]) -> None:
        """
        Saves a list of strings to a text file, with each item written on a new line.

        Args:
            output_file: The path to the output text file.
            data: The list of strings to be written to the file.
        """
        with open(output_file, "w") as f:
            for item in data:
                f.write(item + "\n")
