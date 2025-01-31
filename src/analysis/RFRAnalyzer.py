import os
from typing import Optional

import numpy as np
import pandas as pd
import shapiq
from joblib import Parallel, delayed
from shapiq import InteractionValues
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.analysis.BaseMLAnalyzer import BaseMLAnalyzer
from src.utils.utilfuncs import NestedDict


class RFRAnalyzer(BaseMLAnalyzer):
    """
    Implements functionality for Random Forest Regression (RFR) analysis.

    This class extends the `BaseMLAnalyzer` to provide a specific implementation
    for nonlinear models using `RandomForestRegressor`. It includes additional
    methods for calculating SHAP interaction values (SHAP-IA) and aggregating
    results across imputations and repetitions.

    Additional Attributes (for other Attributes, see BaseMLAnalyzer):
        model (RandomForestRegressor): The Random Forest Regressor used for prediction.
        rng_ (np.random.RandomState): Local random number generator for reproducibility.
    """

    def __init__(
        self,
        cfg_analysis: NestedDict,
        output_dir: str,
        df: pd.DataFrame,
        rep: Optional[int],
        rank: Optional[int],
    ) -> None:
        """
        Initializes the RFRAnalyzer instance with the specified configuration and data.

        Args:
            cfg_analysis: Configuration dictionary specifying analysis parameters.
            output_dir: Directory where the analysis results will be stored.
            df: Input DataFrame containing features and labels for the analysis.
            rep: Repetition index for cross-validation splits.
            rank: Rank identifier for multi-node parallelism.
        """
        super().__init__(cfg_analysis, output_dir, df, rep, rank)
        self.model = RandomForestRegressor(
            random_state=self.cfg_analysis["random_state"]
        )
        self.rng_ = np.random.RandomState(
            self.cfg_analysis["random_state"]
        )  # Local RNG

    def calculate_shap_ia_values(
        self,
        X_scaled: pd.DataFrame,
        pipeline: Pipeline,
        combo_index_mapping: dict[int, tuple[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes SHAP interaction (SHAP-IA) values for Random Forest Regressor.

        This method calculates SHAP-IA values for the scaled dataset using a TreeExplainer
        and processes the results to create arrays representing interaction values and
        base values for all samples. Calculations can be parallelized.

        Args:
            X_scaled: Scaled DataFrame containing the features for SHAP-IA computation.
            pipeline: Sklearn Pipeline containing preprocessing and the fitted model.
            combo_index_mapping: Mapping of feature interactions to indices for SHAP-IA.

        Returns:
            ia_values_arr: 2D array with SHAP-IA values for each sample (rows) and feature
                combination (columns).
            base_values_arr: 1D array containing base values for each sample.
        """
        n_jobs = self.cfg_analysis["parallelize"]["shap_ia_values_n_jobs"]
        chunk_size = X_scaled.shape[0] // n_jobs + (X_scaled.shape[0] % n_jobs > 0)

        results = Parallel(n_jobs=n_jobs, verbose=1, backend=self.joblib_backend)(
            delayed(self.compute_shap_ia_values_for_chunk)(
                pipeline.named_steps["model"].regressor_, X_scaled[i : i + chunk_size]
            )
            for i in range(0, X_scaled.shape[0], chunk_size)
        )

        combined_results = [item for sublist in results for item in sublist]

        ia_values_arr, base_values_arr = self.process_shap_ia_values(
            results=combined_results,
            combo_index_mapping=combo_index_mapping,
            num_samples=X_scaled.shape[0],
        )

        return ia_values_arr, base_values_arr

    def process_shap_ia_values(
        self,
        results: list,
        combo_index_mapping: dict[int, tuple[int]],
        num_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Processes SHAP-IA results into arrays for analysis.

        The `results` from SHAP-IA computation contain:
        - `dict_values`: A dictionary with feature index tuples as keys and SHAP-IA values as values.
        - `baseline_value`: A scalar base value for the corresponding sample.

        The more extensive summarization procedure of the SHAP interaction values into more interpretable
        units is done in the 'ClusterSummarizer' class.

        Args:
            results: List of SHAP-IA computation results for individual samples.
            combo_index_mapping: Mapping of feature interactions to array indices.
            num_samples: Number of samples in the dataset.

        Returns:
            ia_values_arr: 2D array with SHAP-IA values for each sample and feature combination.
            base_values_arr: 1D array containing base values for each sample.
        """
        base_values_arr = np.array([sample.baseline_value for sample in results])

        combo_index_mapping_reverse = {v: k for k, v in combo_index_mapping.items()}

        ia_values_dct = [sample.dict_values for sample in results]
        ia_values_arr = np.zeros(
            (num_samples, len(combo_index_mapping)), dtype=np.float32
        )

        for sample_idx, sample_vals in enumerate(ia_values_dct):
            for feature_combo, value in sample_vals.items():
                idx = combo_index_mapping_reverse[feature_combo]
                ia_values_arr[sample_idx, idx] += value

        return ia_values_arr, base_values_arr

    def compute_shap_ia_values_for_chunk(
        self, model: RandomForestRegressor, X_subset: pd.DataFrame
    ) -> list[InteractionValues]:
        """
        Computes SHAP-IA values for a chunk of data.

        This method calculates pairwise SHAP interaction values for the specified subset of features
        using the SHAP-IQ implementation of TreeExplainer.

        Args:
            model: Trained RandomForestRegressor for SHAP-IA computation.
            X_subset: Subset of features for which SHAP-IA values are computed.

        Returns:
            list: List of SHAP-IQ InteractionValue objects for each sample in the subset.
        """
        self.logger.log(f"Currently processing these indices: {X_subset.index[:3]}")
        self.logger.log(
            f"Calculating SHAP for subset of length {len(X_subset)} in process {os.getpid()}"
        )

        explainer = shapiq.TreeExplainer(
            model=model,
            index=self.cfg_analysis["shap_ia_values"]["interaction_index"],
            min_order=self.cfg_analysis["shap_ia_values"]["min_order"],
            max_order=self.cfg_analysis["shap_ia_values"]["max_order"],
        )

        ia_value_lst = []
        for idx, x in X_subset.iterrows():
            ia_values_obj = explainer.explain(x.values)
            ia_value_lst.append(ia_values_obj)

        self.logger.log(
            f"SHAP IA computations for indices {X_subset.index[:3]} COMPLETED"
        )

        return ia_value_lst
