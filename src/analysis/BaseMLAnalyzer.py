import gc
import hashlib
import inspect
import itertools
import json
import os
import pickle
import sys
import threading
from abc import ABC, abstractmethod
from itertools import product
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np
import pandas as pd
import shap
import sklearn
from joblib import Parallel, delayed, Memory
from scipy.stats import spearmanr, pearsonr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    make_scorer,
    get_scorer,
)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.CustomScaler import CustomScaler
from src.analysis.Imputer import Imputer
from src.analysis.PearsonFeaureSelector import PearsonFeatureSelector
from src.analysis.ShuffledGroupKFold import ShuffledGroupKFold
from src.utils.DataSelector import DataSelector
from src.utils.Logger import Logger
from src.utils.Timer import Timer
from src.utils.utilfuncs import NestedDict

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option("future.no_silent_downcasting", True)


class BaseMLAnalyzer(ABC):
    """
    Abstract base class for the machine learning-based prediction procedures.

    This class provides a template for model-specific implementations, encapsulating functionality for:
    - Repeated nested cross-validation (CV).
    - Multiple imputations integrated in the CV procedure.
    - Feature importance analysis using SHAP values and SHAP interaction values.
    - High flexibility in the data pipelines.

    **Implementation Logic**
    -> see "methods_to_apply" in the CFG
        - Select the right data for analysis using the data_selector instance
        - Create an initial log with all important analysis params for sanity checking
        - Drop columns with zero variance
        - Create a sklearn.Pipeline with data preprocessing steps and the prediction model
        - conduct repeated-nested CV (10x10x10). This lies at the core of the class
            - It executes the nested_cv function for different data partitions (this includes the imputations, predictions,
                and SHAP calculations, see function definition for details)
            - At the end, it collects the results and set them as class attributes
        - Summarize the SHAP interaction values, if calculated
        - Store analysis results in the specified output directory

    Attributes:
        cfg_analysis (NestedDict): YAML configuration object specifying analysis parameters.
            Dynamically updated based on conditions (e.g., number of folds set to 10 when running on a cluster).

        ## Cluster and parallelization params
        rank (Optional[int]): Rank for multi-node parallelism. If no multi-node parallelism is used, this is None.
        split_reps (bool): Specifies whether to split repetitions. If True, we run one job per repetition. If False,
            we run all x repetitions (e.g., 10) of the repeated nested CV in one job.
        rep (Optional[int]): If split_reps is True, rep represents the current repetition number (e.g., 5).
            Otherwise, it is None.
        joblib_backend (str): Backend type for joblib-based parallelism (e.g., "loky", "threading").

        ## Defining analysis params
        crit (str): Criterion for analysis as defined in the cfg, could be
            # Main analysis
            - wb_state
            # Supplementary analysis
            - wb_trait
            - pa_state
            - na_state
            - pa_trait
            - na_trait
        feature_combination (str): The current combination of features to be included in the analysis, could be
            # Main analysis
            - pl (person-level variables)
            - srmc (self-reported micro context / ESM)
            - sens (Sensed variables)
            - mac (Country-level variables)
            - pl_srmc
            - pl_sens
            - pl_mac
            - pl_srmc_sens
            - pl_srmc_mac
            - all_in
            # Supplementary analyses, excluding neuroticism facets and self-esteem
            - pl_nnse
            - pl_srmc_nnse
            - pl_sens_nnse
            - pl_srmc_sens_nnse
            - pl_srmc_mac_nnse
            - all_in_nnse
            # Supplementary analyses, run pl + pl_srmc for a selected sample with high predictive performance
            - pl_control
            - pl_srmc_control
        samples_to_include (str): Determining which samples to include in the analysis.
            - If this is "all", we include all samples in every analysis for which the criterion is available
            (and thus impute many values).
            - If this is "selected", we only include the selected samples to reduce the number of imputed values.
            - If this is "control", we run the analysis using only person-level features using the same samples as
              in "selected"
        model (Optional[str]): Prediction model, defined in the subclasses. Is set as None in the BaseMLAnalyzer class.

        ## Util instances
        logger (Logger): Logger instance for managing logs specific to the analysis run.
        timer (Timer): Timer instance for measuring and logging runtime for specific methods and the whole analysis.
        data_selector (DataSelector): DataSelector instance for getting the samples and the features for the current analysis.

        ## Data preparation
        df (pd.DataFrame): DataFrame containing features and criteria for analysis. This is always the full df,
            as the df is passed in the main function and the data_selector gets the specific features and samples per analysis.
        X (Optional[pd.DataFrame]): DataFrame with features extracted from `df` (initialized as None).
            Is assigned in the pipeline.
        y (Optional[pd.Series]): Series containing the criteria for analysis (initialized as None).
            Is assigned in the pipeline.
        datasets_included (Optional[list[str]]): List of datasets used in the current analysis.
            Is assigned in the pipeline.
        rows_dropped_crit_na (Optional[int]): Number of rows dropped due to missing values in the criteria column.
            Is assigned in the pipeline and logged in the initial log.
        id_grouping_col (str): Column name used for ID-based grouping in CV.
        country_grouping_col (str): Column name used for country-based grouping in country-level imputations.
        years_col (str): Column name specifying years used in the country-level imputations.
        meta_vars (list[str]): List of metadata variables used in analysis. These are variables that we need in different
            steps in the analysis pipeline but that are no features used for predicting the criterion
            (i.e., grouping_id_col, country_grouping_col, years_col).
        pipeline (Pipeline): sklearn.Pipeline object defining preprocessing steps and the prediction model.

        ## CV params
        num_imputations (int): Number of imputations to handle missing data.
        num_inner_cv (int): Number of inner CV folds.
        num_outer_cv (int): Number of outer CV folds.
        num_reps (int): Number of repetitions for repeated nested CV.

        ## CV results
        best_models (NestedDict): Dictionary collecting the best models for each imputation/fold/rep.
        repeated_nested_scores (NestedDict): Nested dictionary collecting performance scores for each imputation/fold/rep.
        pred_vs_true (NestedDict): Dictionary storing predicted versus true values for evaluation for each imputation/fold/rep.
        lin_model_coefs (NestedDict): Coefficients of linear models for each imputation/fold/rep. Is only filled with values
            in the ENRAnalyzer subclass.
        shap_results (NestedDict): Dictionary containing SHAP values and associated data for the test sets:
            - "shap_values" (NestedDict): SHAP values for the test sets.
            - "base_values" (NestedDict): Corresponding base values.
            - "data" (NestedDict): Data for which SHAP values were computed.
        shap_ia_results (NestedDict): Dictionary containing SHAP interaction values and base values.

        ## Mapping variables for SHAP interaction values
        combo_index_mapping (Optional[dict[int, tuple[int, int]): Mapping of feature combinations to indices. Only defined if we
            compute SHAP interaction values, None otherwise.
        feature_index_mapping (Optional[dict[int, str]]): Mapping of individual features to indices. Only defined if we
            compute SHAP interaction values, None otherwise.
        num_combos (Optional[int]): Total number of feature combinations. Only defined if we compute SHAP interaction values,
            None otherwise.

        ## Output dir and filenames
        spec_output_path (str): Directory path where result files are stored. Passed as an argument as this
            differ depending on the analysis configurations defined in the config.
        performance_name (str): Output filename for performance results (without file extension), defined in the cfg.
        shap_value_name (str): Output filename for SHAP values (without file extension), defined in the cfg.
        shap_ia_values_for_local_name (str): Output filename for summarized SHAP interaction values that we can copy on
            a local computer. Defined in the cfg.
        shap_ia_values_for_cluster_name (str): Output filename for non-summarized SHAP interaction values that are only stored
            on the supercomputer cluster, as these data files are too large to be stored on a local compüuter. Defined in the cfg.
        lin_model_coefs_name (str): Output filename for linear model coefficients. Defined in the cfg
    """

    @abstractmethod
    def __init__(
        self,
        cfg_analysis: NestedDict,
        spec_output_path: str,
        df: pd.DataFrame,
        rep: int = None,
        rank: int = None,
    ) -> None:
        """
        Initializes the BaseMLAnalyzer ABC class with configuration and analysis details.

        Args:
            cfg_analysis: Configuration dictionary specifying analysis parameters.
            spec_output_path: Analysis specific directory path for the storing results.
            df: DataFrame containing features and criteria for analysis.
            rep: If split_reps is True, rep represents the current repetition number (e.g., 5). Otherwise, it is None.
            rank: Rank for multi-node parallelism. None if not multi-node parallelism is used.
        """
        self.cfg_analysis = cfg_analysis
        self.spec_output_path = spec_output_path

        # Cluster and parallelization params
        self.rank = rank
        self.joblib_backend = self.cfg_analysis["parallelize"]["joblib_backend"]
        self.rep = rep
        self.split_reps = self.cfg_analysis["split_reps"]

        # Analysis params
        self.crit = self.cfg_analysis["params"]["crit"]
        self.feature_combination = self.cfg_analysis["params"]["feature_combination"]
        self.samples_to_include = self.cfg_analysis["params"]["samples_to_include"]
        self.model = None

        # Data preparation
        self.df = df
        self.X = None
        self.y = None
        self.datasets_included = None
        self.rows_dropped_crit_na = None
        self.id_grouping_col = self.cfg_analysis["cv"]["id_grouping_col"]
        self.country_grouping_col = self.cfg_analysis["imputation"][
            "country_grouping_col"
        ]
        self.years_col = self.cfg_analysis["imputation"]["years_col"]
        self.meta_vars = [
            self.id_grouping_col,
            self.country_grouping_col,
            self.years_col,
        ]
        self.pipeline = None

        # Util instances
        self.logger = Logger(
            log_dir=self.spec_output_path,
            log_file="logs",
            rank=self.rank,
            rep=self.rep,
        )
        self.timer = Timer(self.logger)
        self.data_selector = DataSelector(
            cfg_analysis=self.cfg_analysis,
            df=self.df,
            feature_combination=self.feature_combination,
            crit=self.crit,
            samples_to_include=self.samples_to_include,
            meta_vars=self.meta_vars,
        )

        # Methods that get clocked
        self.repeated_nested_cv = self.timer._decorator(self.repeated_nested_cv)
        self.nested_cv = self.timer._decorator(self.nested_cv)
        self.summarize_shap_values_outer_cv = self.timer._decorator(
            self.summarize_shap_values_outer_cv
        )
        self.manual_grid_search = self.timer._decorator(self.manual_grid_search)
        self.impute_datasets = self.timer._decorator(self.impute_datasets)
        self.calculate_shap_ia_values = self.timer._decorator(
            self.calculate_shap_ia_values
        )

        # Results
        self.best_models = {}
        self.repeated_nested_scores = {}
        self.pred_vs_true = {}
        self.shap_results = {"shap_values": {}, "base_values": {}, "data": {}}
        self.shap_ia_results = {"shap_ia_values": {}, "base_values": {}}
        self.lin_model_coefs = {}

        # CV parameters
        self.num_inner_cv = self.cfg_analysis["cv"]["num_inner_cv"]
        self.num_outer_cv = self.cfg_analysis["cv"]["num_outer_cv"]
        self.num_reps = self.cfg_analysis["cv"]["num_reps"]
        self.num_imputations = self.cfg_analysis["imputation"]["num_imputations"]

        # Attributes for shap_ia_values
        self.combo_index_mapping = None
        self.feature_index_mapping = None
        self.num_combos = None

        # Output
        self.performance_name = self.cfg_analysis["output_filenames"][
            "performance"
        ]  # .json
        self.shap_value_name = self.cfg_analysis["output_filenames"][
            "shap_values"
        ]  # .pkl
        self.shap_ia_values_for_local_name = self.cfg_analysis["output_filenames"][
            "shap_ia_values_for_local"
        ]  # .pkl
        self.shap_ia_values_for_cluster_name = self.cfg_analysis["output_filenames"][
            "shap_ia_values_for_cluster"
        ]  # .pkl
        self.lin_model_coefs_name = self.cfg_analysis["output_filenames"][
            "lin_model_coefs"
        ]  # .json

    @property
    def model_name(self) -> str:
        """
        Returns the name of the model as a lowercase string.

        This property retrieves the class name of the model instance and converts it to lowercase.
        The resulting string provides a concise representation of the model's name (i.e., "elasticnet", "randomforestregressor").

        Returns:
            str: The lowercase name of the model class.
        """
        return self.model.__class__.__name__.lower()

    @property
    def imputer(self) -> Imputer:
        """
        Creates and returns an instance of the Imputer class configured for the current analysis.

        The Imputer is initialized with settings derived from the configuration (`cfg_analysis`) and includes:
        - Logging through the logger instance.
        - Model-specific configurations like random state, convergence threshold, and imputation parameters.
        - Columns for grouping data (e.g., by country or year).
        As the Imputer depends on self.model which is defined in the subclass, we cannot set this in the __init__ method.

        Returns:
            Imputer: An Imputer instance initialized with the specified settings.
        """
        return Imputer(
            logger=self.logger,
            model=self.model_name,
            fix_rs=self.cfg_analysis["random_state"],
            max_iter=self.cfg_analysis["imputation"]["max_iter"],
            num_imputations=self.cfg_analysis["imputation"]["num_imputations"],
            conv_thresh=self.cfg_analysis["imputation"]["conv_thresh"],
            tree_max_depth=self.cfg_analysis["imputation"]["tree_max_depth"],
            percentage_of_features=self.cfg_analysis["imputation"][
                "percentage_of_features"
            ],
            n_features_thresh=self.cfg_analysis["imputation"]["n_features_thresh"],
            sample_posterior=self.cfg_analysis["imputation"]["sample_posterior"],
            pmm_k=self.cfg_analysis["imputation"]["pmm_k"],
            country_group_by=self.country_grouping_col,
            years_col=self.years_col,
        )

    @property
    def hyperparameter_grid(self) -> dict[str, dict[list[int, float]]]:
        """
        Retrieves the hyperparameter grid for the current model.

        Returns:
            dict: The hyperparameter grid for the specified model, as defined in the configuration.
        """
        return self.cfg_analysis["model_hyperparameters"][self.model_name]

    def apply_methods(self, comm: Optional[object] = None) -> None:
        """
        Executes the analysis methods specified in the configuration.

        Methods to execute are defined in `cfg_analysis["methods_to_apply"]`. The method dynamically checks for
        their existence in the class and calls them. If `split_reps` is enabled, methods that depend on the results
        from all repetitions are skipped.

        Args:
            comm: Communication object (e.g., MPI communicator) passed to specific methods
                (like `repeated_nested_cv`).

        Raises:
            ValueError: If a method specified in `methods_to_apply` is not implemented.
        """
        for method_name in self.cfg_analysis["methods_to_apply"]:
            if not hasattr(self, method_name):
                raise ValueError(f"Method '{method_name}' is not implemented yet.")

            if self.split_reps and method_name in [
                "get_average_coefficients",
                "process_all_shap_ia_values",
            ]:
                log_message = f"Skipping {method_name} because split_reps is True."
                self.logger.log(log_message)
                print(log_message)
                continue

            log_message = f"  Executing {method_name}"
            self.logger.log(log_message)
            print(log_message)
            method_to_call = getattr(self, method_name)

            if method_name == "repeated_nested_cv":
                method_to_call(comm)

            else:
                method_to_call()

    def select_data(self) -> None:
        """
        Prepares data for analysis by selecting samples, features, and criteria.

        This method uses the `data_selector` instance to:
        - Select the appropriate samples based on `samples_to_include`.
        - Extract features based on the `feature_combination`.
        - Set the analysis criterion (`crit`).

        The processed `X` (features) and `y` (criteria) are assigned to class attributes.
        """
        self.data_selector.select_samples()
        self.data_selector.select_features()
        self.data_selector.select_criterion()

        setattr(self, "X", self.data_selector.X)
        setattr(self, "y", self.data_selector.y)

    def initial_info_log(self) -> None:
        """
        Logs key configuration and data details at the start of the analysis.

        This log includes:
        - Analysis parameters such as the prediction model, criterion, and feature combination.
        - Cross-validation settings (e.g., number of repetitions, folds, imputations).
        - Parallelization details like MPI settings and the number of jobs.
        - Data statistics such as the number of rows, columns, and missing values.

        Helps validate configuration, SLURM script parameter passing, and provides sanity-checking information.
        """
        self.logger.log("----------------------------------------------")

        self.logger.log("Global analysis params")
        self.logger.log(f"    Prediction model: {self.model_name}")
        self.logger.log(f"    Criterion: {self.crit}")
        self.logger.log(f"    Feature combination: {self.feature_combination}")
        self.logger.log(f"    Samples to included: {self.samples_to_include}")
        self.logger.log(f"    Actual datasets: {self.datasets_included}")

        self.logger.log("CV params")
        self.logger.log(f"    N repetitions: {self.num_reps}")
        self.logger.log(f"    N outer_cv: {self.num_outer_cv}")
        self.logger.log(f"    N inner_cv: {self.num_inner_cv}")
        self.logger.log(f"    N imputations: {self.num_imputations}")
        self.logger.log(f"    Model hyperparameters: {self.hyperparameter_grid}")
        self.logger.log(
            f"    N hyperparameter combinations: {len(list(product(*self.hyperparameter_grid.values())))}"
        )
        self.logger.log(
            f'    Optimization metric: {self.cfg_analysis["scoring_metric"]["inner_cv_loop"]["name"]}'
        )
        self.logger.log(
            f'    Max iter imputations: {self.cfg_analysis["imputation"]["max_iter"]}'
        )

        self.logger.log("Parallelization params")
        self.logger.log(
            f'    Use mpi4py to parallelize across nodes: {self.cfg_analysis["use_mpi4py"]}'
        )
        self.logger.log(f"    Split reps: {self.split_reps}")
        self.logger.log(f"    Current rank: {self.rank}")
        self.logger.log(
            f'    N jobs inner_cv: {self.cfg_analysis["parallelize"]["inner_cv_n_jobs"]}'
        )
        self.logger.log(
            f'    N jobs shap: {self.cfg_analysis["parallelize"]["shap_n_jobs"]}'
        )
        self.logger.log(
            f'    N jobs imputation runs: {self.cfg_analysis["parallelize"]["imputation_runs_n_jobs"]}'
        )
        self.logger.log(
            f'    Compute shap IA values: {self.cfg_analysis["shap_ia_values"]["comp_shap_ia_values"]}'
        )
        if self.cfg_analysis["shap_ia_values"]["comp_shap_ia_values"]:
            self.logger.log(
                f'    N jobs shap_ia_values: {self.cfg_analysis["parallelize"]["shap_ia_values_n_jobs"]}'
            )

        self.logger.log("Data params")
        self.logger.log(f"    Number of rows: {self.X.shape[0]}")
        self.logger.log(
            f"    Number of cols: {self.X.shape[1]}, including meta-columns that are no predictors"
        )
        X_copy = self.X.copy()
        X_no_meta_cols = X_copy.drop(
            columns=self.meta_vars + [self.id_grouping_col], errors="raise"
        )
        self.logger.log(
            f"    Number of cols: {X_no_meta_cols.shape[1]}, excluding meta-columns that are no predictors"
        )
        self.logger.log(
            f"    Number of rows dropped due to missing criterion {self.crit}: {self.rows_dropped_crit_na}"
        )
        na_counts = self.X.isna().sum()
        for column, count in na_counts[na_counts > 0].items():
            self.logger.log(f"      Number of NaNs in column {column}: {count}")
        self.logger.log("----------------------------------------------")

    def drop_zero_variance_cols(self) -> None:
        """
        Identifies and removes columns in the features (`self.X`) with zero variance.

        Zero variance columns are those that:
        - Contain only one unique value (excluding NaN).
        - Contain only NaN values.

        This reduces computational overhead during analysis. Logs details about the dropped columns.
        """
        zero_variance_cols = []

        for column in self.X.columns:
            unique_values = self.X[column].nunique(dropna=True)

            if unique_values <= 1:
                zero_variance_cols.append(column)
                self.logger.log(
                    f"      Dropping column '{column}' due to zero variance (only one unique value or NaNs)."
                )

        if zero_variance_cols:
            self.X.drop(columns=zero_variance_cols, inplace=True)
            self.logger.log(
                f"      {len(zero_variance_cols)} column(s) with zero variance dropped: {zero_variance_cols}"
            )

        else:
            self.logger.log("      No columns with zero variance found.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fits the machine learning model to the provided data.

        This method wraps the `fit` method of the model instance and should be implemented
        in subclasses for model-specific behavior. This adheres to the SciKit-Learn interface.

        Args:
            X (pd.DataFrame): Features for training.
            y (pd.Series): Target variable for training.
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions using the fitted machine learning model.

        This method wraps the `predict` method of the model instance and should be implemented
        in subclasses for model-specific behavior. This adheres to the SciKit-Learn interface.

        Args:
            X (pd.DataFrame): Features for generating predictions.

        Returns:
            np.ndarray: Predictions generated by the model.
        """
        return self.model.predict(X)

    def create_pipeline(self):
        """
        Creates a preprocessing and modeling pipeline for the analysis.

        The pipeline includes:
        - Preprocessing steps like scaling features (`X`) and the target variable (`y`).
        - Optional feature selection based on correlation for sensing features (not used in final code).
        - The wrapped estimator for the repeated nested cross-validation (CV) procedure.
        - Optional caching to improve pipeline efficiency (not used in final code, no improvements).
        - Set the finished pipeline as a feature attribute

        Sets the created pipeline as a class attribute (`self.pipeline`).
        """
        preprocessor = CustomScaler()
        target_scaler = StandardScaler()

        # wrap the model, as preprocessing steps are only applied to X, not to y
        model_wrapped = TransformedTargetRegressor(
            regressor=self.model, transformer=target_scaler
        )
        if self.cfg_analysis["cv"]["warm_start"]:
            model_wrapped.regressor.set_params(warm_start=True)

        if self.cfg_analysis["cv"]["cache_pipe"]:
            cache_folder = "./cache_directory"
            memory = Memory(cache_folder, verbose=0)
        else:
            memory = None

        if "_fs" in self.feature_combination:
            self.logger.log("    -> Include feature selection for sensing features")
            feature_selector = PearsonFeatureSelector(
                num_features=self.cfg_analysis["feature_selection"][
                    "num_sensing_features"
                ],
                target_prefix="sens_",
            )

            pipe = Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("feature_selection", feature_selector),
                    ("model", model_wrapped),
                ],
                memory=memory,
            )

        else:
            pipe = Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", model_wrapped),
                ],
                memory=memory,
            )

        setattr(self, "pipeline", pipe)

    def nested_cv(
        self,
        rep: int = None,
        X: pd.DataFrame = None,
        y: pd.Series = None,
        fix_rs: int = None,
        dynamic_rs: int = None,
    ) -> tuple[
        NestedDict,
        NestedDict,
        list,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Executes the nested cross-validation procedure, the core functionality of the BaseMLAnalyzer class.

        This method performs a single repetition of repeated nested cross-validation (CV) to evaluate machine learning models
        on a given dataset. Nested CV ensures unbiased model evaluation by separating hyperparameter tuning and
        model evaluation into inner and outer CV loops.

        **Workflow**:
        1. **Outer CV Splits**:
           - The outer loop splits the data into training and test sets for unbiased model evaluation.
           - Ensures no overlap in groups between training and test sets when `id_grouping_col` is used.

        2. **Inner CV and Grid Search**:
           - The inner loop, applied to the outer training set, performs hyperparameter tuning using GridSearchCV.
           - Identifies the best model based on a user-specified scoring metric.

        3. **Dataset Imputation**:
           - Missing values in training and test datasets are handled through imputation.
           - Multiple imputed datasets are generated for all outer test sets, all inner validation sets.

        4. **Model Evaluation**:
           - The best models from GridSearchCV are evaluated on the test sets for multiple metrics.
           - Optionally stores predictions vs. true values for further analysis.

        5. **SHAP Value Calculation**:
           - Aggregates SHAP values across outer folds for both train and test sets.
           - Optionally computes SHAP interaction values for supported models (e.g., RandomForestRegressor).

        **Special Features**:
        - Supports metadata routing (e.g., aligning sample weights with features during training).
        - Ensures group-aware splitting in cross-validation when `id_grouping_col` is specified.
        - Summarizes SHAP values and SHAP interaction values for the train and test sets.

        Args:
            rep: The repetition number for repeated nested CV (e.g., 0-9 for 10x10 CV).
            X: Feature matrix for the analysis. Defaults to the class attribute `self.X`.
            y: Target variable for the analysis. Defaults to the class attribute `self.y`.
            fix_rs: Fixed random state used for reproducibility across all analyses, defined in the config.
            dynamic_rs: Dynamic random state, varying across repetitions, to ensure different data partitions.

        Returns:
            tuple: Results from the nested CV procedure, containing:
                - nested_scores_rep (NestedDict): Metrics evaluated on the test sets for each outer fold and imputation.
                  Example: {"outer_fold_0": {"imp_0": {"r2": 0.75, "spearman": 0.65, ...}}}.
                - pred_vs_true_rep (NestedDict): Predictions vs. true values for each outer fold and imputation.
                  Example: {"outer_fold_0": {"imp_0": {index: (predicted_value, true_value)}}}.
                - ml_model_params (list): Best models or model parameters for each outer fold and imputation.
                - rep_shap_values (np.ndarray): SHAP values for the test set, summarized across outer folds.
                  Shape: (n_samples, n_features).
                - rep_base_values (np.ndarray): Base values for SHAP predictions on the test set.
                  Shape: (n_samples,).
                - rep_data (np.ndarray): Scaled feature values for SHAP analysis.
                  Shape: (n_samples, n_features).
                - ia_test_shap_values (Optional[np.ndarray]): SHAP interaction values for the test set,
                  summarized across outer folds if computed. Shape: (n_samples, n_features, n_features). None otherwise.
                - ia_base_values (Optional[np.ndarray]): Base values for SHAP interaction values if computed. None otherwise.

        Raises:
            AssertionError: If indices between training and target data do not match or group splitting fails.
            ValueError: If an unsupported model type is encountered during SHAP value calculation.

        Example:
            ```
            # Run nested CV for a specific repetition
            results = analyzer.nested_cv(rep=1, X=features, y=target, fix_rs=42, dynamic_rs=101)
            ```
        """
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        X, y = X.align(y, axis=0, join="inner")

        inner_cv = ShuffledGroupKFold(
            n_splits=self.num_inner_cv, random_state=fix_rs, shuffle=True
        )
        outer_cv = ShuffledGroupKFold(
            n_splits=self.num_outer_cv, random_state=dynamic_rs, shuffle=True
        )

        ml_pipelines = []
        ml_model_params = []
        nested_scores_rep = {}
        pred_vs_true_rep = {}
        X_train_imputed_lst = []
        X_test_imputed_lst = []

        for cv_idx, (train_index, test_index) in enumerate(
            outer_cv.split(X, y, groups=X[self.id_grouping_col])
        ):
            self.logger.log("-----------------------------------")
            self.logger.log("-----------------------------------")
            self.logger.log("#########")
            self.logger.log(
                f"Outer fold number {cv_idx}, going from 0 to {self.num_outer_cv - 1}"
            )
            self.logger.log("#########")

            ml_pipelines_sublst = []
            ml_model_params_sublst = []
            nested_scores_rep[f"outer_fold_{cv_idx}"] = {}
            pred_vs_true_rep[f"outer_fold_{cv_idx}"] = {}

            train_indices = X.index[train_index]
            test_indices = X.index[test_index]
            X_train, X_test = X.loc[train_indices], X.loc[test_indices]
            y_train, y_test = y.loc[train_indices], y.loc[test_indices]

            assert (
                X_train.index == y_train.index
            ).all(), "Indices between train data differ"
            assert (
                X_test.index == y_test.index
            ).all(), "Indices between test data differ"
            assert (
                len(
                    set(X_train[self.id_grouping_col]).intersection(
                        set(X_test[self.id_grouping_col])
                    )
                )
                == 0
            ), "Grouping in outer_cv did not work as expected"

            imputed_datasets = self.impute_datasets(
                X_train_outer_cv=X_train,
                X_test_outer_cv=X_test,
                y_train_outer_cv=y_train,
                inner_cv=inner_cv,
                num_imputations=self.num_imputations,
                groups=X_train[self.id_grouping_col],
                n_jobs=self.cfg_analysis["parallelize"]["imputation_runs_n_jobs"],
            )
            X_train_imputed_lst_for_fold = [
                imputed_datasets[num_imp]["outer_fold"]["X_train_for_test_full"]
                for num_imp in imputed_datasets
            ]
            X_train_imputed_lst.append(X_train_imputed_lst_for_fold)
            X_test_imputed_lst_for_fold = [
                imputed_datasets[num_imp]["outer_fold"]["X_test_imputed"]
                for num_imp in imputed_datasets
            ]
            X_test_imputed_lst.append(X_test_imputed_lst_for_fold)

            scoring_inner_cv = get_scorer(
                self.cfg_analysis["scoring_metric"]["inner_cv_loop"]["name"]
            )
            param_grid = list(ParameterGrid(self.hyperparameter_grid))
            for num_imp in range(self.num_imputations):
                nested_scores_rep[f"outer_fold_{cv_idx}"][f"imp_{num_imp}"] = {}
                grid_search_results = self.manual_grid_search(
                    imputed_datasets=imputed_datasets[num_imp],
                    y=y_train,
                    param_grid=param_grid,
                    scorer=scoring_inner_cv,
                    inner_cv=inner_cv,
                    n_jobs=self.cfg_analysis["parallelize"]["inner_cv_n_jobs"],
                )
                best_model = grid_search_results["best_model"]

                if self.model_name == "elasticnet":
                    ml_model_params_sublst.append(
                        best_model.named_steps["model"].regressor_
                    )
                else:
                    ml_model_params_sublst.append(
                        best_model.named_steps["model"].regressor_.get_params()
                    )
                ml_pipelines_sublst.append(best_model)

                X_test_current = imputed_datasets[num_imp]["outer_fold"][
                    "X_test_imputed"
                ]
                X_test_current = X_test_current.drop(
                    columns=self.meta_vars + [self.id_grouping_col], errors="ignore"
                )
                y_test_current = y_test.loc[X_test_current.index]

                scores = self.get_scores(
                    grid_search=SimpleNamespace(best_estimator_=best_model),
                    X_test=X_test_current,
                    y_test=y_test_current,
                )

                for metric, score in scores.items():
                    nested_scores_rep[f"outer_fold_{cv_idx}"][f"imp_{num_imp}"][
                        metric
                    ] = score

                if self.cfg_analysis["store_pred_and_true"]:
                    pred_vs_true_rep[f"outer_fold_{cv_idx}"][
                        f"imp_{num_imp}"
                    ] = self.get_pred_and_true_crit(
                        grid_search=SimpleNamespace(best_estimator_=best_model),
                        X_test=X_test_current,
                        y_test=y_test_current,
                    )

                del best_model
                del X_test_current

            ml_model_params.append(ml_model_params_sublst)
            ml_pipelines.append(ml_pipelines_sublst)

        X_filtered = X.drop(columns=self.meta_vars, errors="ignore")

        (
            rep_shap_values,
            rep_base_values,
            rep_data,
            ia_test_shap_values,
            ia_base_values,
        ) = self.summarize_shap_values_outer_cv(
            X_train_imputed_lst=X_train_imputed_lst,
            X_test_imputed_lst=X_test_imputed_lst,
            X=X_filtered,
            y=y,
            groups=X[self.id_grouping_col],
            pipelines=ml_pipelines,
            outer_cv=outer_cv,
        )

        del ml_pipelines
        gc.collect()

        return (
            nested_scores_rep,
            pred_vs_true_rep,
            ml_model_params,
            rep_shap_values,
            rep_base_values,
            rep_data,
            ia_test_shap_values,  # mean across imps
            ia_base_values,  # mean across imps
        )

    def add_study_var(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds one binary variable per study to the input DataFrame.
        Each variable indicates whether a row belongs to a given study,
        based on whether the row index starts with that study's name.

        Parameters:
        - df: Input DataFrame with row indices containing study prefixes.

        Returns:
        - pd.DataFrame: DataFrame with added binary columns for each study.
        """
        df.index = df.index.astype(str)

        samples_included = self.cfg_analysis["feature_sample_combinations"][self.feature_combination]

        for study in samples_included:
            col_name = f"study_var_{study}"
            df[col_name] = df.index.str.startswith(study).astype(int)

        return df

    def remove_study_var(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes the binary study indicator columns from the input DataFrame.
        These are columns named in the format 'study_var_{study_name}'.

        Parameters:
        - df: Input DataFrame with study indicator columns.

        Returns:
        - pd.DataFrame: DataFrame with study indicator columns removed.
        """
        samples_included = self.cfg_analysis["feature_sample_combinations"][self.feature_combination]

        for study in samples_included:
            col_name = f"study_var_{study}"
            if col_name in df.columns:
                df.drop(columns=col_name, inplace=True)

        return df

    def manual_grid_search(
        self,
        imputed_datasets: dict[str, dict[str, pd.DataFrame]],
        y: pd.Series,
        inner_cv: ShuffledGroupKFold,
        param_grid: list[dict[str, any]],
        scorer: callable,
        n_jobs: int,
    ) -> dict[str, any]:
        """
        Mimics the behavior of GridSearchCV using pre-imputed datasets and parallelizes evaluation across parameter grids.

        This function:
        1. Evaluates each parameter combination on multiple inner CV folds.
        2. Selects the best parameter combination based on the scoring metric.
        3. Re-fits the best model on the entire training dataset.

        We do this to optimize parallelization efficiency, because when using GridSearchCV we would not be able to
        fully parallelize the imputation process.

        Args:
            imputed_datasets: Dictionary containing datasets split by fold.
                - Each fold contains keys like 'X_train_for_val_full', 'train_indices', and 'val_indices'.
            y: Target variable for the analysis.
            inner_cv: Cross-validation splitter for the inner CV loop.
            param_grid: List of dictionaries, each specifying a parameter combination to evaluate.
            scorer: Scoring function that takes the model, validation features, and validation target as inputs.
            n_jobs: Number of parallel jobs to run.

        Returns:
            dict: Dictionary containing:
                - `all_results` (list): Evaluation results for all parameter combinations.
                - `best_params` (dict): Best parameter combination based on the scoring metric.
                - `best_score` (float): Best score achieved with the best parameter combination.
                - `best_model` (Pipeline): The model fitted on the full training data using the best parameters.
        """

        def evaluate_param_combination(
            params: dict[str, any],
            imputed_datasets: dict[str, dict[str, pd.DataFrame]],
            y: pd.Series,
            inner_cv: ShuffledGroupKFold,
            scorer: callable,
            pipeline: any,
            meta_vars: list[str],
        ) -> dict[str, any]:
            """
            Evaluates a single parameter combination across inner CV folds.

            For each fold:
            - Trains the model on the training set using the specified parameters.
            - Evaluates the model on the validation set using the provided scoring function.

            Args:
                params: Dictionary specifying a parameter combination to evaluate.
                imputed_datasets: Dictionary containing datasets split by fold.
                y: Target variable for the analysis.
                inner_cv: Cross-validation splitter for the inner CV loop.
                scorer: Scoring function to evaluate the model.
                pipeline: Pipeline containing the preprocessing steps and the model.
                meta_vars: List of metadata variables to exclude from feature matrices.

            Returns:
                dict: Dictionary containing:
                    - `params` (dict): Evaluated parameter combination.
                    - `score` (float): Average score across all folds.
            """
            param_results = []

            for fold in range(inner_cv.get_n_splits()):
                fold_name = f"inner_fold_{fold}"
                dataset = imputed_datasets[fold_name]
                X_full = dataset["X_train_for_val_full"]
                y_full = y.loc[X_full.index]

                train_indices = dataset["train_indices"]
                val_indices = dataset["val_indices"]

                X_train = X_full.loc[train_indices].drop(
                    columns=meta_vars, errors="ignore"
                )
                y_train = y_full.loc[train_indices]
                X_val = X_full.loc[val_indices].drop(columns=meta_vars, errors="ignore")
                y_val = y_full.loc[val_indices]

                model = clone(pipeline)
                model.set_params(**params)

                model.fit(X_train, y_train)
                score = scorer(model, X_val, y_val)
                param_results.append(score)

            avg_score = np.mean(param_results)
            return {"params": params, "score": avg_score}

        results = Parallel(
            n_jobs=n_jobs,
            backend=self.joblib_backend,
            verbose=3,
        )(
            delayed(evaluate_param_combination)(
                params,
                imputed_datasets,
                y,
                inner_cv,
                scorer,
                self.pipeline,
                self.meta_vars,
            )
            for params in param_grid
        )

        best_result = max(results, key=lambda x: x["score"])
        best_score = best_result["score"]
        best_params = best_result["params"]
        outer_fold_data = imputed_datasets["outer_fold"]

        X_train_full = outer_fold_data["X_train_for_test_full"]
        y_train_full = y.loc[X_train_full.index]
        X_train_full = X_train_full.drop(
            columns=self.meta_vars + [self.id_grouping_col], errors="ignore"
        )

        best_model = clone(self.pipeline)
        best_model.set_params(**best_params)
        best_model.fit(X_train_full, y_train_full)

        nested_scores_outer_fold_imp = {
            "all_results": results,
            "best_params": best_params,
            "best_score": best_score,
            "best_model": best_model,
        }

        return nested_scores_outer_fold_imp

    def impute_datasets(
        self,
        X_train_outer_cv: pd.DataFrame,
        X_test_outer_cv: pd.DataFrame,
        y_train_outer_cv: pd.Series,
        inner_cv: ShuffledGroupKFold,
        groups: pd.Series,
        num_imputations: int,
        n_jobs: int,
    ) -> NestedDict:
        """
        Imputes data for outer and inner CV folds across multiple imputations.

        This function handles:
        - **Inner CV**:
            - Splits the outer training dataset (`X_train_outer_cv`) into training and validation sets based on
              the provided `inner_cv` splitter.
            - For each fold, stores the datasets (`X_train` and `X_val`) along with their indices.
        - **Outer CV**:
            - Prepares the full training dataset (excluding the grouping column) and the test dataset
              (`X_test_outer_cv`) for imputation.
        - Executes parallel imputations for both inner and outer folds across the specified number of imputations.
        - Combines imputed datasets and reorders them according to the original indices where necessary.

        **Implementation Logic**:
        1. **Data Preparation**:
            - For each fold in `inner_cv`, split the outer training dataset into `X_train` and `X_val`,
              storing their indices and datasets.
            - Remove the grouping column (`self.id_grouping_col`) from `X_train_outer_cv` and `X_test_outer_cv` for outer CV.
        2. **Task Creation**:
            - Create a list of tasks to be parallelized. Each task corresponds to a fold and imputation combination.
        3. **Parallel Processing**:
            - Impute datasets in parallel using `self.impute_single_dataset`, which handles the imputation logic
              for a single dataset.
        4. **Reconstruction and Storage**:
            - For inner folds, reconstruct the full dataset (`X_train_for_val_full`) by combining `X_train` and `X_val`.
            - For outer folds, store the imputed training (`X_train_for_test_full`) and test datasets (`X_test_imputed`).
            - Store the imputed datasets in a nested dictionary organized by imputation number (`num_imp`) and fold.

        Args:
            X_train_outer_cv:Training data for the outer CV fold.
            X_test_outer_cv: Test data for the outer CV fold.
            y_train_outer_cv: Target variable for the outer training data.
            inner_cv: CV splitter for inner cross-validation folds.
            groups: Grouping column used to ensure group-based splits.
            num_imputations: Number of imputations to perform.
            n_jobs: Number of parallel jobs for imputations.

        Returns:
            NestedDict: Nested dictionary where:
                - Top-level keys are the imputation numbers (`int`).
                - Second-level keys are fold names (e.g., `"inner_fold_0"`, `"outer_fold"`).
                - Values are dictionaries containing imputed datasets and metadata.
        """
        data_dct = {}

        # Step 1: Data preparation
        if self.cfg_analysis["imputation"]["add_study_var"]:
            X_train_outer_cv = self.add_study_var(df=X_train_outer_cv)
            X_test_outer_cv = self.add_study_var(df=X_test_outer_cv)

        for fold, (train_idx, val_idx) in enumerate(
            inner_cv.split(X_train_outer_cv, y_train_outer_cv, groups=groups)
        ):
            train_indices = X_train_outer_cv.index[train_idx]
            val_indices = X_train_outer_cv.index[val_idx]
            X_train_inner_cv, X_val = (
                X_train_outer_cv.loc[train_indices],
                X_train_outer_cv.loc[val_indices],
            )
            data_dct[f"inner_fold_{fold}"] = [
                X_train_inner_cv,
                X_val,
                train_indices,
                val_indices,
            ]

        X_train_outer_cv = X_train_outer_cv.drop(
            self.id_grouping_col, axis=1, errors="ignore"
        )
        X_test_outer_cv = X_test_outer_cv.drop(
            self.id_grouping_col, axis=1, errors="ignore"
        )
        data_dct["outer_fold"] = [X_train_outer_cv, X_test_outer_cv]

        # Step 2: Task creation for parallel processing
        tasks = []
        result_dct = {}
        for num_imp in range(num_imputations):
            result_dct[num_imp] = {}
            for fold, (
                X_train,
                X_val_or_test,
                *indices,
            ) in data_dct.items():  # X_train_copy, X_val_or_test_copy,
                result_dct[num_imp][fold] = {}
                tasks.append(
                    (fold, num_imp, X_train, X_val_or_test, indices)
                )  # X_train_copy, X_val_or_test_copy,

        # Step 3: Parallel imputation
        results = Parallel(n_jobs=n_jobs, backend=self.joblib_backend, verbose=3)(
            delayed(self.impute_single_dataset)(fold, num_imp, X_train, X_val_or_test)
            for fold, num_imp, X_train, X_val_or_test, _ in tasks
        )

        # Step 4: Reconstruction and result storage
        for task_idx, (fold, num_imp, _, _, indices) in enumerate(tasks):
            X_train_result, X_val_or_test_result = results[task_idx]

            if self.cfg_analysis["imputation"]["add_study_var"]:
                X_train_result = self.remove_study_var(df=X_train_result)
                X_val_or_test_result = self.remove_study_var(df=X_val_or_test_result)

            if fold.startswith("inner"):
                recreated_df = pd.concat([X_train_result, X_val_or_test_result])
                recreated_df = recreated_df.reindex(X_train_outer_cv.index)
                train_indices, val_indices = indices

                result_dct[num_imp][fold] = {
                    "X_train_for_val_full": recreated_df,
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                }

            else:
                result_dct[num_imp][fold] = {
                    "X_train_for_test_full": X_train_result,
                    "X_test_imputed": X_val_or_test_result,
                }

        return result_dct

    def impute_single_dataset(
        self,
        fold: str,
        num_imp: int,
        X_train: pd.DataFrame,
        X_val_or_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Imputes missing values in a single dataset for a specific fold and imputation iteration.

        This function:
        - Fits the imputer on the training data.
        - Transforms both the training and validation/test data using the imputer.
        - Removes metadata columns if present.
        - Checks for and asserts the absence of NaN values in the imputed datasets.

        Args:
            fold: Identifier for the fold ("inner_fold_x" or "outer_fold").
            num_imp: Current imputation iteration (used for random seeds).
            X_train: Training dataset for imputation.
            X_val_or_test: Validation or test dataset for imputation.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Imputed training and validation/test datasets.

        Raises:
            AssertionError: If NaN values are found in the imputed training or validation/test datasets.
        """
        imputer = clone(self.imputer)
        self.logger.log("--------------------------------------------------")

        # RUn this as separate tasks that only do the evaluation and quit afterward
        if self.cfg_analysis["imputation"]["evaluate_robustness"]:
            for holdout_frac in self.cfg_analysis["imputation"]["holdout_fractions"]:
                self.logger.log(f"    Evaluate robustness of imputation")
                self.logger.log(f"      Holdout fraction: {holdout_frac}")
                imputer.evaluate_imputation_robustness(
                    X=X_train,
                    holdout_frac=holdout_frac,
                    random_state=42,
                    num_imp=num_imp
                )
            sys.exit()

        else:
            self.logger.log(
                f"    Starting to impute dataset {fold} imputation number {num_imp}"
            )
            self.log_thread()

            imputer.fit(
                X=X_train, num_imputation=num_imp
            )  # We need num_imp for different random seeds

            self.logger.log(
                f"    Number of Cols with NaNs in X_train: {len(X_train.columns[X_train.isna().any()])}"
            )
            X_train_imputed = imputer.transform(X=X_train)

            self.logger.log(
                f"    Number of Cols with NaNs in X_test: {len(X_train.columns[X_val_or_test.isna().any()])}"
            )
            X_val_test_imputed = imputer.transform(X=X_val_or_test)

            X_train_imputed = X_train_imputed.drop(columns=self.meta_vars, errors="ignore")
            X_val_test_imputed = X_val_test_imputed.drop(
                columns=self.meta_vars, errors="ignore"
            )

            assert self.check_for_nan(
                X_train_imputed, dataset_name="X_train_imputed"
            ), "There are NaN values in X_train_imputed!"
            assert self.check_for_nan(
                X_val_test_imputed, dataset_name="X_test_imputed"
            ), "There are NaN values in X_test_imputed!"

            return X_train_imputed, X_val_test_imputed

    def check_for_nan(self, df: pd.DataFrame, dataset_name: str = "") -> bool:
        """
        Checks for NaN values in a DataFrame and logs details if any are found.

        For each column containing NaN values, logs the column name and the number of NaN entries.

        Args:
            df: The DataFrame to check for NaN values.
            dataset_name: An optional name for the dataset to include in log messages.

        Returns:
            bool: True if no NaN values are found, False otherwise.
        """
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            for col in nan_columns:
                n_nans = df[col].isna().sum()
                self.logger.log(
                    f"{dataset_name} - Column '{col}' has {n_nans} NaN values."
                )
                print(f"{dataset_name} - Column '{col}' has {n_nans} NaN values.")

        return not bool(nan_columns)

    def get_scores(
        self, grid_search, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, float]:
        """
        Evaluates the best model on the test set using specified scoring metrics.

        This method:
        - Retrieves scoring functions based on the configuration.
        - Applies each scoring metric to evaluate the best estimator on the test data.

        Args:
            grid_search: The trained grid search object containing the best estimator.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            dict: Calculated scores for each metric specified in the configuration.
        """
        scoring_functions = {
            "neg_mean_squared_error": get_scorer("neg_mean_squared_error"),
            "r2": get_scorer("r2"),
            "spearman": make_scorer(self.spearman_corr),
            "pearson": make_scorer(self.pearson_corr),
        }

        scorers = {
            metric: scoring_functions[metric]
            for metric in self.cfg_analysis["scoring_metric"]["outer_cv_loop"]
        }

        scores = {
            metric: scorer(grid_search.best_estimator_, X_test, y_test)
            for metric, scorer in scorers.items()
        }

        return scores

    @staticmethod
    def get_pred_and_true_crit(
        grid_search, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[int, tuple[float, float]]:
        """
        Creates a mapping of sample indices to predicted and true values for the test set to store pred and true values.

        Args:
            grid_search: The trained grid search object containing the best estimator.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            dict: A mapping where keys are sample indices and values are tuples (predicted_value, true_value).
        """
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        pred_true_dict = {
            idx: (pred, true) for idx, pred, true in zip(X_test.index, y_pred, y_test)
        }

        return pred_true_dict

    def compute_shap_for_fold(
        self,
        num_cv_: int,
        pipeline: Pipeline,
        num_test_indices: list[np.ndarray[int]],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        Union[list[int], np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[list[int]],
    ]:
        """
        Computes SHAP and SHAP interaction values for a specific outer CV fold.

        This method is a parallelized implementation used within "summarize_shap_values_outer_cv".
        It calculates SHAP values (and optionally SHAP interaction values) for the test set of a given outer CV fold.

        **Implementation Logic**:
        1. Scale `X_train` and `X_test` using the preprocessing steps in the pipeline.
        2. If feature selection is applied, reduce the feature space and map selected features back to their
           original indices for easier interpretation.
        3. Compute SHAP values for the test set using the scaled training and test data.
        4. If SHAP interaction values are enabled and the model is `RandomForestRegressor`, calculate interaction
           values using pre-defined or newly created feature combinations.
        5. Return SHAP values, interaction values, and metadata for the outer CV fold.

        Args:
            num_cv_: Identifier for the current outer CV fold.
            pipeline: The pipeline containing preprocessing and the trained model for the fold.
            num_test_indices: A dictionary mapping fold numbers to test indices for the current outer CV loop.
            X_train: Training features for the outer CV fold.
            X_test: Test features for the outer CV fold.

        Returns:
            tuple: A tuple containing:
                - shap_values_test (np.ndarray): SHAP values for the test set (n_samples x n_features).
                - base_values_test (np.ndarray): Base values for SHAP predictions on the test set.
                - X_test_scaled (pd.DataFrame): Scaled test set features.
                - test_indices (list[int]): Indices of test samples for the current outer CV fold.
                - shap_ia_values_arr (Optional[np.ndarray]): SHAP interaction values for the test set (if computed).
                - shap_ia_base_values_arr (Optional[np.ndarray]): Base values for SHAP interaction values.
                - feature_indices (Optional[list[int]]): Original feature indices after feature selection (if applicable).
        """
        X_train_scaled = pipeline.named_steps["preprocess"].transform(X_train)  # scaler
        X_test_scaled = pipeline.named_steps["preprocess"].transform(X_test)  # scaler

        if "feature_selection" in pipeline.named_steps:
            X_train_scaled = pipeline.named_steps["feature_selection"].transform(
                X_train
            )
            X_test_scaled = pipeline.named_steps["feature_selection"].transform(X_test)

            original_feature_names = X_test.columns  # Feature names before selection
            selected_feature_names = (
                X_test_scaled.columns
                if isinstance(X_test_scaled, pd.DataFrame)
                else pipeline.named_steps["feature_selection"].selected_features_
            )
            feature_indices = [
                original_feature_names.get_loc(name)
                for name in selected_feature_names
                if name in original_feature_names
            ]

        else:
            feature_indices = None

        (
            shap_values_test,
            base_values_test,
        ) = self.calculate_shap_values(
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            pipeline=pipeline,
        )

        if (
            self.model_name == "randomforestregressor"
            and self.cfg_analysis["shap_ia_values"]["comp_shap_ia_values"]
        ):
            if self.combo_index_mapping is None:
                (
                    combo_index_mapping,
                    feature_index_mapping,
                    num_combos,
                ) = self.create_index_mappings(X_test)
                self.combo_index_mapping = combo_index_mapping
                self.feature_index_mapping = feature_index_mapping
                self.num_combos = num_combos

            shap_ia_values_arr, shap_ia_base_values_arr = self.calculate_shap_ia_values(
                X_scaled=X_test_scaled,
                pipeline=pipeline,
                combo_index_mapping=self.combo_index_mapping,
            )

        else:
            shap_ia_values_arr = None
            shap_ia_base_values_arr = None

        return (
            shap_values_test,
            base_values_test,
            X_test_scaled,
            num_test_indices[num_cv_],
            shap_ia_values_arr,
            shap_ia_base_values_arr,
            feature_indices,
        )

    def calculate_shap_ia_values(
        self, X_scaled: pd.DataFrame, pipeline: Pipeline, combo_index_mapping: dict
    ) -> tuple[dict, list[float]]:
        """
        Computes SHAP interaction values for feature combinations.

        This is a placeholder method intended to be implemented in subclasses with complex prediction models.
        It should calculate SHAP interaction values for the provided scaled feature set (`X_scaled`),
        using the specified pipeline and mapping feature combinations to indices.

        Args:
            X_scaled: Scaled feature set for which SHAP interaction values are computed.
            pipeline: The model pipeline used to generate SHAP values.
            combo_index_mapping: A mapping between feature combinations and unique indices.

        Returns:
            tuple:
                - dict: SHAP interaction values for each feature combination.
                - list[float]: Summary statistics or aggregated SHAP interaction values.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def summarize_shap_values_outer_cv(
        self,
        X_train_imputed_lst: list[list[pd.DataFrame]],
        X_test_imputed_lst: list[list[pd.DataFrame]],
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        pipelines: list[list[sklearn.pipeline.Pipeline]],
        outer_cv: ShuffledGroupKFold,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """
        Summarizes SHAP values from nested CV procedure.

        This function calculates SHAP values for test samples across outer folds and imputations,
        then aggregates these values to ensure that each sample's SHAP values are represented only once
        for each repetition. The methodology follows Scheda & Diciotti (2022) and focuses only on test set SHAP values.

        Additionally, if SHAP interaction values are enabled and the model is a `RandomForestRegressor`,
        it computes and aggregates SHAP interaction values.

        **Implementation Logic**:
        1. **Test Sample Identification**:
           - Determine the numerical indices of test samples for each outer fold using the `outer_cv` object.
        2. **Template Initialization**:
           - Create zero-filled 3D arrays (`n_samples x n_features x n_imputations`) to store SHAP values, base values,
             and input data for all samples.
        3. **SHAP Computation**:
           - For each outer fold and imputation, compute SHAP values for the test set using `compute_shap_for_fold`.
           - Store SHAP values in the corresponding array positions, distinguishing by fold and imputation.
           - If feature selection is applied, ensure only selected features are aggregated.
        4. **SHAP Interaction Values**:
           - If SHAP interaction values are enabled (`self.cfg_analysis["shap_ia_values"]["comp_shap_ia_values"]`),
             compute and aggregate these values similarly to SHAP values.
        5. **Aggregation**:
           - Divide base values by the number of outer CV folds to normalize repeated calculations.
           - Return aggregated SHAP and interaction values.

        Args:
            X_train_imputed_lst: Nested list of imputed training DataFrames.
                - First-level index corresponds to the outer fold.
                - Second-level index corresponds to the imputation.
            X_test_imputed_lst: Nested list of imputed test DataFrames.
                - Structured similarly to `X_train_imputed_lst`.
            X: Raw feature set containing all samples (training and test).
            y: Target variable corresponding to `X`.
            groups: Grouping variable for the nested cross-validation splits.
            pipelines: Nested list of pipelines, where each pipeline corresponds to the best estimator for a specific fold
                and imputation.
            outer_cv: A `GroupKFold` object used for splitting the data into train and test sets for the outer CV loop.

        Returns:
            tuple: A tuple containing:
                - rep_shap_values (np.ndarray): Aggregated SHAP values for all test samples.
                    Shape: `(n_samples, n_features, n_imputations)`.
                - rep_base_values (np.ndarray): Aggregated base values for all test samples.
                    Shape: `(n_samples, n_imputations)`.
                - rep_data (np.ndarray): Aggregated scaled feature values for all test samples.
                    Shape: `(n_samples, n_features, n_imputations)`.
                - ia_test_shap_values (Optional[np.ndarray]): Aggregated SHAP interaction values for test samples if computed.
                    Shape: `(n_samples, n_combos)`. None if interaction values are not computed.
                - ia_test_base_values (Optional[np.ndarray]): Aggregated base values for SHAP interaction values if computed.
                    Shape: `(n_samples,)`. None if interaction values are not computed.
        """
        print("---------------------")
        print("Calculate SHAP values")
        num_test_indices = [test for train, test in outer_cv.split(X, y, groups=groups)]

        # 3D arrays (rows x features x imputations)
        rep_shap_values = np.zeros(
            (X.shape[0], X.shape[1], self.num_imputations), dtype=np.float32
        )
        rep_base_values = np.zeros(
            (X.shape[0], self.num_imputations), dtype=np.float32
        )  # we get different base values per fold
        rep_data = np.zeros(
            (X.shape[0], X.shape[1], self.num_imputations), dtype=np.float32
        )

        results = [
            [
                self.compute_shap_for_fold(
                    num_cv_=num_cv_,
                    pipeline=pipelines[num_cv_][num_imputation],
                    num_test_indices=num_test_indices,
                    X_train=X_train_imputed_lst[num_cv_][num_imputation],
                    X_test=X_test_imputed_lst[num_cv_][num_imputation],
                )
                for num_imputation in range(self.num_imputations)
            ]
            for num_cv_, _ in enumerate(pipelines)
        ]

        for num_cv_, fold_results in enumerate(results):
            for num_imputation, (
                shap_values_template,
                base_values_template,
                X_test_scaled,
                test_idx,
                _,
                _,
                feature_indices,
            ) in enumerate(fold_results):
                if (
                    "feature_selection"
                    in pipelines[num_cv_][num_imputation].named_steps
                ):
                    rep_shap_values[
                        np.ix_(test_idx, feature_indices, [num_imputation])
                    ] += shap_values_template.astype(np.float32)[..., np.newaxis]
                    rep_base_values[
                        test_idx, num_imputation
                    ] += base_values_template.flatten().astype(np.float32)
                    rep_data[
                        np.ix_(test_idx, feature_indices, [num_imputation])
                    ] += X_test_scaled.astype(np.float32).to_numpy()[..., np.newaxis]

                else:
                    rep_shap_values[
                        test_idx, :, num_imputation
                    ] += shap_values_template.astype(np.float32)
                    rep_base_values[
                        test_idx, num_imputation
                    ] += base_values_template.flatten().astype(np.float32)
                    rep_data[test_idx, :, num_imputation] += X_test_scaled.astype(
                        np.float32
                    )

        # We need to divide the base values by num_outer_cv, because we get base_values in every outer fold
        rep_base_values = rep_base_values / self.num_outer_cv

        if (
            self.model_name == "randomforestregressor"
            and self.cfg_analysis["shap_ia_values"]["comp_shap_ia_values"]
        ):
            ia_test_shap_values = np.zeros(
                (X.shape[0], self.num_combos, self.num_imputations), dtype=np.float32
            )
            ia_test_base_values = np.zeros(
                (X.shape[0], self.num_imputations), dtype=np.float32
            )

            for num_cv_, fold_results in enumerate(results):
                for num_imputation, (
                    _,
                    _,
                    _,
                    test_idx,
                    shap_ia_values_arr,
                    shap_ia_base_values_arr,
                    _,
                ) in enumerate(fold_results):
                    ia_test_shap_values[
                        test_idx, :, num_imputation
                    ] += shap_ia_values_arr
                    ia_test_base_values[
                        test_idx, num_imputation
                    ] += shap_ia_base_values_arr

            ia_test_shap_values = ia_test_shap_values.mean(
                axis=2
            )  # Shape: (test_idx, features)
            ia_test_base_values = ia_test_base_values.mean(axis=1)  # Shape: (test_idx)

        else:
            ia_test_shap_values = None
            ia_test_base_values = None

        return (
            rep_shap_values,
            rep_base_values,
            rep_data,
            ia_test_shap_values,  # already mean across imps
            ia_test_base_values,  # already mean across imps
        )

    def create_index_mappings(
        self, X: pd.DataFrame
    ) -> tuple[dict[int, tuple[int, int]], dict[int, str], int]:
        """
        Creates mappings for feature combinations and indices for efficient SHAP interaction value storage.

        This function generates:
        - A mapping between feature combinations (based on their indices) and unique numerical indices.
        - A mapping between feature indices and feature names, as the SHAP-IQ explainer returns only feature indices.
        - The total number of feature combinations.

        Args:
            X: DataFrame containing the feature set, where columns represent features.

        Returns:
            tuple: A tuple containing:
                - combination_to_index (dict): Mapping of unique numerical indices to feature index combinations.
                - feature_to_index (dict): Mapping of feature indices to feature names.
                - num_combinations (int): Total number of feature combinations generated.
        """
        min_order_shap = self.cfg_analysis["shap_ia_values"]["min_order"]
        max_order_shap = self.cfg_analysis["shap_ia_values"]["max_order"]

        num_features = X.shape[1]
        feature_indices = range(num_features)
        feature_names = X.columns

        feature_to_index = {
            feature_idx: feature
            for feature_idx, feature in zip(feature_indices, feature_names)
        }

        combinations = []
        for r in range(min_order_shap, max_order_shap + 1):
            combinations.extend(itertools.combinations(feature_indices, r))

        combination_to_index = {
            idx: combination for idx, combination in enumerate(combinations)
        }

        return combination_to_index, feature_to_index, len(combinations)

    def repeated_nested_cv(self, comm: object = None) -> None:
        """
        Executes a repeated nested cross-validation procedure.

        This method performs repeated nested cross-validation with optional parallelization using mpi4py.
        The primary goal is to evaluate and store machine learning model performance, SHAP values,
        and SHAP interaction values across multiple repetitions and folds, while ensuring scalability
        for large datasets and complex models.

        Note: In the final analysis, we did not use mpi4py.

        **Workflow**:
        1. **MPI Parallelization**:
           - If `comm` is provided, the repetitions are distributed across MPI processes. Each process
             handles a subset of repetitions.
           - If `comm` is not provided, the repetitions are either run sequentially or split by jobs
             when `self.split_reps` is enabled.

        2. **Repetition Handling**:
           - For each repetition:
             - Compute nested cross-validation (`nested_cv`) to obtain:
               - Model performance metrics.
               - Predictions vs. true values.
               - Best model parameters.
               - SHAP values for feature importance.
               - (Optional) SHAP interaction values.
             - Save large arrays (e.g., SHAP interaction values) to disk for efficiency.
             - Log hashes of the input data for consistency checks.

        3. **Result Aggregation**:
           - If MPI is used:
             - Each process gathers its results using `comm.gather`.
             - On rank 0 (master process), results from all processes are merged.
           - If MPI is not used:
             - Results are directly aggregated without additional communication.

        4. **Final Result Processing**:
           - The aggregated results are stored in class attributes:
             - `self.best_models`: Best models for each repetition.
             - `self.shap_results`: SHAP values and related data for each repetition.
             - `self.repeated_nested_scores`: Performance scores for each repetition.
             - `self.pred_vs_true`: Predictions vs. true values for each repetition.
             - (Optional) SHAP interaction values are loaded and stored if enabled.

        Args:
            comm: (Optional) MPI communicator object. Used for parallelizing repetitions
                  across multiple processes. If `None`, no MPI parallelization is applied.

        Returns:
            None: All results are stored in class attributes for further analysis.

        Raises:
            Exception: If there are missing repetitions in the aggregated results or issues
                       with file I/O for SHAP interaction values.
        """
        if comm:
            rank = comm.Get_rank()
            size = comm.Get_size()
        else:
            rank = 0
            size = 1

        print(f"Process {rank} of {size} is running")
        self.logger.log(f"Process {rank} of {size} is running")

        X = self.X.copy()
        y = self.y.copy()
        fix_random_state = self.cfg_analysis["random_state"]

        if comm:
            all_reps = list(range(self.num_reps))
            my_reps = all_reps[rank::size]
            print(f"    Rank {rank}: Handling repetitions {my_reps}")
            self.logger.log(f"    [Rank {rank}] Handling repetitions {my_reps}")

        else:
            if self.rep is not None:
                my_reps = [self.rep]
            else:
                my_reps = list(range(self.num_reps))
            print(f"    Running repetitions {my_reps}")
            self.logger.log(f"    Running repetitions {my_reps}")

        results = []
        results_file_paths = []
        for rep in my_reps:
            print(f"    Rank {rank}: Starting repetition {rep}")
            self.logger.log(f"    [Rank {rank}] Starting repetition {rep}")
            data_hash = hashlib.md5(self.X.to_csv().encode()).hexdigest()
            print(f"    Rank {rank} data hash: {data_hash}")
            self.logger.log(f"    Rank {rank} data hash: {data_hash}")

            dynamic_rs = fix_random_state + rep
            (
                nested_scores_rep,
                pred_vs_true_rep,
                ml_model_params,
                rep_shap_values,
                rep_base_values,
                rep_data,
                rep_ia_values,
                rep_ia_base_values,
            ) = self.nested_cv(
                rep=rep,
                X=X,
                y=y,
                fix_rs=fix_random_state,
                dynamic_rs=dynamic_rs,
            )

            # Unpack result and exclude large arrays
            result_without_large_arrays = (
                nested_scores_rep,
                pred_vs_true_rep,
                ml_model_params,
                rep_shap_values,
                rep_base_values,
                rep_data,
            )
            results.append((rep, result_without_large_arrays))
            shap_val_dct = {
                "shap_values": rep_shap_values,
                "base_values": rep_base_values,
                "data": rep_data,
                "feature_names": self.X.columns.tolist(),
            }

            print(f"    [Rank {rank}] Finished repetition {rep}")
            self.logger.log(f"    Rank {rank}: Finished repetition {rep}")

            if self.cfg_analysis["shap_ia_values"]["comp_shap_ia_values"]:
                file_name_ia_values = os.path.join(
                    self.spec_output_path, f"shap_ia_values_rep_{rep}.pkl"
                )
                file_name_ia_base_values = os.path.join(
                    self.spec_output_path, f"shap_ia_base_values_rep_{rep}.pkl"
                )

                with open(file_name_ia_values, "wb") as f:
                    pickle.dump(rep_ia_values, f)
                with open(file_name_ia_base_values, "wb") as f:
                    pickle.dump(rep_ia_base_values, f)

                results_file_paths.append(
                    (rep, file_name_ia_values, file_name_ia_base_values)
                )
                ia_values_mappings = {
                    "combo_index_mapping": self.combo_index_mapping.copy(),
                    "feature_index_mapping": self.feature_index_mapping.copy(),
                    "num_combos": self.num_combos,
                }
                file_name_ia_values_mappings = os.path.join(
                    self.spec_output_path, f"ia_values_mappings_rep_{rep}.pkl"
                )

                with open(file_name_ia_values_mappings, "wb") as f:
                    pickle.dump(ia_values_mappings, f)

            best_models_file = os.path.join(
                self.spec_output_path, f"best_models_rep_{rep}.pkl"
            )
            with open(best_models_file, "wb") as f:
                pickle.dump(ml_model_params, f)

            if self.split_reps:
                nested_scores_file = os.path.join(
                    self.spec_output_path, f"cv_results_rep_{rep}.json"
                )
                with open(nested_scores_file, "w") as file:
                    json.dump(nested_scores_rep, file, indent=4)

                pred_vs_true_file = os.path.join(
                    self.spec_output_path, f"pred_vs_true_rep_{rep}.json"
                )
                with open(pred_vs_true_file, "w") as file:
                    json.dump(pred_vs_true_rep, file, indent=4)

                shap_values_file = os.path.join(
                    self.spec_output_path, f"shap_values_rep_{rep}.pkl"
                )
                with open(shap_values_file, "wb") as f:
                    pickle.dump(shap_val_dct, f)

        if comm:
            all_results = comm.gather((results, results_file_paths), root=0)
            if rank == 0:
                final_results = []
                all_file_paths = []
                for res, paths in all_results:
                    final_results.extend(res)
                    all_file_paths.extend(paths)
                print(f"  [Rank {rank}] Collected all results")
                self.logger.log(f"  [Rank {rank}] Collected all results")

        else:
            final_results = results
            all_file_paths = results_file_paths

        if not comm or (comm and rank == 0):
            for rep, (
                nested_scores_rep,
                pred_vs_true_rep,
                best_models,
                rep_shap_values,
                rep_base_values,
                rep_data,
            ) in final_results:
                print(f"Processing rep {rep}")
                self.best_models[f"rep_{rep}"] = best_models
                self.shap_results["shap_values"][f"rep_{rep}"] = rep_shap_values
                self.shap_results["base_values"][f"rep_{rep}"] = rep_base_values
                self.shap_results["data"][f"rep_{rep}"] = rep_data
                self.repeated_nested_scores[f"rep_{rep}"] = nested_scores_rep
                self.pred_vs_true[f"rep_{rep}"] = pred_vs_true_rep

            if self.cfg_analysis["shap_ia_values"]["comp_shap_ia_values"]:
                for rep, ia_values_path, ia_base_values_path in all_file_paths:
                    with open(ia_values_path, "rb") as f:
                        rep_shap_ia_values_test = pickle.load(f)
                    with open(ia_base_values_path, "rb") as f:
                        rep_shap_ia_base_values = pickle.load(f)

                    self.shap_ia_results["shap_ia_values"][
                        f"rep_{rep}"
                    ] = rep_shap_ia_values_test
                    self.shap_ia_results["base_values"][
                        f"rep_{rep}"
                    ] = rep_shap_ia_base_values

            try:
                for rep in range(len(self.repeated_nested_scores)):
                    print(
                        f"scores for rep {rep}: ",
                        self.repeated_nested_scores[f"rep_{rep}"],
                    )

            except Exception as e:
                print("I do not contain all repetitions")
                self.logger.log("I do not contain all repetitions")

    def store_analysis_results(self) -> None:
        """
        Stores prediction results, SHAP values, and SHAP interaction values to files.

        This function saves analysis results (e.g., performance metrics, predictions, SHAP values)
        into JSON or pickle files within the specified output directory. Linear model coefficients
        are also saved if applicable.
        """
        if self.rank == 0 and not self.split_reps:
            os.makedirs(self.spec_output_path, exist_ok=True)
            print(self.spec_output_path)
            rep = self.rep if self.rep is not None else "all"

            cv_results_filename = os.path.join(
                self.spec_output_path, f"{self.performance_name}_rep_{rep}.json"
            )
            with open(cv_results_filename, "w") as file:
                json.dump(self.repeated_nested_scores, file, indent=4)

            pred_vs_true_filename = os.path.join(
                self.spec_output_path, f"pred_vs_true_rep_{rep}.json"
            )
            with open(pred_vs_true_filename, "w") as file:
                json.dump(self.pred_vs_true, file, indent=4)

            self.shap_results["feature_names"] = self.X.columns.tolist()
            shap_values_filename = os.path.join(
                self.spec_output_path, f"{self.shap_value_name}_rep_{rep}.pkl"
            )
            with open(shap_values_filename, "wb") as f:
                pickle.dump(self.shap_results, f)

            if self.lin_model_coefs:
                lin_model_coefs_filename = os.path.join(
                    self.spec_output_path, f"{self.lin_model_coefs_name}_rep_{rep}.json"
                )
                with open(lin_model_coefs_filename, "w") as file:
                    json.dump(self.lin_model_coefs, file, indent=4)

    def get_average_coefficients(self) -> None:
        """
        Computes and retrieves average coefficients for linear models.

        This method is implemented in subclasses specific to linear models.
        It is used to extract coefficients of the predictors across cross-validation folds.
        """
        pass

    def process_all_shap_ia_values(self) -> None:
        """
        Aggregates SHAP interaction values.

        This method is implemented in the `RandomForestRegressor` subclass to process
        SHAP interaction values across repetitions and imputations.
        """
        pass

    def calculate_shap_values(
        self,
        X_train_scaled: pd.DataFrame,
        X_test_scaled: pd.DataFrame,
        pipeline: Pipeline,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes SHAP values for the test dataset using a tree-based or linear explainer.

        This method calculates SHAP values for test samples, using the appropriate SHAP explainer
        based on the model type (e.g., tree-based models or elastic net). It also allows parallelized
        computation of SHAP values for subsets of the data.

        Args:
            X_train_scaled: Scaled training dataset used as background data for SHAP computation.
            X_test_scaled: Scaled test dataset for which SHAP values are calculated.
            pipeline: Sklearn pipeline containing preprocessing steps and the fitted model.

        Returns:
            tuple:
                - np.ndarray: SHAP values for the test set (n_samples x n_features).
                - np.ndarray: Base values associated with the SHAP values.
        """
        if self.model_name == "elasticnet":
            n_samples = len(X_train_scaled)
            explainer = shap.LinearExplainer(
                model=pipeline.named_steps["model"].regressor_,
                masker=X_train_scaled,
                nsamples=n_samples,
            )

        elif self.model_name == "randomforestregressor":
            explainer = shap.explainers.Tree(
                model=pipeline.named_steps["model"].regressor_,
                feature_perturbation="tree_path_dependent",
            )

        else:
            raise ValueError(f"Model {self.model_name} not implemented")

        n_jobs = self.cfg_analysis["parallelize"]["shap_n_jobs"]
        chunk_size = X_test_scaled.shape[0] // n_jobs + (
            X_test_scaled.shape[0] % n_jobs > 0
        )
        print("chunk_size:", chunk_size)

        results = Parallel(
            n_jobs=n_jobs,
            backend=self.joblib_backend,
            verbose=3,
        )(
            delayed(self.calculate_shap_for_chunk)(
                explainer, X_test_scaled[i : i + chunk_size]
            )
            for i in range(0, X_test_scaled.shape[0], chunk_size)
        )

        shap_values_lst, base_values_lst = zip(*results)
        shap_values_array = np.vstack(shap_values_lst)
        base_values_array = np.concatenate(base_values_lst)

        return shap_values_array, base_values_array

    def calculate_shap_for_chunk(
        self,
        explainer: Union[shap.explainers.Tree, shap.explainers.LinearExplainer],
        X_subset: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates SHAP values for a subset of data to enable parallel computation.

        Args:
            explainer: SHAP explainer object (e.g., `shap.TreeExplainer`).
            X_subset: Subset of the test dataset for which SHAP values are computed.

        Returns:
            tuple:
                - np.ndarray: SHAP values for the data subset.
                - np.ndarray: Base values associated with the SHAP values.
        """
        explanation = explainer(X_subset)
        shap_values = explanation.values
        base_values = explanation.base_values
        return shap_values, base_values

    @staticmethod
    def spearman_corr(y_true_raw: pd.Series, y_pred_raw: pd.Series) -> float:
        """
        Calculates Spearman's rank correlation coefficient between true and predicted values.

        Args:
            y_true_raw: True target values.
            y_pred_raw: Predicted target values.

        Returns:
            float: Spearman's rank correlation coefficient (rho).
        """
        rank_corr, _ = spearmanr(y_true_raw, y_pred_raw)
        return rank_corr

    @staticmethod
    def pearson_corr(y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Calculates Pearson's correlation coefficient between true and predicted values.

        Args:
            y_true: True target values.
            y_pred: Predicted target values.

        Returns:
            float: Pearson's correlation coefficient.
        """
        return pearsonr(y_true, y_pred)[0]

    def log_thread(self) -> None:
        """
        Logs the current process ID, thread name, and calling method name.

        This is useful for debugging and tracking parallel processing activities.
        """
        process_id = os.getpid()
        thread_name = threading.current_thread().name

        current_frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(current_frame, 2)
        method_name = caller_frame[1].function

        print(
            f"        Executing {method_name} using Process ID: {process_id}, Thread: {thread_name}"
        )
        self.logger.log(
            f"        Executing {method_name} using Process ID: {process_id}, Thread: {thread_name}"
        )
