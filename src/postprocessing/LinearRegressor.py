import os
from typing import Union

import pandas as pd
import statsmodels.api as sm
from sklearn import clone
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import RegressionResults

from src.analysis.Imputer import Imputer
from src.utils.DataSaver import DataSaver
from src.utils.DataSelector import DataSelector
from src.utils.Logger import Logger
from src.utils.utilfuncs import apply_name_mapping, format_p_values, NestedDict


class LinearRegressor:
    """
    Computes data-driven linear regression models with the best features for a given analysis setting
    using statsmodels. The class supports data preprocessing, feature selection, missing value imputation,
    and regression model creation. Additionally, it provides functionality to create regression coefficient
    tables in Excel format.

    Attributes:
        cfg_preprocessing (NestedDict): Yaml config specifying details on preprocessing (e.g., scales, items).
        cfg_analysis (NestedDict): Yaml config specifying details on the ML analysis (e.g., CV, models).
        cfg_postprocessing (NestedDict): Yaml config specifying details on postprocessing (e.g., tables, plots).
        name_mapping (NestedDict): Mapping of feature names for presentation purposes.
        df (pd.DataFrame): Input DataFrame containing the dataset for analysis.
        feature_combination (str): Name of the feature combination used for regression.
        crit (str): The criterion variable for the regression analysis.
        samples_to_include (Union[str, list[str]]): Samples to include in the analysis.
        cv_shap_results_path (str): Base path of the main results to construct the path to the x best features.
        model_for_features (str): Model name used to select the best features.
        meta_vars (list[str]): List of metadata variables to exclude from analysis.
        num_features (int): Number of features to select for regression analysis. Default is 6.
        datasets_included (List[str] or None): List of datasets included in the analysis. Populated during runtime.
        X (Optional[pd.DataFrame]): Selected feature set for regression. Populated during runtime.
        y (Optional[pd.Series]): Criterion variable for regression. Populated during runtime.
        rows_dropped_crit_na (Optional[int]): Number of rows dropped due to missing criterion values. Populated during runtime.
        meta_vars (list[str]): List of metadata variables to exclude from analysis.
        data_saver (DataSaver): Instance of DataSaver for saving outputs (e.g., regression tables).
        dataselector (DataSelector): Instance of DataSelector for selecting samples and features.
        logger (Logger): Logger instance for tracking events and outputs of the imputation procedure.
        country_grouping_col (str): Column name for grouping data by country, as specified in the configuration.
        years_col (str): Column name for grouping data by year, as specified in the configuration.
    """

    def __init__(
        self,
        cfg_preprocessing: NestedDict,
        cfg_analysis: NestedDict,
        cfg_postprocessing: NestedDict,
        name_mapping: NestedDict,
        df: pd.DataFrame,
        feature_combination: str,
        crit: str,
        samples_to_include: Union[str, list[str]],
        cv_shap_results_path: str,
        model_for_features: str,
        meta_vars: list[str],
        num_features: int = 6,
    ) -> None:
        """
        Initializes the LinearRegressor class with preprocessing, analysis, and regression settings.

        Args:
            cfg_preprocessing: Configuration dictionary for preprocessing settings.
            cfg_analysis: Configuration dictionary for analysis settings.
            cfg_postprocessing: Yaml config specifying details on postprocessing (e.g., tables, plots).
            name_mapping: Mapping of feature names for presentation purposes.
            df: Input DataFrame containing the dataset for analysis.
            feature_combination: Name of the feature combination used for regression.
            crit: The criterion variable for the regression analysis.
            samples_to_include: Samples to include in the analysis.
            cv_shap_results_path: Path to SHAP results for cross-validation.
            model_for_features: Model name used to select the best features.
            meta_vars: List of metadata variables to exclude from analysis.
            num_features: Number of features to select for regression analysis. Default is 6.
        """
        self.cfg_preprocessing = cfg_preprocessing
        self.cfg_analysis = cfg_analysis
        self.cfg_postprocessing = cfg_postprocessing
        self.name_mapping = name_mapping

        self.cv_shap_results_path = cv_shap_results_path
        self.df = df

        self.feature_combination = feature_combination
        self.crit = crit
        self.samples_to_include = samples_to_include
        self.model_for_features = model_for_features  # RFR
        self.num_features = num_features

        self.datasets_included = None
        self.X = None
        self.y = None
        self.rows_dropped_crit_na = None
        self.meta_vars = meta_vars

        self.data_saver = DataSaver()

        self.dataselector = DataSelector(
            self.cfg_analysis,
            self.df,
            self.feature_combination,
            self.crit,
            self.samples_to_include,
            self.meta_vars,
        )

        self.logger = Logger(
            log_file=self.cfg_preprocessing["general"]["log_name"],
        )

        self.country_grouping_col = self.cfg_analysis["imputation"][
            "country_grouping_col"
        ]
        self.years_col = self.cfg_analysis["imputation"]["years_col"]

    @property
    def imputer(self) -> Imputer:
        """
        Creates and returns an instance of the Imputer class configured for the current analysis.

        The Imputer is initialized with settings derived from cfg_analysis and cfg_postprocessing.
        - If parameters are the same as in the ML-based analysis, we use the params for cfg_analysis
        - If parameters differ (e.g., num_imputations), we use the params for cfg_postprocessing

        Returns:
            Imputer: An Imputer instance initialized with the specified settings.
        """
        postprocessing_imputation_params = self.cfg_postprocessing[
            "calculate_exp_lin_models"
        ]["imputation"]
        return Imputer(
            logger=self.logger,
            model=postprocessing_imputation_params["model"],
            fix_rs=self.cfg_analysis["random_state"],
            max_iter=1,  # self.cfg_analysis["imputation"]["max_iter"],
            num_imputations=postprocessing_imputation_params["num_imputations"],
            conv_thresh=self.cfg_analysis["imputation"]["conv_thresh"],
            tree_max_depth=self.cfg_analysis["imputation"]["tree_max_depth"],
            percentage_of_features=postprocessing_imputation_params[
                "percentage_of_features"
            ],
            n_features_thresh=postprocessing_imputation_params["n_features_thresh"],
            sample_posterior=self.cfg_analysis["imputation"]["sample_posterior"],
            pmm_k=self.cfg_analysis["imputation"]["pmm_k"],
            country_group_by=self.country_grouping_col,
            years_col=self.years_col,
        )

    def get_regression_data(self) -> None:
        """
        Prepares the input and target data for explanatory regression analysis.
        - Selects and stores the target variable `y`.
        - Selects initial feature set `X` from the data.
        - Applies feature selection based on SHAP values and stores the refined feature set.
        """
        self.dataselector.select_samples()
        X = self.dataselector.select_features()
        self.y = self.dataselector.select_criterion()
        self.X = self.dataselector.select_best_features(
            df=X,
            root_path=self.cv_shap_results_path,
            model=self.model_for_features,
            num_features=self.num_features,
        )

    def compute_regression_models(self) -> RegressionResults:
        """
        Fits an Ordinary Least Squares (OLS) regression model using the selected features and criterion variable.

        This method performs the following steps:
        1. Handles missing values by imputing them with the custom linear imputer used in the ML-based analysis.
        2. Standardizes the feature set to have zero mean and unit variance.
        3. Fits an OLS regression model with the standardized features and an added intercept term.


        Returns:
            sm.OLS: A fitted statsmodels OLS regression results object containing:
                - Model parameters (coefficients).
                - Statistical test results.
                - Summary statistics including R², adjusted R², and p-values.
        """
        X = self.X.copy()

        imputer = clone(self.imputer)
        imputer.fit(X, num_imputation=1)  # this may take a while
        X_imputed = imputer.transform(X=X)

        self.y = self.y.loc[X_imputed.index]

        X_imputed = X_imputed.drop(columns=self.meta_vars, errors="ignore")

        scaler = StandardScaler()
        X_imputed_scaled = pd.DataFrame(
            scaler.fit_transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index,
        )

        X_imputed_scaled_intercept = sm.add_constant(X_imputed_scaled)
        model = sm.OLS(self.y, X_imputed_scaled_intercept).fit()

        print("-----")
        print(f"Regression results for {self.feature_combination} - {self.crit}")
        print(model.summary())

        return model

    def create_coefficients_table(
        self,
        feature_combination: str,
        model: RegressionResults,
        output_dir: str,
    ) -> None:
        """
        Creates a regression table from a fitted statsmodels object and saves it as an Excel file.

        The table includes:
        - Predictors (independent variables)
        - Estimates (coefficients)
        - Confidence Intervals (CI)
        - P-values
        - Model summary statistics (e.g., R² and Adjusted R², number of observations)

        Args:
            feature_combination (str): The name of the feature combination for the table title.
            model (sm.OLS): A fitted statsmodels regression object.
            output_dir (str): Directory to save the resulting Excel file.
        """
        coefficients = model.params
        conf_int = model.conf_int()
        p_values = model.pvalues

        lin_model_cfg = self.cfg_postprocessing["calculate_exp_lin_models"]
        decimals = lin_model_cfg["decimals"]
        store = lin_model_cfg["store"]
        base_filename = lin_model_cfg["base_filename"]

        regression_table = pd.DataFrame(
            {
                "Predictors": coefficients.index,
                "Estimates": coefficients.values,
                "CI": conf_int.apply(
                    lambda x: f"[{x[0]:.{decimals}f}, {x[1]:.{decimals}f}]", axis=1
                ),
                "p": p_values.values,
            }
        )

        regression_table["Estimates"] = regression_table["Estimates"].apply(
            lambda x: f"{x:.{decimals}f}"
        )
        regression_table["p"] = format_p_values(regression_table["p"].tolist())
        regression_table["Predictors"] = apply_name_mapping(
            features=list(regression_table["Predictors"]),
            name_mapping=self.name_mapping,
            prefix=True,
        )

        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        observations = model.nobs

        footer_rows = pd.DataFrame(
            {
                "Predictors": ["Observations", "R² / R² adjusted"],
                "Estimates": [
                    f"{observations}",
                    f"{r_squared:.{decimals}f} / {adj_r_squared:.{decimals}f}",
                ],
                "CI": [None, None],
                "p": [None, None],
            }
        )
        final_table = pd.concat([regression_table, footer_rows], ignore_index=True)

        if store:
            output_path = os.path.join(
                output_dir, f"{base_filename}_{feature_combination}.xlsx"
            )
            self.data_saver.save_excel(
                df=final_table, output_path=output_path, index=False
            )
