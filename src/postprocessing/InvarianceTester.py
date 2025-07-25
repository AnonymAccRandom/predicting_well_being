import os
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import rpy2.robjects as ro

ro.r('install.packages("lavaan.mi", repos="https://cran.rstudio.com")')

from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

semtools = importr("semTools")
lavaan = importr("lavaan")

def _df_to_r(df: pd.DataFrame):
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)


from src.utils.DataLoader import DataLoader
from src.utils.DataSaver import DataSaver
from src.utils.utilfuncs import NestedDict

class InvarianceTester:
    """
    Provides tools for evaluating measurement invariance across countries and datasets.

    This class implements configural, metric, and scalar invariance testing using multi-group confirmatory factor
    analysis (CFA) via lavaan (R).

    Key functionalities include:
    - Loading and organizing preprocessed item-level data across multiple datasets
    - Running CFA-based invariance tests across both countries (within a unified dataset) and across datasets
    - Returning tidy outputs with model fit statistics (RMSEA, CFI, χ²-diff) and flags for each level of invariance

    Methods assume that all required dataset-specific configuration is provided in the form of nested dictionaries,
    parsed from YAML or equivalent config files.
    """

    def __init__(self, cfg_preprocessing: NestedDict,
                 cfg_postprocessing: NestedDict,
                 name_mapping: NestedDict) -> None:
        """
        Initializes the InvarianceTester with configuration settings and utility objects.

        - Stores configuration dictionaries for preprocessing and postprocessing
        - Extracts invariance-specific settings from the postprocessing config
        - Instantiates helper objects for data loading and saving
        - Prepares an internal attribute to store the loaded data

        Args:
            cfg_preprocessing: Nested dictionary containing paths and settings for data input and filtering.
            cfg_postprocessing: Nested dictionary containing construct definitions, mappings, and invariance test parameters.
            name_mapping: Nested dictionary containing names and mappings for personality constructs.
        """
        self.cfg_preprocessing = cfg_preprocessing
        self.cfg_postprocessing = cfg_postprocessing
        self.name_mapping = name_mapping
        self.cfg_invariance = self.cfg_postprocessing["measurement_invariance"]

        self.data_loader = DataLoader()
        self.data_saver = DataSaver()
        self.data_dict = None

    def load_data(self) -> None:
        """
        Loads the Trait DFs from preprocessed data files.

        - Scans the specified directory for files that start with 'raw_trait_df'
        - Extracts the source identifier from each file name (e.g., 'cocoesm' from 'raw_trait_df_cocoesm.parquet')
        - Loads each file into a DataFrame and stores it in a dictionary with the extracted identifier as key

        Sets:
            dict[str, pd.DataFrame]: A dictionary where keys are source identifiers and values are the corresponding DataFrames.
        """
        path_to_preprocessed_data = self.cfg_preprocessing["general"]["path_to_preprocessed_data"]
        data_dict = {"trait_features": {}, "trait_wb_items": {}, "state_wb_items": {}}

        for file in os.listdir(path_to_preprocessed_data):
            if file.startswith("wb_items") and "state_wb_items" in self.cfg_invariance["general"]["to_include"]:
                dataset = file.split("_")[-1]
                data_dict["state_wb_items"][dataset] = self.data_loader.read_pkl(os.path.join(path_to_preprocessed_data, file))
            if file.startswith("raw_trait_df") and "trait_features" in self.cfg_invariance["general"]["to_include"]:
                dataset = file.split("_")[-1]
                data_dict["trait_features"][dataset] = self.data_loader.read_pkl(os.path.join(path_to_preprocessed_data, file))
            if file.startswith("trait_wb_items") and "trait_wb_items" in self.cfg_invariance["general"]["to_include"]:
                dataset = file.split("_")[-1]
                data_dict["trait_wb_items"][dataset] = self.data_loader.read_pkl(os.path.join(path_to_preprocessed_data, file))

        data_dict = {k: v for k, v in data_dict.items() if v}
        setattr(self, "data_dict", data_dict)

    def test_invariance_across_countries(self) -> NestedDict:
        """
        Tests measurement invariance of constructs across countries using the CoCoESM dataset.

        For each construct block (e.g., trait features, well-being items), this method evaluates whether
        constructs are measured equivalently across countries by applying configural, metric, and scalar
        invariance testing using multi-group CFA.

        The analysis:
        - Focuses exclusively on the harmonized CoCoESM dataset
        - Filters out countries with insufficient sample size based on thresholds defined in the config
        - Uses model fit criteria (RMSEA, CFI) to assess invariance levels
        - Returns a nested dictionary with results per variable block and minimum country sample size

        Returns:
            dict[str, dict[int, pd.DataFrame]]:
                Outer keys are variable blocks (e.g., "trait_features").
                Inner keys are the minimum number of participants per country used for filtering.
                Each value is a DataFrame summarizing RMSEA, CFI, and invariance flags per construct.
        """
        out_tables = {}
        invariance_cfg_country = self.cfg_invariance["across_countries"]

        for type_of_var, vals in self.data_dict.items():

            # var_type for mapping
            if type_of_var == "trait_features":
                var_type = "pl"
            else:
                var_type = "crit"

            # Iterate over different number of min participants per country
            out_inner_tables = {}
            for min_samples_per_country in self.cfg_invariance["general"]["min_samples_per_country"]:
                current_df = deepcopy(vals["cocoesm"])

                df_filtered = current_df[current_df["country"].map(current_df["country"].value_counts())
                                 >= min_samples_per_country]

                merged_dict = self._build_cocoesm_dfs(
                    construct_cfg=invariance_cfg_country[type_of_var],
                    df=df_filtered,
                )

                block_rows = []
                for factor, df in merged_dict.items():
                    # columns minus grouping variable = substantive items
                    item_cols = [c for c in df.columns if c != "country"]
                    res = self.run_invariance(
                        df=df,
                        items=item_cols,
                        factor=factor,
                        group_col="country",
                        method_items=None,
                        var_type=var_type,
                    )
                    if res is not None:
                        block_rows.append(res)

                out_inner_tables[min_samples_per_country] = (
                    pd.concat(block_rows, axis=0, join="outer", ignore_index=True) if block_rows else pd.DataFrame()
                )
            out_tables[type_of_var] = out_inner_tables

        return out_tables

    def test_invariance_across_datasets(self) -> dict[str, pd.DataFrame]:
        """
        Tests measurement invariance of constructs across datasets.

        For each construct block (e.g., trait features, well-being items), this method evaluates whether
        constructs are measured equivalently across datasets using configural, metric, and scalar invariance
        testing via multi-group CFA.

        The analysis:
        - Focuses on constructs assessed with at least three items
        - Includes only constructs administered identically across at least five out of nine datasets (TODO Check)
        - Applies model fit criteria (RMSEA, CFI) and change indices (ΔRMSEA, ΔCFI) to evaluate invariance
        - Returns one summary table per block with model fit statistics and invariance flags for each construct

        Returns:
            dict[str, pd.DataFrame]:
                A dictionary mapping each variable block (e.g., "trait_features") to a DataFrame
                summarizing RMSEA, CFI, and invariance flags for each construct across datasets.
        """
        invariance_cfg = self.cfg_postprocessing["measurement_invariance"]["across_datasets"]
        out_tables = {}

        for type_of_var in ("trait_wb_items", "state_wb_items"):
            merged_dict = self._merge_all_dataset_dfs(type_of_var=type_of_var,
                                                      invariance_cfg=invariance_cfg[type_of_var])

            # var_type for mapping
            if type_of_var == "trait_features":
                var_type = "pl"
            else:
                var_type = "crit"

            block_rows = []
            for factor, df in merged_dict.items():
                item_cols = [c for c in df.columns if c != "dataset"]
                res = self.run_invariance(
                    df=df,
                    items=item_cols,
                    factor=factor,
                    group_col="dataset",
                    method_items=None,
                    var_type=var_type,
                )
                if res is not None:
                    block_rows.append(res)

            out_tables[type_of_var] = (
                pd.concat(block_rows, ignore_index=True) if block_rows else pd.DataFrame()
            )

        return out_tables

    def _merge_all_dataset_dfs(self, type_of_var: str, invariance_cfg: NestedDict) -> dict[str, pd.DataFrame]:
        """
        Constructs harmonized item-level DataFrames across datasets for each construct.

        For every construct defined in the config, the method aligns item columns across datasets using the
        provided renaming mappings and appends a `dataset` column to track data provenance. All datasets
        are vertically concatenated per construct to prepare data for invariance testing.

        Args:
            type_of_var: Variable block identifier (e.g., "trait_features", "trait_wb_items").
            invariance_cfg: Configuration list specifying constructs, their items, and dataset inclusion/mappings.

        Returns:
            dict[str, pd.DataFrame]: A dictionary mapping construct names to merged DataFrames.
                Each DataFrame:
                - contains only harmonized item columns in the target naming scheme
                - has identical column order across datasets
                - includes a `dataset` column indicating the source dataset
        """
        merged = {}

        for var_entry in invariance_cfg:
            var_name = var_entry["name"]
            target_items = self._get_target_items(var_entry)

            chunks = []
            for dataset, df in self.data_dict[type_of_var].items():
                if dataset in var_entry["to_include"]:
                    rename_map = self._build_renaming_mapping(var_entry, dataset)

                    raw_cols = list(rename_map.keys()) if rename_map else target_items
                    subset = df[raw_cols].rename(columns=rename_map)

                    subset = subset[target_items]  # enforce order / presence
                    subset["dataset"] = dataset  # keep provenance
                    chunks.append(subset)

            merged[var_name] = pd.concat(chunks, ignore_index=True)

        return merged

    @staticmethod
    def _build_cocoesm_dfs(
            construct_cfg: list[dict],
            df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        Builds harmonized item-level DataFrames per construct using only the CoCoESM dataset.

        Each construct-specific DataFrame includes all specified items and a `country` column for grouping.
        This method prepares data for invariance testing across countries by enforcing consistent column
        ordering and selecting only relevant items.

        Args:
            construct_cfg: A list of construct configuration dictionaries, each containing:
                           - "name": the construct label (e.g., "sincerity")
                           - "item_names": the ordered list of item column names to include
            df: The CoCoESM DataFrame from which construct-specific subsets are extracted.

        Returns:
            dict[str, pd.DataFrame]: A dictionary mapping construct names to filtered DataFrames,
            each containing the specified items and a `country` column.
        """
        merged = {}

        for entry in construct_cfg:
            factor = entry["name"]
            items = entry["item_names"]
            items += ["country"]

            # select & order columns
            subset = df[items].copy()
            merged[factor] = subset

        return merged

    def _get_target_items(self, post_cfg: NestedDict) -> list[str]:
        """
        Determines the canonical item names for a construct using the cocointernational naming scheme.

        This method resolves the harmonized item names expected across datasets based on the provided
        mapping configuration. If no explicit mapping is defined, it defaults to the item order in the
        first available dataset.

        Args:
            post_cfg: Configuration dictionary for a construct, including optional item mappings and the construct name.

        Returns:
            list[str]: Ordered list of standardized item names used for alignment across datasets.
        """
        mapping_block = post_cfg.get("mapping", {})

        if mapping_block:
            first_map = next(iter(mapping_block.values()))
            return list(first_map.values())

        first_dataset = next(iter(self.data_dict[post_cfg["name"]].values()))

        return list(first_dataset.columns)

    @staticmethod
    def _build_renaming_mapping(post_cfg: NestedDict, dataset: str) -> dict[str, str]:
        """
        Retrieves the column renaming map for a specific dataset.

        If item names differ across datasets, this method provides a mapping from original column
        names to harmonized (target) names to ensure consistency across datasets.

        Args:
            post_cfg: Configuration dictionary for a construct, including dataset-specific mappings.
            dataset: Dataset identifier (e.g., "cocout", "cocoms").

        Returns:
            dict[str, str]: Dictionary mapping original column names to target names.
                            Returns an empty dictionary if no renaming is needed.
        """
        mapping_block = post_cfg.get("mapping", {})
        return mapping_block.get(dataset, {})

    def _get_config_for_type(self,
                             var_type: str,
                             config_type: str = "preprocessing") -> list[dict]:
        """
        Return the config list for a given variable type and config type.

        Args:
            var_type: One of {"trait_features", "trait_wb_items", "state_wb_items"}
            config_type: One of {"preprocessing", "postprocessing", ...}

        Returns:
            A list of variable config dictionaries (from YAML or other config).
        """
        config_attr = f"cfg_{config_type}"
        if not hasattr(self, config_attr):
            raise AttributeError(f"No configuration found for '{config_type}'.")

        cfg = getattr(self, config_attr)

        if var_type == "trait_features":
            return cfg["person_level"]["personality"]
        elif var_type == "trait_wb_items":
            return cfg["person_level"]["criterion"]
        elif var_type == "state_wb_items":
            return cfg["esm_based"]["criterion"]
        else:
            raise NotImplementedError(f"Unknown variable type: {var_type}")

    def _get_items_and_construct(  # noqa: N802
            self,
            var_type:str,
            dataset: str,
            df: pd.DataFrame,
            var_config: dict,
            add_country: bool = False
    ) -> tuple[pd.DataFrame, str]:
        """
        Extracts a numeric item-level DataFrame and builds a one-factor CFA model string.

        The method filters and aligns item columns for a given construct and dataset. It optionally appends a
        `country` column for grouping and returns a Lavaan-style measurement model suitable for CFA or invariance testing.

        - Drops constructs with fewer than two valid items
        - Applies special correction to emotion items if relevant
        - Excludes the grouping variable from the indicator list in the model string

        Args:
            var_type: Variable block identifier (e.g., "trait_wb_items").
            dataset: Dataset identifier (e.g., "emotions").
            df: Full trait-level DataFrame.
            var_config: Configuration dictionary specifying the construct name and item definitions.
            add_country: Whether to include a `country` column in the returned DataFrame.

        Returns:
            tuple[pd.DataFrame, str]:
                - The filtered DataFrame containing only the selected items (and optionally `country`)
                - A Lavaan model string defining a one-factor measurement model for the construct
                Returns an empty DataFrame and empty string if fewer than two valid items are found.
        """
        items = var_config["item_names"].get(dataset, [])
        if dataset == "emotions" and var_type in ["trait_wb_items", "state_wb_items"]:
            items = self._correct_emotions_items(items)

        items = [i for i in items if i in df.columns]
        if add_country:
            items.append("country")

        if len(items) < 2:
            return pd.DataFrame(), ""

        tmp = df[items]

        if tmp.shape[1] < 2:
            return pd.DataFrame(), ""

        factor = var_config["name"]
        indicators = [c for c in tmp.columns if c != "country"]  # drop grouping col
        model = f"{factor} =~ " + " + ".join(["1*" + indicators[0], *indicators[1:]])

        return tmp, model

    @staticmethod
    def conduct_cfa(
            data: pd.DataFrame,
            model_str: str,
            group_col: Optional[str] = "country",
    ) -> pd.DataFrame | None:
        """
        Conducts a one-factor confirmatory factor analysis (CFA) using lavaan, separately for each group.

        For each group defined in `group_col`, this method estimates a CFA model and extracts fit statistics.
        If `group_col` is None, the analysis is performed on the entire sample. Groups with fewer than two items
        or fewer than three observations are skipped.

        Args:
            data: A wide-format DataFrame containing item columns and a grouping column.
            model_str: A Lavaan model string specifying the one-factor CFA structure.
            group_col: Name of the grouping variable. If None, a single CFA is run on the full sample.

        Returns:
            pd.DataFrame | None: A tidy DataFrame with one row per group, containing:
                - "factor": Name of the latent factor
                - "rmsea": Root mean square error of approximation
                - "cfi": Comparative fit index
                - "number_of_items": Number of items used in the model
                - "group": Group identifier
                Returns None if no model could be successfully estimated.
        """
        if not model_str:
            return None

        factor_name = model_str.split(" =~")[0].strip()
        item_cols = [c for c in data.columns if c not in (group_col,)]

        rows = []
        try:
            groups = [None] if group_col is None else data[group_col].dropna().unique()

            for grp in groups:
                df_sub = data[item_cols] if grp is None else data.loc[data[group_col] == grp, item_cols]

                # need at least 2 items & >1 row
                if df_sub.shape[1] < 2 or df_sub.shape[0] < 3:
                    continue

                r_df = _df_to_r(df_sub)
                fit = lavaan.sem(model_str, data=r_df, missing="fiml", meanstructure=True)

                rmsea, cfi = map(float, lavaan.fitmeasures(fit, ro.StrVector(("rmsea", "cfi"))))
                rows.append(
                    {
                        "factor": factor_name,
                        "rmsea": rmsea,
                        "cfi": cfi,
                        "number_of_items": df_sub.shape[1],
                        "group": "all" if grp is None else grp,
                    }
                )

        except Exception as err:
            print(f"[lavaan error] {err}")
            return None

        if not rows:
            return None

        return pd.DataFrame(rows).round(3)

    @staticmethod
    def _fit_lavaan(model: str, r_df, group: str) -> object:
        """
        Fits a multi-group CFA model in lavaan using FIML estimation.

        This is a convenience wrapper around `lavaan.sem` with common settings for
        measurement invariance testing, including missing data handling and mean structure estimation.

        Args:
            model: Lavaan model string specifying the CFA structure.
            r_df: R-compatible DataFrame (already converted from pandas).
            group: Name of the grouping variable in the dataset.

        Returns:
            An R lavaan model fit object.
        """
        return lavaan.sem(
            model,
            data=r_df,
            missing="fiml",
            group=group,
            meanstructure=True
        )

    def run_invariance(
            self,
            df: pd.DataFrame,
            items: list[str],
            factor: str,
            group_col: str,
            method_items: list[str] | None = None,
            var_type: str = "pl"
    ) -> pd.DataFrame | None:
        """
        Performs multi-group CFA to test configural, metric, and scalar invariance.

        For a given construct, this method estimates three nested models (base, weak, and strong) using lavaan.
        It optionally includes a method factor to account for method effects. Model fits are compared using
        standard fit indices (RMSEA, CFI, χ²-diff) and returned in a tidy DataFrame.

        The method:
        - Filters invalid or duplicated columns
        - Builds a model string (with optional method factor)
        - Converts data to R format and runs three levels of invariance models
        - Compares model fits and adds invariance flags

        Args:
            df: Input DataFrame containing item and group columns.
            items: List of item column names to include in the factor model.
            factor: Name of the latent factor.
            group_col: Name of the column indicating group membership (e.g., "country" or "dataset").
            method_items: Optional list of item names to define an orthogonal method factor.
            var_type: "pl" or "crit" (for correct name mapping)

        Returns:
            pd.DataFrame | None: A tidy DataFrame summarizing fit indices and invariance results,
            or None if model fitting was not possible due to insufficient data.
        """
        df = df.loc[:, ~df.T.duplicated()]
        items = [c for c in items if c != group_col]
        if len(items) < 2 or df[group_col].nunique() < 2:
            return None

        model = self._build_invariance_model(factor, items, method_items)
        r_df = _df_to_r(df[[*items, group_col]])
        fit_base = self._fit_multi_group(model, r_df, group_col)

        fit_weak = self._fit_multi_group(model, r_df, group_col, equal=["loadings"])
        fit_strong = self._fit_multi_group(model, r_df, group_col,
                                           equal=["loadings", "intercepts"])
        try:
            out = self._compare_invariance_fits(fit_base, fit_weak, fit_strong, n_items=len(items))

        except:  # Any lavaan error, e.g. non-convergence
            return pd.DataFrame()
            # format output df

        out = out.round(3)
        mapped_construct = self.name_mapping[var_type][factor]
        out.insert(0, "Construct", mapped_construct)

        if group_col == "dataset":
            out.insert(2, "Number of datasets", df[group_col].nunique())
        if group_col == "country":
            out.insert(2, "Number of Countries", df[group_col].nunique())

        return out

    @staticmethod
    def _build_invariance_model(
            factor: str,
            items: list[str],
            method_items: list[str] | None = None,
    ) -> str:
        """
        Constructs a Lavaan model string for confirmatory factor analysis (CFA).

        The model defines a one-factor structure with fixed scaling (`1*item1`) and optionally includes
        an orthogonal method factor if `method_items` are provided.

        Args:
            factor: Name of the latent construct.
            items: List of item names loading on the main factor.
            method_items: Optional list of item names loading on a method factor.

        Returns:
            str: A Lavaan model string specifying the factor structure for CFA.
        """
        loadings = " + ".join(["1*" + items[0], *items[1:]])
        model = f"{factor} =~ {loadings}"
        if method_items:
            m_line = " + ".join(method_items)
            model += f"\nmethod =~ {m_line}\n{factor} ~~ 0*method"
        return model

    @staticmethod
    def _fit_multi_group(
            model: str,
            r_df,
            group_col: str,
            equal: list[str] | None = None,
    ):
        """
        Fits a multi-group CFA model in lavaan with optional equality constraints.

        This wrapper handles full-information maximum likelihood (FIML) estimation and mean structures.
        Equality constraints (e.g., loadings, intercepts) are applied if specified to test metric or scalar invariance.

        Args:
            model: Lavaan model string specifying the CFA structure.
            r_df: R-compatible DataFrame with indicators and grouping variable.
            group_col: Name of the grouping variable.
            equal: Optional list of constraints to apply across groups (e.g., ["loadings"]).

        Returns:
            An R lavaan model fit object representing the estimated multi-group CFA.
        """
        kwargs = {}
        if equal:
            kwargs["group.equal"] = ro.StrVector(equal)
        return lavaan.sem(
            model,
            data=r_df,
            group=group_col,
            missing="fiml",
            meanstructure=True,
            **kwargs,
        )

    @staticmethod
    def _compare_invariance_fits(
                                 fit_base,
                                 fit_weak,
                                 fit_strong,
                                 n_items: int,
                                 base_cfi: float = 0.90,
                                 base_rmsea: float = 0.08,
                                 delta_cfi_bound: float = -0.02,  # 0.01, 0.02
                                 delta_rmsea_bound: float = 0.03,  # 0.015, 0.03
                                 ) -> pd.DataFrame:
        """
        Compares fit indices across nested CFA models to assess measurement invariance.

        This method evaluates configural, metric, and scalar invariance by:
        - Extracting fit statistics (RMSEA, CFI) and their changes across models
        - Computing χ²-differences and associated p-values
        - Flagging invariance success based on established cutoffs

        Args:
            fit_base: Lavaan model object for the configural model.
            fit_weak: Lavaan model object with loadings constrained (metric invariance).
            fit_strong: Lavaan model object with loadings and intercepts constrained (scalar invariance).
            n_items: Number of indicators used in the model.
            base_cfi: Minimum acceptable CFI for configural fit.
            base_rmsea: Maximum acceptable RMSEA for configural fit.
            delta_cfi_bound: Minimum acceptable change in CFI for metric/scalar invariance.
            delta_rmsea_bound: Maximum acceptable change in RMSEA for metric/scalar invariance.

        Returns:
            pd.DataFrame: A summary table with:
                - Model-level statistics (RMSEA, CFI, χ², df, p-value)
                - Change indices (ΔRMSEA, ΔCFI)
                - Binary flags indicating support for configural, metric, and scalar invariance
        """
        cmp = semtools.compareFit(base=fit_base, weak=fit_weak, strong=fit_strong)

        rmsea = list(cmp.slots["fit"].rx2("rmsea"))
        cfi = list(cmp.slots["fit"].rx2("cfi"))
        delta_rmsea = [np.nan] + list(cmp.slots["fit.diff"].rx2("rmsea"))
        delta_cfi = [np.nan] + list(cmp.slots["fit.diff"].rx2("cfi"))
        dfs = list(cmp.slots["fit"].rx2("df"))

        out = pd.DataFrame(
            {
                "MI": ["configural", "metric", "scalar"],
                "Number of Items": n_items,
                "Degrees of Freedom": dfs,
                "RMSEA": rmsea,
                "delta_RMSEA": delta_rmsea,
                "CFI": cfi,
                "delta_CFI": delta_cfi,
            }
        )

        # 1) Configural: evaluate fit of the *base* model itself
        cfg_ok = (out["CFI"] >= base_cfi) & (out["RMSEA"] <= base_rmsea)
        out["Configural MI"] = np.where(out["MI"] == "configural", cfg_ok.astype(int), np.nan)

        # 2 & 3) Weak & strong: same Δ-criterion, applied to their respective rows
        delta_ok = (out["delta_CFI"] >= delta_cfi_bound) & (out["delta_RMSEA"] <= delta_rmsea_bound)
        out["Metric MI"] = np.where(out["MI"] == "metric", delta_ok.astype(int), np.nan)
        out["scalar MI"] = np.where(out["MI"] == "scalar", delta_ok.astype(int), np.nan)

        out = out.drop(columns="MI")
        return out.round(3)

    @staticmethod
    def _merge_emotions_items(df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges redundant emotion items from different sources into unified columns.

        Emotion items may be reported as pairs (e.g., `occup_relaxed`, `int_relaxed`).
        This method consolidates such pairs into a single column per emotion by taking
        the row-wise mean if both exist, or by retaining the single available value.

        Args:
            df: A DataFrame containing emotion item columns with prefixes such as "occup_" or "int_".

        Returns:
            pd.DataFrame: A new DataFrame containing only the merged emotion columns, named by emotion suffix.
        """
        suffix_map = {}
        for col in df.columns:
            if "_" in col:  # ignore unexpected names
                prefix, suffix = col.split("_", 1)
                if prefix in {"occup", "int"}:
                    suffix_map.setdefault(suffix, []).append(col)

        merged = pd.DataFrame(index=df.index)

        for suffix, cols in suffix_map.items():
            if len(cols) == 2:  # both occup_* and int_*
                merged[suffix] = df[cols].mean(axis=1, skipna=True)
            else:  # only one of the two
                merged[suffix] = df[cols[0]]

        return merged

    @staticmethod
    def _correct_emotions_items(items: list[str]) -> list[str]:
        """
        Normalizes emotion item names by removing leading prefixes.

        Strips prefixes like "occup_" or "int_" from item names, retaining only the
        suffix (e.g., "relaxed"). Ensures unique, cleaned item names while preserving order.

        Args:
            items: A list of raw item names with possible source-specific prefixes.

        Returns:
            list[str]: A list of cleaned item names without prefixes, duplicates removed.
        """
        seen = set()
        cleaned = []

        for item in items:
            suffix = item.split("_", 1)[-1]  # part after first "_"
            if suffix not in seen:
                seen.add(suffix)
                cleaned.append(suffix)

        return cleaned
