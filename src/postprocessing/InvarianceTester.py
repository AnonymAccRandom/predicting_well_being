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
print()

def _df_to_r(df: pd.DataFrame):
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)


from src.utils.DataLoader import DataLoader
from src.utils.DataSaver import DataSaver
from src.utils.utilfuncs import NestedDict



class InvarianceTester:
    """
    Class for testing measurement invariance
    """


    def __init__(self, cfg_preprocessing: NestedDict,  cfg_postprocessing: NestedDict):
        self.cfg_preprocessing = cfg_preprocessing
        self.cfg_postprocessing = cfg_postprocessing
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

    def test_invariance_across_countries(self) -> dict:
        """
        Returns:

        """
        out_tables = {}
        invariance_cfg_country = self.cfg_invariance["across_countries"]

        for type_of_var, vals in self.data_dict.items():

            # Iterate over different number of excluded samples
            out_inner_tables = {}
            for min_samples_per_country in self.cfg_invariance["general"]["min_samples_per_country"]:
                current_df = deepcopy(vals["cocoesm"])

                df_filtered = current_df[current_df["country"].map(current_df["country"].value_counts())
                                 >= min_samples_per_country]

                merged_dict = self._build_cocoesm_dfs(
                    type_of_var=type_of_var,
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
                    )
                    if res is not None:
                        block_rows.append(res)

                out_inner_tables[min_samples_per_country] = (
                    pd.concat(block_rows, axis=0, join="outer", ignore_index=True) if block_rows else pd.DataFrame()
                )
            out_tables[type_of_var] = out_inner_tables

        return out_tables

    def test_invariance_across_datasets(self) -> dict:
        """

        Returns:

        """
        """
        Loop over datasets × traits and build a dict:

            {dataset_name: DataFrame with rmsea & cfi for every trait}

        Returns:
            dict[str, pd.DataFrame]
        """
        invariance_cfg = self.cfg_postprocessing["measurement_invariance"]["across_datasets"]
        out_tables = {}

        for type_of_var in ("trait_features", "trait_wb_items", "state_wb_items"):
            #  aligned data for every construct in this block
            merged_dict = self._merge_all_dataset_dfs(type_of_var=type_of_var,
                                                      invariance_cfg=invariance_cfg[type_of_var])

            block_rows = []
            for factor, df in merged_dict.items():
                # columns minus grouping variable = substantive items
                item_cols = [c for c in df.columns if c != "dataset"]
                res = self.run_invariance(
                    df=df,
                    items=item_cols,
                    factor=factor,
                    group_col="dataset",
                    method_items=None,
                )
                if res is not None:
                    block_rows.append(res)

            out_tables[type_of_var] = (
                pd.concat(block_rows, ignore_index=True) if block_rows else pd.DataFrame()
            )

        return out_tables

    def _merge_all_dataset_dfs(self, type_of_var: str, invariance_cfg: dict) -> dict[str, pd.DataFrame]:
        """
        Merge the item–level data for **all** constructs defined in
        `self.cfg_postprocessing`.

        Args:
            None

        Returns:
            Dict[str, pd.DataFrame]:
                Keys are construct names (e.g., "sincerity"),
                values are merged dataframes ready for measurement-invariance
                analyses.  Each dataframe

                • contains only the target (cocointernational) item columns
                • has identical column order across datasets
                • carries an extra column ``dataset`` that flags the source.
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

    def _build_cocoesm_dfs(
            self,
            construct_cfg: list[dict],
            df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        Extract one dataframe per construct *only* from the ``cocoesm`` dataset.

        Args:
            type_of_var:  Variable block key, e.g. ``"trait_features"``.
            construct_cfg: A list of dicts, each with at least
                           ``{"name": <str>, "item_names": <list[str]>}``
                           (see the compact config you provided).

        Returns:
            Dict[str, pd.DataFrame]:
                • Keys are construct / factor names (e.g. ``"sincerity"``).
                • Values are dataframes that contain exactly the requested
                  ``item_names`` columns (same order) and an extra
                  ``dataset`` column pre-filled with the string ``"cocoesm"``.
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

    def _get_target_items(self, post_cfg: dict) -> list[str]:
        """
        Collect the cocointernational item names for a construct.

        Args:
            post_cfg: One variable's dict taken from `self.cfg_postprocessing`.

        Returns:
            List[str]: Ordered list of item names in the cocointernational
            (target) naming scheme.
        """
        mapping_block = post_cfg.get("mapping", {})

        # Prefer the first mapping's *values* as canonical order
        if mapping_block:
            first_map = next(iter(mapping_block.values()))
            return list(first_map.values())

        # If no mapping section exists (all datasets already match),
        # treat the existing columns as canonical — assume the first
        # dataframe’s columns give the order.
        try:
            first_dataset = next(iter(self.data_dict[post_cfg["name"]].values()))

        except KeyError:
            print()

        return list(first_dataset.columns)

    def _build_renaming_mapping(self, post_cfg: dict, dataset: str) -> dict[str, str]:
        """
        Create a {old_col: new_col} mapping for one dataset.

        Args:
            post_cfg:   One variable's dict from `cfg_postprocessing`.
            dataset:    Dataset key, e.g. "cocout" or "cocoms".

        Returns:
            Dict[str, str]: Column-renaming dictionary. Empty when no
            renaming is required.
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
        Return a numeric item DataFrame and a one-factor model for semopy.

        Args:
            dataset: Dataset key, e.g. ``"emotions"``.
            df: Full trait-level DataFrame.
            personality_var: One entry from the YAML personality config.

        Returns:
            tuple[pd.DataFrame, str]:
                * DataFrame with item columns cast to *float* (rows with other
                  types become **NaN**).
                * Lavaan-style model string, e.g.
                  ``"sociability =~ 1*item1 + item2 + item3"``.
                  If fewer than two items are present, returns an empty
                  DataFrame and empty string.
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

        # tmp = tmp.loc[:, tmp.var(skipna=True) > 0]
        if tmp.shape[1] < 2:
            return pd.DataFrame(), ""

        factor = var_config["name"]
        indicators = [c for c in tmp.columns if c != "country"]  # drop grouping col
        model = f"{factor} =~ " + " + ".join(["1*" + indicators[0], *indicators[1:]])
        return tmp, model

    def conduct_cfa(
            self,
            data: pd.DataFrame,
            model_str: str,
            group_col: Optional[str] = "country",
    ) -> pd.DataFrame | None:
        """
        Run a one-factor CFA (lavaan) separately for each group in `group_col`
        and return a tidy dataframe with one row per group.

        Args:
            data:       Wide dataframe with item columns *plus* a grouping column.
            model_str:  lavaan model string, e.g.  "sociability =~ 1*bfi2_1 + bfi2_16 + …"
            group_col:  Column that identifies groups (default "country").
                        If None, the whole dataframe is treated as one group.

        Returns:
            pd.DataFrame with columns
                ["factor", "rmsea", "cfi", "number_of_items", "group"]
                (group = country name by default), rounded to 3 decimals.
            Returns None when the model cannot be fitted in any group.
        """
        if not model_str:
            return None

        factor_name = model_str.split(" =~")[0].strip()
        item_cols = [c for c in data.columns if c not in (group_col,)]

        rows = []
        try:
            # loop over all groups (or single shot if group_col is None)
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

    # ---------- 2.  fit one lavaan model ---------------------------------------
    def _fit_lavaan(self, model: str, r_df, group: str):
        """Wrapper for lavaan::sem (FIML, multi-group)."""
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
    ) -> pd.DataFrame | None:
        """
        Base / weak / strong invariance using lavaan + group.equal
        (orthogonal 'method' factor handled via `method_items`).
        """
        # ── sanity ───────────────────────────────────────────────────────
        df = df.loc[:, ~df.T.duplicated()]
        items = [c for c in items if c != group_col]
        if len(items) < 2 or df[group_col].nunique() < 2:
            return None

        # ── 1. build model ───────────────────────────────────────────────
        model = self._build_invariance_model(factor, items, method_items)

        # ── 2. prepare data (indicators + group column) ─────────────────
        r_df = _df_to_r(df[[*items, group_col]])

        # ── 3. fit base / weak / strong ─────────────────────────────────
        fit_base = self._fit_multi_group(model, r_df, group_col)
        # print(ro.r['summary'](fit_base))  # Can print R summary output like this
        fit_weak = self._fit_multi_group(model, r_df, group_col, equal=["loadings"])
        fit_strong = self._fit_multi_group(model, r_df, group_col,
                                           equal=["loadings", "intercepts"])
        try:
            for lbl, fit in zip(("base", "weak", "strong"),
                                (fit_base, fit_weak, fit_strong)):
                fm = ro.r("fitMeasures")(fit)  # named numeric vector
                chi2 = float(fm.rx2("chisq")[0])  # length-1 → grab element
                df_m = int(fm.rx2("df")[0])
                try:
                    print(f"{lbl}: χ² = {chi2:.2f}, df = {df_m}, χ²/df = {chi2 / df_m:.3f}")
                except ZeroDivisionError:
                    print(f"{lbl}: <UNK> = {chi2:.2f}")

            # ── 4. compare & tidy output ────────────────────────────────────
            out = self._compare_invariance_fits(fit_base, fit_weak, fit_strong, n_items=len(items))
            out = out.round(3)
            out.insert(0, "factor", factor)

            return out

        except: # Any lavaan error related to non-convergen
            return pd.DataFrame()

    def _build_invariance_model(
            self,
            factor: str,
            items: list[str],
            method_items: list[str] | None = None,
    ) -> str:
        """
        Construct a CFA model string.

        * The first substantive item fixes the latent scale (`1*item1`).
        * Optionally adds an orthogonal `method` factor.
        """
        loadings = " + ".join(["1*" + items[0], *items[1:]])
        model = f"{factor} =~ {loadings}"
        if method_items:
            m_line = " + ".join(method_items)
            model += f"\nmethod =~ {m_line}\n{factor} ~~ 0*method"
        return model

    def _fit_multi_group(
            self,
            model: str,
            r_df,
            group_col: str,
            equal: list[str] | None = None,
    ):
        """Wrapper around lavaan.sem with FIML and meanstructure."""
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

    def _compare_invariance_fits(self,
                                 fit_base,
                                 fit_weak,
                                 fit_strong,
                                 n_items: int,
                                 base_cfi: float = 0.90,
                                 base_rmsea: float = 0.08,
                                 delta_cfi_bound: float = -0.02,  # 0.01, 0.02
                                 delta_rmsea_bound: float = 0.03,  # 0.015, 0.03
                                 ) -> pd.DataFrame:
        """Return RMSEA, ΔRMSEA, CFI, ΔCFI, χ²-diff, df-diff, p-diff."""
        cmp = semtools.compareFit(base=fit_base, weak=fit_weak, strong=fit_strong)

        rmsea = list(cmp.slots["fit"].rx2("rmsea"))
        cfi = list(cmp.slots["fit"].rx2("cfi"))
        delta_rmsea = [np.nan] + list(cmp.slots["fit.diff"].rx2("rmsea"))
        delta_cfi = [np.nan] + list(cmp.slots["fit.diff"].rx2("cfi"))

        nested = cmp.slots["nested"]
        chisq_diff = list(nested.rx2("Chisq diff"))
        dfs = list(cmp.slots["fit"].rx2("df"))
        p_diff = list(nested.rx2("Pr(>Chisq)"))

        out = pd.DataFrame(
            {
                "model": ["base", "weak", "strong"],
                "n_items": n_items,
                "dfs": dfs,
                "rmsea": rmsea,
                "delta_rmsea": delta_rmsea,
                "cfi": cfi,
                "delta_cfi": delta_cfi,
                "chisq_diff": chisq_diff,
                "p_diff": p_diff,
            }
        )

        # ── invariance flag ──────────────────────────────────────────
        # 1) Configural: evaluate fit of the *base* model itself
        cfg_ok = (out["cfi"] >= base_cfi) & (out["rmsea"] <= base_rmsea)
        out["configural_MI"] = np.where(out["model"] == "base", cfg_ok.astype(int), np.nan)

        # 2 & 3) Weak & strong: same Δ-criterion, applied to their respective rows
        delta_ok = (out["delta_cfi"] >= delta_cfi_bound) & (out["delta_rmsea"] <= delta_rmsea_bound)
        out["weak_MI"] = np.where(out["model"] == "weak", delta_ok.astype(int), np.nan)
        out["strong_MI"] = np.where(out["model"] == "strong", delta_ok.astype(int), np.nan)

        return out.round(3)

    # TODO: Maybe merge this functionality with the creation of big df?
    def _merge_emotions_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge complementary emotion columns (``occup_*`` / ``int_*``) into a
        single column per emotion.

        Expected input
        --------------
        * ``self.emotions_df`` – a DataFrame whose columns follow the pattern
          ``occup_<emotion>`` or ``int_<emotion>`` (e.g. ``occup_relaxed``,
          ``int_relaxed``).
        * Both columns may be present, only one, or neither.
          - If **both** exist, the new column is the row-wise **mean** of the two
            (ignoring NaN, so it defaults to the non-missing value).
          - If **one** exists, the new column is simply a copy of that column.
        * The merged columns are named **only** by the emotion suffix
          (``relaxed``, ``enthusiastic`` …).

        Returns
        -------
        pd.DataFrame
            A fresh DataFrame containing the merged emotion columns **only**
            (original ``occup_*`` / ``int_*`` columns are dropped).
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

    def _correct_emotions_items(self, items: list[str]) -> list[str]:
        """
        Strip the leading ``occup_`` / ``int_`` (or any prefix before the first
        underscore) from emotion–item names.

        Examples
        ['relaxed', 'enthusiastic']

        Notes
        -----
        * Order is preserved; duplicates are removed.
        * If a name contains **no** underscore, it is returned unchanged.
        """
        seen = set()
        cleaned = []

        for item in items:
            suffix = item.split("_", 1)[-1]  # part after first "_"
            if suffix not in seen:
                seen.add(suffix)
                cleaned.append(suffix)

        return cleaned


