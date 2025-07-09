import os

import numpy as np
import pandas as pd

import rpy2.robjects as ro
ro.r('install.packages("lavaan.mi", repos="https://cran.rstudio.com")')

from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# Activate automatic pandas <-> R data-frame conversion
# lavaan = importr("lavaan")
#ro.r('install.packages("semTools", repos="https://cran.rstudio.com")')
semtools = importr("semTools")
lavaan = importr("lavaan")
#print("semTools version:", ro.r('packageVersion("semTools")')[0])
#print("lavaan version:", ro.r('packageVersion("lavaan")')[0])
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
            if file.startswith("raw_trait_df"):
                dataset = file.split("_")[-1]
                data_dict["trait_features"][dataset] = self.data_loader.read_pkl(os.path.join(path_to_preprocessed_data, file))
            if file.startswith("trait_wb_items"):
                dataset = file.split("_")[-1]
                data_dict["trait_wb_items"][dataset] = self.data_loader.read_pkl(os.path.join(path_to_preprocessed_data, file))
            if file.startswith("wb_items"):
                dataset = file.split("_")[-1]
                data_dict["state_wb_items"][dataset] = self.data_loader.read_pkl(os.path.join(path_to_preprocessed_data, file))

        setattr(self, "data_dict", data_dict)

    def test_invariance_across_countries(self) -> None:
        """
        Do this at first only for coco international, as
        Returns:

        """
        tables = {"trait_features": {}, "trait_wb_items": {}, "state_wb_items": {}}

        for type_of_var, vals in self.data_dict.items():
            current_cfg = self._get_config_for_type(type_of_var)

            for dataset, big_df in vals.items():
                if dataset =="cocoesm":
                    test = big_df["country"].value_counts()
                    big_df_filtered = big_df[big_df["country"].map(big_df["country"].value_counts()) >= 100]

                    rows = []
                    for current_var in current_cfg:
                        df_items, model = self._get_items_and_construct(
                            var_type=type_of_var,
                            dataset=dataset,
                            df=big_df_filtered,
                            var_config=current_var,
                            add_country=True
                        )
                        test2 = df_items["country"].value_counts()

                        items = df_items.columns.tolist()
                        items = [c for c in items if c != "country"]
                        factor_name = current_var["name"]  # e.g. "assertiveness"

                        inv_table = self.run_invariance(
                            df=df_items,  # add grouping column
                            items=items,
                            factor=factor_name,
                            group_col="country"  # or any grouping variable
                        )

                        print(inv_table)
                    print()



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
        tables = {"trait_features": {}, "trait_wb_items": {}, "state_wb_items": {}}

        for type_of_var, vals in self.data_dict.items():
            current_cfg = self._get_config_for_type(type_of_var)

            all_datasets_df = self._merge_dataset_dfs()


            for dataset, big_df in vals.items():
                if dataset == "emotions" and type_of_var in ["trait_wb_items", "state_wb_items"]:
                    big_df = self._merge_emotions_items(df=big_df)
                    print()

                rows = []
                for current_var in current_cfg:
                    df_items, model = self._get_items_and_construct(
                        var_type=type_of_var,
                        dataset=dataset,
                        df=big_df,
                        var_config=current_var,
                    )
                    if not df_items.empty:
                        row = self.conduct_cfa(df_items, model)
                        if row is not None:
                            rows.append(row)
                if rows:
                    tables[type_of_var][dataset] = pd.concat(rows, ignore_index=True)

        return tables

    def _merge_dataset_dfs(self) -> pd.DataFrame:
        """
        Prepares the data so that we can conduct tests of measurement invariance
        according to R standards / syntax. The function
        - Filters each dataframe for the variables defined in the config for
          measurement invariance testing across datasets
        - Aligns the item_names (if there are differences)
        - Concatenates all dataframes vertically into one dataframe
        - creates a "grouping_column" that contains the dataset name (this could simply
          be based on the original index of the dfs which is structured like this:
          "dataset_num"

        Returns:

        """
        for



    def _get_config_for_type(self, var_type: str) -> list[dict]:
        """
        Return the correct preprocessing config list for a given variable type.

        Args:
            var_type: One of {"trait_features", "trait_wb_items", "state_wb_items"}

        Returns:
            A list of variable config dictionaries (from YAML).
        """
        if var_type == "trait_features":
            return self.cfg_preprocessing["person_level"]["personality"]
        elif var_type == "trait_wb_items":
            return self.cfg_preprocessing["person_level"]["criterion"]
        elif var_type == "state_wb_items":
            return self.cfg_preprocessing["esm_based"]["criterion"]
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

        tmp = (
            df[items]
            #.apply(pd.to_numeric, errors="ignore")
            #.astype(float, errors="ignore")
        )
        # tmp = tmp.loc[:, tmp.var(skipna=True) > 0]
        if tmp.shape[1] < 2:
            return pd.DataFrame(), ""

        factor = var_config["name"]
        indicators = [c for c in tmp.columns if c != "country"]  # drop grouping col
        model = f"{factor} =~ " + " + ".join(["1*" + indicators[0], *indicators[1:]])
        return tmp, model

    def conduct_cfa(self, data: pd.DataFrame, model_str: str):
        """
        Run one-factor CFA in lavaan via rpy2, return factor, rmsea, cfi.
        """
        if not model_str:
            return None

        try:
            r_df = _df_to_r(data)
            fit = lavaan.sem(model_str,
                             data=r_df,
                             missing="fiml",
                             meanstructure=True)

            rmsea, cfi = map(float,
                             lavaan.fitmeasures(fit,
                                                ro.StrVector(("rmsea", "cfi"))))
            return pd.DataFrame(
                {"factor": [model_str.split(" =~")[0].strip()],
                 "rmsea": [rmsea],
                 "cfi": [cfi],
                 "number_of_items": len(data.columns)}
            )

        except Exception as err:
            print(f"[lavaan error] {err}")
            return None

        except Exception as e:
            # lavaan throws an R exception → just skip this scale
            print(f"[lavaan error] {e}")
            return None

    def _build_invariance_models(
            self,
            items: list[str],
            factor: str,
            n_groups: int,
    ) -> dict[str, str]:
        """Return lavaan strings for base, weak, strong with dynamic group count."""

        # ---------- helper: label vectors ---------------------------------
        def labs(prefix: str) -> str:
            return ",".join(f"{prefix}{g + 1}" for g in range(n_groups))

        # ---------- factor loadings ---------------------------------------
        def cfg_load(i: int, itm: str) -> str:
            if i == 0:
                return f"1*{itm}"
            return f"c({labs(f'a{i}')})*{itm}"

        cfg_loading = " + ".join(cfg_load(i, itm) for i, itm in enumerate(items))
        weak_loading = " + ".join(["1*" + items[0]] +
                                  [f"a{i}*{itm}" for i, itm in enumerate(items[1:], 1)])
        strong_loading = weak_loading

        # ---------- intercepts --------------------------------------------
        def cfg_int(i: int, itm: str) -> str:
            if i == 0:
                return f"{itm} ~ 0*1"
            return f"{itm} ~ c({labs(f'b{i}')})*1"

        cfg_ints = "\n".join(cfg_int(i, itm) for i, itm in enumerate(items))
        weak_ints = cfg_ints
        strong_ints = "\n".join(
            [f"{items[0]} ~ 0*1"] +
            [f"{itm} ~ b{i}*1" for i, itm in enumerate(items[1:], 1)]
        )

        # ---------- latent variance & means -------------------------------
        latent = (
            f"{factor} ~~ c({','.join(f'v{g + 1}' for g in range(n_groups))})*{factor}\n"
            f"{factor} ~  c({','.join(f'm{g + 1}' for g in range(n_groups))})*1"
        )

        # ---------- assemble templates ------------------------------------
        template = "{f} =~ {load}\n\n{ints}\n\n{latent}"

        return {
            "base": template.format(f=factor, load=cfg_loading, ints=cfg_ints, latent=latent),
            "weak": template.format(f=factor, load=weak_loading, ints=weak_ints, latent=latent),
            "strong": template.format(f=factor, load=strong_loading, ints=strong_ints, latent=latent),
        }

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
        items = [c for c in items if c != group_col]
        if len(items) < 2 or df[group_col].nunique() < 2:
            return None

        # ── 1. build model ───────────────────────────────────────────────
        # method_items = items.copy()
        model = self._build_invariance_model(factor, items, method_items)

        # ── 2. prepare data (indicators + group column) ─────────────────
        r_df = _df_to_r(df[[*items, group_col]])

        # ── 3. fit base / weak / strong ─────────────────────────────────
        fit_base = self._fit_multi_group(model, r_df, group_col)
        fit_weak = self._fit_multi_group(model, r_df, group_col, equal=["loadings"])
        fit_strong = self._fit_multi_group(model, r_df, group_col,
                                           equal=["loadings", "intercepts"])

        for lbl, fit in zip(("base", "weak", "strong"),
                            (fit_base, fit_weak, fit_strong)):
            fm = ro.r("fitMeasures")(fit)  # named numeric vector
            chi2 = float(fm.rx2("chisq")[0])  # length-1 → grab element
            df_m = int(fm.rx2("df")[0])
            print(f"{lbl}: χ² = {chi2:.2f}, df = {df_m}, χ²/df = {chi2 / df_m:.3f}")

        # ── 4. compare & tidy output ────────────────────────────────────
        out = self._compare_invariance_fits(fit_base, fit_weak, fit_strong)
        out.insert(0, "factor", factor)
        return out

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

    def _compare_invariance_fits(self, fit_base, fit_weak, fit_strong) -> pd.DataFrame:
        """Return RMSEA, ΔRMSEA, CFI, ΔCFI, χ²-diff, df-diff, p-diff."""
        cmp = semtools.compareFit(base=fit_base, weak=fit_weak, strong=fit_strong)

        rmsea = list(cmp.slots["fit"].rx2("rmsea"))
        cfi = list(cmp.slots["fit"].rx2("cfi"))
        delta_rmsea = [np.nan] + list(cmp.slots["fit.diff"].rx2("rmsea"))
        delta_cfi = [np.nan] + list(cmp.slots["fit.diff"].rx2("cfi"))

        nested = cmp.slots["nested"]
        chisq_diff = list(nested.rx2("Chisq diff"))
        df_diff = list(nested.rx2("Df diff"))
        p_diff = list(nested.rx2("Pr(>Chisq)"))

        return pd.DataFrame(
            {
                "model": ["base", "weak", "strong"],
                "rmsea": rmsea,
                "delta_rmsea": delta_rmsea,
                "cfi": cfi,
                "delta_cfi": delta_cfi,
                "chisq_diff": chisq_diff,
                "df_diff": df_diff,
                "p_diff": p_diff,
            }
        )

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


