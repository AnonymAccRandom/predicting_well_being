"""
Creates the Source Data file for Nature Communications from the stored results.

Reads only the result directories (results/), never the participant data (data/), and never
re-runs the cross-validation. For each display item (Figure 2, Figure 3, Supplementary Figures and
Tables built from results), it writes one labeled Excel file to results/source_data/Source Data/,
plus a README.md that maps each display item to its file and to the postprocessing method that
regenerates it. The folder is zipped to results/source_data/Source Data.zip. The display items and
all other settings are in the block create_source_data of configs/cfg_postprocessing.yaml.

Person-level data are not included: the SHAP files contain one row per participant (SHAP values and
imputed predictor values), so the Source Data contain per-predictor summaries of the SHAP values only.

Run from src/ with the repository root on PYTHONPATH, like main.py (see README):
python create_source_data.py
"""

import json
import os
import re
import shutil

import numpy as np
import pandas as pd

from src.postprocessing.ShapProcessor import ShapProcessor
from src.utils.DataLoader import DataLoader
from src.utils.SINumbering import SINumbering


def load_configs() -> tuple[dict, dict, dict]:
    """
    Loads the configs needed to locate and label the result files.

    Returns:
        tuple[dict, dict, dict]: cfg_postprocessing, cfg_analysis, and the name mapping.
    """
    data_loader = DataLoader()
    cfg_postprocessing = data_loader.read_yaml("../configs/cfg_postprocessing.yaml")
    cfg_analysis = data_loader.read_yaml("../configs/cfg_analysis.yaml")
    name_mapping = data_loader.read_yaml("../configs/name_mapping.yaml")
    return cfg_postprocessing, cfg_analysis, name_mapping


def load_fold_level_results(
    sig_tests_dir: str, crit: str, model_names: dict, skip_dirs: list[str]
) -> pd.DataFrame:
    """
    Loads the outer-fold performance values (10 repetitions x 10 outer folds x 5 imputations per model)
    with their identifiers from the cv_results_rep_*.json files.

    Args:
        sig_tests_dir: Directory containing <feature_combination>/<samples>/<crit>/<model>/cv_results_rep_*.json
        crit: Criterion to load.
        model_names: Model name in the result paths: label in the output.
        skip_dirs: Directories in sig_tests_dir that hold no fold-level results.

    Returns:
        pd.DataFrame: One row per model, repetition, outer fold, and imputation.
    """
    rows = []
    for feature_combination in sorted(os.listdir(sig_tests_dir)):
        fc_dir = os.path.join(sig_tests_dir, feature_combination)
        if not os.path.isdir(fc_dir) or feature_combination in skip_dirs:
            continue
        for samples in sorted(os.listdir(fc_dir)):
            crit_dir = os.path.join(fc_dir, samples, crit)
            if not os.path.isdir(crit_dir):
                continue
            for model in sorted(os.listdir(crit_dir)):
                model_dir = os.path.join(crit_dir, model)
                if not os.path.isdir(model_dir):  # e.g., .DS_Store
                    continue
                for filename in sorted(os.listdir(model_dir)):
                    match = re.fullmatch(r"cv_results_rep_(\d+)\.json", filename)
                    if not match:
                        continue
                    with open(os.path.join(model_dir, filename)) as f:
                        rep_results = json.load(f)
                    for outer_fold, fold_results in rep_results.items():
                        for imputation, metrics in fold_results.items():
                            rows.append(
                                {
                                    "feature_combination": feature_combination,
                                    "samples": samples,
                                    "model": model_names.get(model, model),
                                    "repetition": int(match.group(1)),
                                    "outer_fold": int(outer_fold.split("_")[-1]),
                                    "imputation": int(imputation.split("_")[-1]),
                                    "r": metrics["pearson"],
                                    "R2": metrics["r2"],
                                    "spearman": metrics["spearman"],
                                    "neg_mse": metrics["neg_mean_squared_error"],
                                }
                            )
    return pd.DataFrame(rows)


def flatten_cv_summary(all_cv_results: dict, model_names: dict) -> pd.DataFrame:
    """
    Flattens the condensed CV results into one row per crit, samples, feature combination, model, and metric.

    Args:
        all_cv_results: Content of all_cv_results.json.
        model_names: Model name in the results: label in the output.

    Returns:
        pd.DataFrame: One row per analysis and metric with M and SDs as columns.
    """
    rows = []
    for crit, crit_results in all_cv_results.items():
        for samples, samples_results in crit_results.items():
            for feature_combination, fc_results in samples_results.items():
                for model, model_results in fc_results.items():
                    for metric, stats in model_results.items():
                        rows.append(
                            {
                                "crit": crit,
                                "samples": samples,
                                "feature_combination": feature_combination,
                                "model": model_names.get(model, model),
                                "metric": metric,
                                **{stat: float(value) for stat, value in stats.items()},
                            }
                        )
    return pd.DataFrame(rows)


def summarize_shap(shap_exp_dct: dict, feature_combinations: list[str]) -> pd.DataFrame:
    """
    Summarizes SHAP values per predictor (mean absolute SHAP value, mean SHAP value, rank) for the
    feature combinations shown in a beeswarm figure. Person-level values are not returned.

    Args:
        shap_exp_dct: SHAP explanation object per feature combination.
        feature_combinations: Feature combinations (panels) to summarize.

    Returns:
        pd.DataFrame: One row per feature combination and predictor, sorted by rank.
    """
    rows = []
    for feature_combination in feature_combinations:
        if feature_combination not in shap_exp_dct:
            continue
        shap_exp = shap_exp_dct[feature_combination]
        values = np.asarray(shap_exp.values)
        mean_abs = np.abs(values).mean(axis=0)
        ranks = (-mean_abs).argsort().argsort() + 1
        for idx, feature_name in enumerate(shap_exp.feature_names):
            rows.append(
                {
                    "feature_combination": feature_combination,
                    "n": values.shape[0],
                    "predictor": feature_name,
                    "rank_mean_abs_shap": int(ranks[idx]),
                    "mean_abs_shap": float(mean_abs[idx]),
                    "mean_shap": float(values[:, idx].mean()),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["feature_combination", "rank_mean_abs_shap"])
    return df


def write_excel(
    sheets: dict[str, pd.DataFrame], path: str, index: bool = False
) -> None:
    """
    Writes several DataFrames as sheets of one Excel file.

    Args:
        sheets: Sheet name (max. 31 characters are used) and DataFrame.
        path: Output path of the Excel file.
        index: If the index is written.
    """
    with pd.ExcelWriter(path) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=index)


def main() -> None:
    """
    Creates the Source Data folder, its README, and the zipped Source Data file from the stored results.

    Prints the size of each file and flags files above the editor's limit (max_file_size_mb).
    """
    cfg_postprocessing, cfg_analysis, name_mapping = load_configs()
    si_numbering = SINumbering(cfg_postprocessing["si_numbering"])
    cfg = cfg_postprocessing["create_source_data"]
    model_names = cfg["model_names"]
    data_paths = cfg_postprocessing["general"]["data_paths"]
    base_result_path = data_paths["base_path"]
    main_results_path = os.path.join(base_result_path, data_paths["main_results"])
    sig_tests_path = os.path.join(base_result_path, data_paths["sig_tests"])

    output_dir = os.path.join(cfg["output_dir"], cfg["dir_name"])
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    written = []  # (display item in R2, final label, file name, content)

    # Figure 2 and the test tables: fold-level values
    cfg_fig2 = cfg["figure_2"]
    fold_level = load_fold_level_results(
        sig_tests_path, cfg_fig2["crit"], model_names, cfg["fold_level_skip_dirs"]
    )
    all_cv_results = DataLoader().read_json(
        os.path.join(main_results_path, "all_cv_results.json")
    )
    cv_summary = flatten_cv_summary(all_cv_results, model_names)

    fig2_fold, fig2_summary = [], []
    for panel, cfg_panel in cfg_fig2["panels"].items():
        samples, combos = cfg_panel["samples"], cfg_panel["feature_combinations"]
        fold_sel = fold_level[
            (fold_level["samples"] == samples)
            & fold_level["feature_combination"].isin(combos)
        ]
        fig2_fold.append(fold_sel.assign(panel=panel))
        summary_sel = cv_summary[
            (cv_summary["crit"] == cfg_fig2["crit"])
            & (cv_summary["samples"] == samples)
            & cv_summary["feature_combination"].isin(combos)
            & (cv_summary["metric"] == cfg_fig2["metric"])
        ]
        fig2_summary.append(summary_sel.assign(panel=panel))
    fig2_fold = pd.concat(fig2_fold)[["panel", *fold_level.columns]]
    fig2_summary = pd.concat(fig2_summary)
    write_excel(
        {"fold_level_values": fig2_fold, "mean_sd": fig2_summary},
        os.path.join(output_dir, "Figure_2.xlsx"),
    )
    written.append(
        (
            "Figure 2",
            "Figure 2",
            "Figure_2.xlsx",
            "Outer-fold r per model (n = 500: 10 repetitions x 10 outer folds x 5 imputations); M and SD per model",
        )
    )

    # Figure 3 and Supplementary Figures S1 to S15: SHAP summaries
    meta_vars = [
        cfg_analysis["cv"]["id_grouping_col"],
        cfg_analysis["imputation"]["country_grouping_col"],
        cfg_analysis["imputation"]["years_col"],
    ]
    shap_processor = ShapProcessor(
        cfg_postprocessing=cfg_postprocessing,
        name_mapping=name_mapping,
        base_result_path=base_result_path,
        cv_shap_results_path=main_results_path,
        processed_results_filenames=cfg_postprocessing["general"][
            "processed_filenames"
        ],
        meta_vars=meta_vars,
    )
    # only used for samples_to_include "combo", which the Source Data do not use
    col_assignment = cfg_postprocessing["create_shap_plots"]["col_assignment"]
    shap_cache = {}

    def get_shap(crit: str, samples: str, model: str) -> dict:
        """
        Loads the SHAP explanation objects for one crit, samples, and model (cached).

        Args:
            crit: Criterion, e.g., "wb_state".
            samples: Samples to include, e.g., "selected".
            model: Prediction model, e.g., "randomforestregressor".

        Returns:
            dict: SHAP explanation object per feature combination.
        """
        key = (crit, samples, model)
        if key not in shap_cache:
            shap_cache[key] = shap_processor.prepare_shap_data(
                model_to_plot=model,
                crit_to_plot=crit,
                samples_to_include=samples,
                col_assignment=col_assignment,
            )
        return shap_cache[key]

    cfg_fig3 = cfg["figure_3"]
    fig3_sheets = {
        "shap_summary": summarize_shap(
            get_shap(cfg_fig3["crit"], cfg_fig3["samples"], cfg_fig3["model"]),
            cfg_fig3["feature_combinations"],
        )
    }
    ia_data = shap_processor.prepare_shap_ia_data(
        crit_to_plot=cfg_fig3["crit"],
        samples_to_include=cfg_fig3["samples"],
        model_to_plot=cfg_fig3["model"],
        feature_combination_to_plot=cfg_fig3["ia_feature_combination"],
    )
    fig3_sheets["interaction_summary"] = summarize_shap(ia_data, list(ia_data.keys()))
    write_excel(fig3_sheets, os.path.join(output_dir, "Figure_3.xlsx"))
    written.append(
        (
            "Figure 3",
            "Figure 3",
            "Figure_3.xlsx",
            "Per predictor and panel: n, mean absolute SHAP, mean SHAP, rank (panels a to c); "
            "same for interaction values (panel d)",
        )
    )

    for figure, cfg_figure in cfg["supplementary_figures"].items():
        crit, samples, model = (
            cfg_figure["crit"],
            cfg_figure["samples"],
            cfg_figure["model"],
        )
        combos = cfg_figure["feature_combinations"]
        label = si_numbering.figure_label(figure)
        filename = f"{label.replace(' ', '_')}.xlsx"
        write_excel(
            {"shap_summary": summarize_shap(get_shap(crit, samples, model), combos)},
            os.path.join(output_dir, filename),
        )
        written.append(
            (
                f"Supplementary Figure {figure}",
                label,
                filename,
                f"Per predictor and panel: n, mean absolute SHAP, mean SHAP, rank "
                f"({model_names[model]}, {samples}, {crit})",
            )
        )
        # Free memory, the SHAP objects are large
        del shap_cache[(crit, samples, model)]

    # Supplementary Tables
    missing = []
    table_files = cfg["supplementary_tables"]
    fold_level_tables = cfg["fold_level_tables"]
    cv_summary_tables = cfg["cv_summary_tables"]
    for table in sorted(
        set(table_files) | set(fold_level_tables) | set(cv_summary_tables),
        key=lambda s: int(s[1:]),
    ):
        sheets, content = {}, []
        for sheet_name, rel_path in table_files.get(table, {}).items():
            path = os.path.join(base_result_path, rel_path)
            if os.path.exists(path):
                sheets[sheet_name] = pd.read_excel(path, header=None)
                content.append(f"results/{rel_path}")
            else:
                missing.append((table, f"results/{rel_path}"))
        if table in cv_summary_tables:
            crit, samples = (
                cv_summary_tables[table]["crit"],
                cv_summary_tables[table]["samples"],
            )
            sheets["mean_sd"] = cv_summary[
                (cv_summary["crit"] == crit) & (cv_summary["samples"] == samples)
            ]
            content.append(
                f"results/{data_paths['main_results']}/all_cv_results.json ({samples})"
            )
        if table in fold_level_tables:
            sheets["fold_level_values"] = fold_level[
                fold_level["samples"].isin(fold_level_tables[table])
            ]
            content.append(
                f"fold-level values ({', '.join(fold_level_tables[table])} samples)"
            )
        if not sheets:
            continue
        label = si_numbering.table_label(table)
        filename = f"{label.replace(' ', '_')}.xlsx"
        write_excel(sheets, os.path.join(output_dir, filename))
        written.append(
            (f"Supplementary Table {table}", label, filename, "; ".join(content))
        )

    # README and size check
    sizes = {
        filename: os.path.getsize(os.path.join(output_dir, filename)) / 1e6
        for _, _, filename, _ in written
    }
    write_readme(
        output_dir, written, sizes, missing, si_numbering, cfg["regenerated_by"]
    )
    shutil.make_archive(
        os.path.join(cfg["output_dir"], cfg["dir_name"]),
        "zip",
        cfg["output_dir"],
        cfg["dir_name"],
    )

    zip_size = (
        os.path.getsize(os.path.join(cfg["output_dir"], f"{cfg['dir_name']}.zip")) / 1e6
    )
    print(
        f"Source Data written to {output_dir} ({len(written)} files), zip: {zip_size:.1f} MB"
    )
    for filename, size in sizes.items():
        flag = "  <-- above limit" if size > cfg["max_file_size_mb"] else ""
        print(f"  {filename}: {size:.2f} MB{flag}")
    for table, path in missing:
        print(f"  MISSING for {table}: {path}")


def write_readme(
    output_dir: str,
    written: list[tuple[str, str, str, str]],
    sizes: dict[str, float],
    missing: list[tuple[str, str]],
    si_numbering: SINumbering,
    regenerated_by: dict[str, str],
) -> None:
    """
    Writes the README of the Source Data folder.

    Args:
        output_dir: The Source Data folder.
        written: Display item in R2, final label, file name, and content description of each written file.
        sizes: File size in MB per file name.
        missing: Display item and result file for each item that could not be included.
        si_numbering: Final numbers of the Supplementary Tables and Figures.
        regenerated_by: Display item (R2 numbers): postprocessing method that regenerates it.
    """
    lines = [
        "# Source Data",
        "",
        "Source data for the figures and tables built from the results of the machine-learning analyses.",
        "Each file is labeled with its display item in the final version of the Supplementary Information.",
        "The column 'R2 number' gives the number in the revised submission (R2). Supplementary Tables",
        f"{', '.join(si_numbering.supplementary_data)} of R2 are provided as Supplementary Data",
        "(`src/create_supplementary_data.py`).",
        "",
        "All files were created by `src/create_source_data.py` from the result files in `results/`, without",
        "re-running the cross-validation. The SHAP files contain one row per participant and are therefore",
        "summarized per predictor here; the full SHAP matrices are part of the data deposit.",
        "",
        "| Display item | R2 number | File | Content | Size (MB) |",
        "|---|---|---|---|---|",
    ]
    for item, label, filename, content in written:
        lines.append(
            f"| {label} | {item} | {filename} | {content} | {sizes[filename]:.2f} |"
        )
    lines += [
        "",
        "## Regenerating the display items",
        "",
        "Set `methods_to_apply` in `configs/cfg_postprocessing.yaml` and run `python src/main.py` (see the",
        "repository README). Methods marked 'needs data/' require the preprocessed data from the data deposit.",
        "",
        "| Display item | Postprocessing method |",
        "|---|---|",
    ]
    for item, method in regenerated_by.items():
        label = item if item.startswith("Figure") else si_numbering.relabel(item)
        lines.append(f"| {label} | {method} |")
    if missing:
        lines += ["", "## Not included (result file not found)", ""]
        lines += [f"- {table}: {path}" for table, path in missing]
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
