"""
Creates the Supplementary Data files for Nature Communications.

The editor asked to convert Supplementary Tables longer than one page into Supplementary Data. This
script writes one Excel file per moved table to results/supplementary_data/, each with a 'data' sheet
and a 'notes' sheet (title, table note, source). The tables, their numbers, and all other settings are
in the blocks si_numbering and create_supplementary_data of configs/cfg_postprocessing.yaml.

- Supplementary Data 1 to 4 (R2: S2, S8, S9, S18) are overviews, codebooks, and descriptives that
  exist only as Word tables. They are read from the Supplementary Information docx, with tracked
  changes accepted.
- Supplementary Data 5 to 8 (R2: S20 to S23, measurement invariance) are built from the stored
  result files in results/mi_across_datasets/ and results/mi_across_countries/. Title and note are
  taken from the docx.

Never reads the participant data (data/).

Run from src/ with the repository root on PYTHONPATH, like main.py (see README):
python create_supplementary_data.py <path to SI docx>
"""

import os
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from src.utils.DataLoader import DataLoader

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
# Row order per construct in the MI results
MI_MODELS = ["configural", "metric", "scalar"]


def paragraph_text(el: ET.Element) -> str:
    """
    Returns the visible text of an element with tracked insertions kept and tracked deletions dropped
    (deleted text is stored in w:delText, not w:t). Paragraphs, line breaks, and tabs become spaces.

    Args:
        el: A w:p, w:tc, or any other element of the document body.

    Returns:
        str: The text, stripped.
    """
    paragraphs = el.findall(f".//{W}p") if el.tag != W + "p" else [el]
    texts = [
        "".join(
            node.text or "" if node.tag == W + "t" else " "
            for node in p.iter()
            if node.tag in (W + "t", W + "br", W + "cr", W + "tab")
        )
        for p in paragraphs
    ]
    return " ".join(" ".join(t.split()) for t in texts if t.strip())


def table_rows(tbl: ET.Element) -> list[list[str]]:
    """
    Reads a Word table into rows of cell texts, one entry per cell (horizontally merged cells count
    once). Vertically merged cells repeat the text of the cell where the merge starts.

    Args:
        tbl: A w:tbl element.

    Returns:
        list[list[str]]: One list of cell texts per table row.
    """
    rows, above = [], {}  # above: grid column: text of the last cell starting there
    for tr in tbl.findall(W + "tr"):
        row, grid_col = [], 0
        for tc in tr.findall(W + "tc"):
            tc_pr = tc.find(W + "tcPr")
            span, v_merge = 1, None
            if tc_pr is not None:
                grid_span = tc_pr.find(W + "gridSpan")
                span = int(grid_span.get(W + "val")) if grid_span is not None else 1
                v_merge_el = tc_pr.find(W + "vMerge")
                if v_merge_el is not None:
                    v_merge = v_merge_el.get(W + "val") or "continue"
            text = (
                above.get(grid_col, "") if v_merge == "continue" else paragraph_text(tc)
            )
            above[grid_col] = text
            row.append(text)
            grid_col += span
        rows.append(row)
    return rows


def read_si_tables(docx_path: str, tables: list[str]) -> dict[str, dict]:
    """
    Collects title, Word tables, and note of the requested Supplementary Tables from the SI docx.
    A table runs from its caption ("Supplementary Table S8 - ...") to the next paragraph starting
    with "Supplementary "; tables split over several Word tables (S8, S9) are returned as parts.

    Args:
        docx_path: Path to the Supplementary Information docx.
        tables: R2 table numbers, e.g., ["S2", "S8"].

    Returns:
        dict[str, dict]: Per table: "title" (str), "parts" (list of row lists), "note" (str).
    """
    with zipfile.ZipFile(docx_path) as docx:
        body = ET.fromstring(docx.read("word/document.xml")).find(W + "body")

    found, current = {}, None
    for el in body:
        if el.tag == W + "p":
            text = paragraph_text(el)
            if text.startswith("Supplementary "):
                words = text.split()
                current = (
                    words[2]
                    if words[1] == "Table" and len(words) > 2 and words[2] in tables
                    else None
                )
                if current:
                    found[current] = {"title": text, "parts": [], "note": []}
            elif current and text and found[current]["parts"]:
                found[current]["note"].append(text)
        elif el.tag == W + "tbl" and current:
            found[current]["parts"].append(table_rows(el))

    missing = [table for table in tables if table not in found]
    if missing:
        raise ValueError(f"Not found in {docx_path}: {', '.join(missing)}")
    for content in found.values():
        content["note"] = "\n".join(content["note"])
    return found


def to_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts columns whose non-empty cells are all numbers to floats and leaves the others as text.

    Args:
        df: Table with the cell texts of a Word table.

    Returns:
        pd.DataFrame: The table with numeric columns converted.
    """
    for col in df.columns:
        values = df[col].replace("", pd.NA).dropna()
        converted = pd.to_numeric(
            values.str.replace("−", "-", regex=False), errors="coerce"
        )
        if len(values) and converted.notna().all():
            df[col] = pd.to_numeric(
                df[col].replace("", pd.NA).str.replace("−", "-", regex=False)
            )
    return df


def parts_to_frame(parts: list[list[list[str]]]) -> pd.DataFrame:
    """
    Stacks Word tables that share a header row (S2, S8, S9) into one DataFrame. Header rows that
    the docx repeats inside a table (for readability across page breaks) are dropped.

    Args:
        parts: The Word tables of one Supplementary Table, each as rows of cell texts.

    Returns:
        pd.DataFrame: The header of the first part as columns, the data rows of all parts.
    """
    header = parts[0][0]
    rows = []
    for part in parts:
        if part[0] != header:
            raise ValueError(f"Header differs between table parts: {part[0]}")
        rows += [row for row in part[1:] if row != header]
    if any(len(row) != len(header) for row in rows):
        raise ValueError("Row length differs from the header")
    return to_numeric_columns(pd.DataFrame(rows, columns=header))


def descriptives_to_frame(parts: list[list[list[str]]]) -> pd.DataFrame:
    """
    Converts the descriptives table (R2: S18) into one row per variable. Its rows are either section
    rows (first cell filled, e.g., "Person-lv. predictors") or variable rows (first cell empty,
    then variable, M, SD, and scale endpoints or units). The section becomes a column. Header rows
    repeated inside the table are dropped.

    Args:
        parts: The Word table(s) of the table.

    Returns:
        pd.DataFrame: Columns Section, Variable, M, SD, Scale Endpoints/Units.
    """
    rows, section = [], ""
    for part in parts:
        header = [cell for cell in part[0] if cell]
        if header != ["Variable", "M", "SD", "Scale Endpoints/ Units"]:
            raise ValueError(f"Unexpected header of the descriptives table: {part[0]}")
        for row in part[1:]:
            if [cell for cell in row if cell] == header:
                continue
            if row[0]:
                section = row[0]
            elif any(row[1:]):
                if len(row) != 5:
                    raise ValueError(f"Unexpected variable row: {row}")
                rows.append([section, *row[1:]])
    columns = ["Section", "Variable", "M", "SD", "Scale Endpoints/Units"]
    return to_numeric_columns(pd.DataFrame(rows, columns=columns))


def mi_to_frame(
    base_result_path: str, mi_table: dict, cfg: dict
) -> tuple[pd.DataFrame, list[str]]:
    """
    Builds a measurement invariance table from the stored results of the configured constructs. As in
    the SI tables, only constructs with at least mi_min_items items are reported (the stored results
    across countries with >= 100 participants also hold two-item constructs, whose models have
    negative df).

    Args:
        base_result_path: The results directory.
        mi_table: Directory ("dir") and file suffix ("suffix") of the stored results of one table.
        cfg: The create_supplementary_data block of cfg_postprocessing.yaml.

    Returns:
        tuple[pd.DataFrame, list[str]]: The table with a Block column, and the result files read.
    """
    frames, sources = [], []
    for construct, block in cfg["mi_blocks"].items():
        rel_path = os.path.join(
            mi_table["dir"], f"{construct}{mi_table['suffix']}.xlsx"
        )
        df = pd.read_excel(os.path.join(base_result_path, rel_path), index_col=0)
        frames.append(df.rename(columns=cfg["mi_column_names"]).assign(Block=block))
        sources.append(f"results/{rel_path}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Number of Items"] >= cfg["mi_min_items"]].reset_index(drop=True)
    # Each construct has one row per model, in the order configural, metric, scalar
    models = df.groupby(["Block", "Construct"], sort=False).cumcount()
    if (df.groupby(["Block", "Construct"], sort=False).size() != len(MI_MODELS)).any():
        raise ValueError(f"Expected {len(MI_MODELS)} rows per construct in {sources}")
    df["Model"] = models.map(dict(enumerate(MI_MODELS)))
    # A model without fit indices did not converge; as in the SI, it gets no MI code (not 0)
    not_converged = df["RMSEA"].isna() & df["CFI"].isna()
    df.loc[not_converged, ["Configural MI", "Metric MI", "Scalar MI"]] = np.nan
    first = ["Block", "Construct", "Model"]
    return df[[*first, *[col for col in df.columns if col not in first]]], sources


def apply_corrections(df: pd.DataFrame, corrections: list[list[str]]) -> pd.DataFrame:
    """
    Fills cells that are empty in the docx table (e.g., a sub-group missing in some rows). Each
    correction names the row by the value in a key column. Only empty cells are filled, so a
    correction fails loudly once the docx itself is fixed or the row is not found.

    Args:
        df: The table read from the docx.
        corrections: [key column, key value, column to fill, value] per cell.

    Returns:
        pd.DataFrame: The corrected table.
    """
    for key_col, key_value, col, value in corrections:
        mask = df[key_col] == key_value
        if mask.sum() != 1:
            raise ValueError(
                f"Correction: {key_col} = {key_value} matches {mask.sum()} rows"
            )
        if df.loc[mask, col].item() not in ("", None):
            raise ValueError(
                f"Correction: {col} for {key_value} is not empty in the docx "
                f"({df.loc[mask, col].item()!r}); remove the correction from the config"
            )
        df.loc[mask, col] = value
    return df


def write_readme(output_dir: str, written: list[tuple[str, str, str, float]]) -> None:
    """
    Writes the README of the Supplementary Data folder.

    Args:
        output_dir: The Supplementary Data folder.
        written: Supplementary Data label, R2 table, file name, and size in MB of each file.
    """
    lines = [
        "# Supplementary Data",
        "",
        "Supplementary Tables of the revised submission (R2) that extend over more than one page,",
        "converted to Supplementary Data at the editor's request. Each file has a 'data' sheet and a",
        "'notes' sheet with the title, the table note, and the source. Created by",
        "`src/create_supplementary_data.py`.",
        "",
        "| Supplementary Data | R2 number | File | Size (MB) |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {label} | {r2_table} | {filename} | {size:.2f} |"
        for label, r2_table, filename, size in written
    ]
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(docx_path: str) -> None:
    """
    Creates the Supplementary Data files and their README in the configured output directory.

    Args:
        docx_path: Path to the Supplementary Information docx (title, notes, and Word-only tables).
    """
    cfg_postprocessing = DataLoader().read_yaml("../configs/cfg_postprocessing.yaml")
    cfg = cfg_postprocessing["create_supplementary_data"]
    supplementary_data = cfg_postprocessing["si_numbering"]["supplementary_data"]
    base_result_path = cfg_postprocessing["general"]["data_paths"]["base_path"]
    output_dir = cfg["output_dir"]

    si_tables = read_si_tables(docx_path, list(supplementary_data))

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    written = []
    for r2_table, number in supplementary_data.items():
        content = si_tables[r2_table]
        if r2_table in cfg["mi_tables"]:
            df, sources = mi_to_frame(base_result_path, cfg["mi_tables"][r2_table], cfg)
            source = "; ".join(sources)
        elif r2_table in cfg["descriptives_tables"]:
            df = descriptives_to_frame(content["parts"])
            source = f"Supplementary Information (R2), Supplementary Table {r2_table}"
        else:
            df = parts_to_frame(content["parts"])
            source = f"Supplementary Information (R2), Supplementary Table {r2_table}"
        if r2_table in cfg["corrections"]:
            df = apply_corrections(df, cfg["corrections"][r2_table])

        label = f"Supplementary Data {number}"
        title = content["title"].split(" ", 3)[3].lstrip("-– ")
        notes = pd.DataFrame(
            {
                "field": ["Title", "Note", "Source", "Formerly"],
                "value": [
                    f"{label}. {title}",
                    content["note"],
                    source,
                    f"Supplementary Table {r2_table} (R2)",
                ],
            }
        )
        filename = f"Supplementary_Data_{number}.xlsx"
        path = os.path.join(output_dir, filename)
        with pd.ExcelWriter(path) as writer:
            df.to_excel(writer, sheet_name="data", index=False)
            notes.to_excel(writer, sheet_name="notes", index=False)
        size = os.path.getsize(path) / 1e6
        written.append((label, r2_table, filename, size))
        flag = "  <-- above limit" if size > cfg["max_file_size_mb"] else ""
        print(f"{label} (R2 {r2_table}): {df.shape[0]} rows, {size:.2f} MB{flag}")

    write_readme(output_dir, written)
    print(f"Supplementary Data written to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/create_supplementary_data.py <path to SI docx>")
    main(sys.argv[1])
