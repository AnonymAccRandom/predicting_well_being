import re


class SINumbering:
    """
    Numbering of the Supplementary Tables and Figures in the final version of the Supplementary
    Information.

    The code keys tables by their number in the revised submission (R2, e.g., "S24"). The tables
    configured as Supplementary Data moved to separate files; the remaining tables are renumbered
    consecutively in their R2 order. Supplementary Figures keep their numbers. All settings come
    from the si_numbering block of cfg_postprocessing.yaml.

    Attributes:
        prefix: Prefix of the final numbers ("" gives "Supplementary Table 16", "S" gives "S16").
        supplementary_data: R2 table number: Supplementary Data number.
        new_table_numbers: R2 table number: final table number, for the tables that stay in the SI.
    """

    def __init__(self, cfg_si_numbering: dict) -> None:
        """
        Args:
            cfg_si_numbering: The si_numbering block of cfg_postprocessing.yaml.
        """
        self.prefix = cfg_si_numbering["prefix"]
        self.supplementary_data = cfg_si_numbering["supplementary_data"]
        remaining = [
            f"S{i}"
            for i in range(1, cfg_si_numbering["n_tables_r2"] + 1)
            if f"S{i}" not in self.supplementary_data
        ]
        self.new_table_numbers = {
            r2_table: new for new, r2_table in enumerate(remaining, start=1)
        }

    def table_label(self, r2_table: str) -> str:
        """
        Returns the final label of an R2 Supplementary Table.

        Args:
            r2_table: Table number in R2, e.g., "S24".

        Returns:
            str: E.g., "Supplementary Table 16" or "Supplementary Data 5".
        """
        if r2_table in self.supplementary_data:
            return f"Supplementary Data {self.supplementary_data[r2_table]}"
        return f"Supplementary Table {self.prefix}{self.new_table_numbers[r2_table]}"

    def figure_label(self, r2_figure: str) -> str:
        """
        Returns the final label of an R2 Supplementary Figure (the number is unchanged).

        Args:
            r2_figure: Figure number in R2, e.g., "S8".

        Returns:
            str: E.g., "Supplementary Figure 8".
        """
        return f"Supplementary Figure {self.prefix}{r2_figure.lstrip('S')}"

    def relabel(self, item: str) -> str:
        """
        Replaces the R2 numbers in an item such as "S24, S25, S28 to S37" by the final numbers.
        Table numbers are renumbered; figure numbers (item starts with "Supplementary Figures")
        only get the final prefix.

        Args:
            item: R2 Supplementary Table numbers, or "Supplementary Figures S1 to S15".

        Returns:
            str: E.g., "Supplementary Tables 16, 17, 20 to 29".
        """
        if item.startswith("Supplementary Figures"):
            return re.sub(r"\bS(\d+)\b", rf"{self.prefix}\1", item)
        numbers = re.sub(
            r"\bS\d+\b",
            lambda m: f"{self.prefix}{self.new_table_numbers[m.group(0)]}",
            item,
        )
        plural = "s" if re.search(r",| to ", item) else ""
        return f"Supplementary Table{plural} {numbers}"
