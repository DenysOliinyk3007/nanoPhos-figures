"""Helper to dump each figure panel's exact plotted source data into the PRIDE
MetaInfo workbook (one sheet per panel), matching MetaInfo_figures_checklist_v01.xlsx.

Usage in a figure notebook (run in Jupyter, after the panel's plotting dataframe is built):

    from metainfo_export import dump_panel
    dump_panel(df, "Figure 2b")          # df = the exact dataframe the plot consumes

The first call creates MetaInfo_figures_checklist_v03.xlsx; subsequent calls append /
replace that panel's sheet, so notebooks can be run in any order. Sheet names must
match the v01 panel names exactly.
"""
from __future__ import annotations
import os
import pandas as pd

DEFAULT_PATH = r"Z:\Denys_nanoPhos\PRIDE\MetaInfo_figures_checklist_v04.xlsx"


def dump_panel(df: pd.DataFrame, sheet: str, path: str = DEFAULT_PATH,
               index: bool = False) -> None:
    """Write `df` to `sheet` in the MetaInfo v04 workbook (create or replace the sheet)."""
    df = pd.DataFrame(df).copy()
    # openpyxl sheet-name limit is 31 chars
    sheet = sheet[:31]
    if not os.path.exists(path):
        with pd.ExcelWriter(path, engine="openpyxl") as xl:
            df.to_excel(xl, sheet_name=sheet, index=index)
    else:
        with pd.ExcelWriter(path, mode="a", if_sheet_exists="replace",
                            engine="openpyxl") as xl:
            df.to_excel(xl, sheet_name=sheet, index=index)
    print(f"  [MetaInfo] wrote '{sheet}'  ({df.shape[0]} rows x {df.shape[1]} cols)")
