"""Сборка годового набора данных 2008-2021"""
from pathlib import Path
import numpy as np, pandas as pd
D = Path(__file__).resolve().parents[1] / "data"
nan = np.nan

wm = pd.read_csv(D / "manual" / "manual_input_watermelon_080711.csv").set_index("year")
wm = wm[["export_qty_t", "export_value_kusd", "import_qty_t", "import_value_kusd"]]


sec = pd.DataFrame({
 "sec2_export_kusd": [151342.7,159451.9,174705.0,282381.2,301695.0,416186.8,590484.5,617965.6,556858.6,583430.8,661701.6,661699.0,657766.5,694112.9],
 "sec2_import_kusd": [nan,501393.8,576325.4,671205.0,nan,973601.8,1604578.4,2187391.5,1950680.9,2145244.7,2005922.6,2118897.8,1813715.0,1590033.6],
}, index=range(2008, 2022))

fx = pd.read_csv(D / "processed" / "fx_usd_daily.csv", parse_dates=["date"])
fxa = fx.groupby("year")["byn_per_usd"].mean().rename("byn_per_usd_avg")

df = wm.join(sec).join(fxa, how="outer")
df.index.name = "year"
df.to_csv(D / "processed" / "dataset_2008_2024_raw.csv")


filled = df.copy()
for c in ["import_qty_t", "import_value_kusd", "sec2_import_kusd"]:
    filled.loc[2012, c] = filled[c].interpolate(limit_area="inside").loc[2012]
filled.to_csv(D / "processed" / "dataset_2008_2024.csv")
print(filled.round(2).to_string())
