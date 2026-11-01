"""Блок «Сезонность»"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FIG, OUT = ROOT / "figures", ROOT / "results"
BLUE, ORANGE, AQUA, VIOLET, GREY, INK, INK2 = "#5f9a3c", "#e4678a", "#a9c79a", "#7a4f9a", "#d9cfc9", "#3b3434", "#6b6060"
BG = "#fdf8f5"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": "#efe4de", "grid.linewidth": .8, "axes.axisbelow": True, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "figure.dpi": 110,
    "savefig.dpi": 140, "savefig.bbox": "tight"})

df = pd.read_csv(ROOT / "data" / "processed" / "dataset_2008_2024.csv", index_col="year")
PROFILE = {
 "import": [.02, .02, .02, .03, .05, .09, .17, .24, .21, .09, .04, .02],
 "export": [.01, .01, .01, .01, .02, .06, .20, .33, .25, .07, .02, .01],
}
assert all(abs(sum(v) - 1) < 1e-9 for v in PROFILE.values())
KAPPA = 60  
TARGETS = {
 "import_qty": ("import_qty_t", "Импорт, количество (т)", "import", BLUE),
 "import_value": ("import_value_kusd", "Импорт, стоимость (тыс. $)", "import", BLUE),
 "export_qty": ("export_qty_t", "Экспорт, количество (т)", "export", ORANGE),
 "export_value": ("export_value_kusd", "Экспорт, стоимость (тыс. $)", "export", ORANGE),
}

def disaggregate(annual, flow, seed):
    rng = np.random.default_rng(seed)
    base = np.array(PROFILE[flow])
    rows = []
    for yr, tot in annual.dropna().loc[:2021].items():
        w = rng.dirichlet(base * KAPPA)
        for m in range(12):
            rows.append((pd.Timestamp(yr, m + 1, 1), tot * w[m]))
    return pd.Series(dict(rows)).asfreq("MS")

monthly = {}
for i, (k, (col, lab, flow, c)) in enumerate(TARGETS.items()):
        monthly[k] = disaggregate(df[col], flow, seed=100 + (0 if flow == "import" else 1))
mdf = pd.DataFrame(monthly); mdf.index.name = "month"
mdf.round(3).to_csv(ROOT / "data" / "processed" / "monthly_SYNTHETIC.csv")
chk = mdf.groupby(mdf.index.year).sum()
assert np.allclose(chk.loc[2013, "import_qty"], df.loc[2013, "import_qty_t"])  # суммы совпадают с годовыми

# --- STL-декомпозиция
fig, axes = plt.subplots(4, 4, figsize=(17, 9.5), sharex="col")
dec_rows = []
for j, (k, (col, lab, flow, c)) in enumerate(TARGETS.items()):
    y = mdf[k].dropna()
    res = STL(y, period=12, robust=True, seasonal=13).fit()
    for i, (nm, comp) in enumerate([("Наблюдения", y), ("Тренд", res.trend), ("Сезонность", res.seasonal), ("Остаток", res.resid)]):
        ax = axes[i, j]; ax.plot(comp.index, comp.values, color=c if i == 0 else [None, INK, AQUA, VIOLET][i], lw=1.2)
        if i == 0: ax.set_title(lab, loc="left", fontsize=10)
        if j == 0: ax.set_ylabel(nm, fontsize=9)
        ax.tick_params(labelsize=7)
    dec_rows.append(pd.DataFrame({"series": k, "observed": y, "trend": res.trend, "seasonal": res.seasonal, "resid": res.resid}))
fig.suptitle("STL-декомпозиция месячных рядов (синтетическая разбивка годовых данных Белстата по сезонному профилю)",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(FIG / "stl_monthly_SYNTHETIC.png"); plt.close(fig)
pd.concat(dec_rows).round(3).to_csv(OUT / "stl_monthly_SYNTHETIC.csv")


fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
mon = ["Я", "Ф", "М", "А", "М", "И", "И", "А", "С", "О", "Н", "Д"]
for ax, flow, k, c in [(axes[0], "import", "import_qty", BLUE), (axes[1], "export", "export_qty", ORANGE)]:
    st = pd.concat(dec_rows); st = st[st.series == k]
    est = st.groupby(st.index.month)["seasonal"].mean()
    share = mdf[k].groupby(mdf.index.month).sum() / mdf[k].sum()
    ax.bar(np.arange(12) - .2, np.array(PROFILE[flow]) * 100, width=.4, color=GREY, label="заложенный профиль, % года")
    ax.bar(np.arange(12) + .2, share.values * 100, width=.4, color=c, label="фактическая доля в синтетическом ряду")
    ax.set_xticks(range(12)); ax.set_xticklabels(mon); ax.set_title(("Импорт" if flow == "import" else "Экспорт") + ": сезонность по месяцам", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7); ax.set_ylabel("% годового объёма", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "seasonal_profile_SYNTHETIC.png"); plt.close(fig)


ann = pd.read_csv(OUT / "forecasts_best_model.csv")
rows = []; fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
for ax, (k, (col, lab, flow, c)) in zip(axes.flat, TARGETS.items()):
    y = mdf[k].dropna()
    mod = SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sims = mod.simulate(nsimulations=36, repetitions=400, anchor="end")
    sims = sims.values.reshape(36, -1).clip(min=0)
    point = mod.get_forecast(36).predicted_mean.values.clip(min=0)
    fut_idx = pd.date_range("2022-01-01", periods=36, freq="MS")
    yr = fut_idx.year
    for Y in (2022, 2023, 2024):
        m = yr == Y
        tot = sims[m].sum(axis=0)
        rows.append(dict(target=k, year=Y, sarima_point=point[m].sum(), lo80=np.quantile(tot, .1), hi80=np.quantile(tot, .9),
                         annual_best=float(ann[(ann.target == k) & (ann.year == Y)].forecast.iloc[0])))
    hist = y.groupby(y.index.year).sum()
    r = pd.DataFrame([x for x in rows if x["target"] == k])
    ax.plot(hist.index, hist.values, color=c, lw=2, marker="o", ms=4, label="Факт (год)")
    ax.fill_between(r.year, r.lo80, r.hi80, color=VIOLET, alpha=.15, label="80% интервал, месячный SARIMA")
    ax.plot(r.year, r.sarima_point, color=VIOLET, lw=2, ls="--", marker="s", ms=5, label="SARIMA(0,1,1)(0,1,1)12, сумма за год")
    ax.plot(r.year, r.annual_best, color=INK, lw=1.4, ls=":", marker="o", ms=4, label="Лучшая годовая модель (блок 4)")
    ax.set_title(lab, loc="left", fontsize=10); ax.legend(frameon=False, fontsize=7, loc="upper left"); ax.tick_params(labelsize=8)
fig.suptitle("Сезонный месячный SARIMA, агрегированный до годового прогноза (на синтетической помесячной разбивке)", x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(FIG / "forecast_monthly_sarima_SYNTHETIC.png"); plt.close(fig)
out = pd.DataFrame(rows).round(1); out.to_csv(OUT / "forecast_monthly_sarima_SYNTHETIC.csv", index=False)
print(out.to_string(index=False))
