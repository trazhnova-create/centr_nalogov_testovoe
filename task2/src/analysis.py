"""Задание 2
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FIG, OUT = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)


BLUE, ORANGE, AQUA, VIOLET, GREY, INK, INK2 = "#5f9a3c", "#e4678a", "#a9c79a", "#7a4f9a", "#d9cfc9", "#3b3434", "#6b6060"
BG = "#fdf8f5"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.grid": True, "grid.color": "#efe4de", "grid.linewidth": .8,
    "axes.axisbelow": True, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight"})

df = pd.read_csv(ROOT / "data" / "processed" / "dataset_2008_2024.csv", index_col="year")
fx = df["byn_per_usd_avg"]
lfx = np.log(fx)

TARGETS = {  
    "import_qty":   ("import_qty_t",      "Импорт арбузов, количество (тонн)",        BLUE),
    "import_value": ("import_value_kusd", "Импорт арбузов, стоимость (тыс. долл.)",   BLUE),
    "export_qty":   ("export_qty_t",      "Экспорт арбузов, количество (тонн)",       ORANGE),
    "export_value": ("export_value_kusd", "Экспорт арбузов, стоимость (тыс. долл.)",  ORANGE),
}
H = 3
FUT = [2022, 2023, 2024]

def series(col):
    return df.loc[2008:2021, col].dropna()


def decompose(y, lam=6.25):
    """Годовые данные: сезонность не определена (период = 1 год). Раскладываем на
    тренд (фильтр Ходрика-Прескотта, lambda=6.25 по Ravn-Uhlig для годовых данных),
    цикл (сглаженное отклонение от тренда, центр. скользящее среднее по 3 годам) и остаток."""
    cyc_raw, trend = hpfilter(y, lamb=lam)[0], hpfilter(y, lamb=lam)[1]
    cycle = cyc_raw.rolling(3, center=True, min_periods=2).mean()
    resid = cyc_raw - cycle
    return pd.DataFrame({"observed": y, "trend": trend, "cycle": cycle, "residual": resid})

def plot_decomp(dec, title, unit, color, fname):
    fig, ax = plt.subplots(4, 1, figsize=(9, 8.2), sharex=True, gridspec_kw={"height_ratios": [1.6, 1, 1, 1]})
    ax[0].plot(dec.index, dec.observed, color=color, lw=2, marker="o", ms=4, label="Наблюдения")
    ax[0].plot(dec.index, dec.trend, color=INK, lw=2, ls="--", label="Тренд (HP)")
    ax[0].set_ylabel(unit, fontsize=8); ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    ax[0].set_title(title, loc="left", fontsize=11)
    ax[1].bar(dec.index, dec.cycle, color=AQUA, width=.7); ax[1].axhline(0, color=INK2, lw=.8); ax[1].set_ylabel("Цикл", fontsize=8)
    ax[2].bar(dec.index, dec.residual, color=VIOLET, alpha=.55, width=.7); ax[2].axhline(0, color=INK2, lw=.8); ax[2].set_ylabel("Остаток", fontsize=8)
    ax[3].set_ylim(-1, 1); ax[3].set_yticks([])
    ax[3].text(.5, .5, "Сезонная компонента не определена: данные годовые (период = 1 год)", transform=ax[3].transAxes,
               ha="center", va="center", fontsize=9, color=INK2); ax[3].set_ylabel("Сезонность", fontsize=8); ax[3].grid(False)
    ax[3].set_xticks(dec.index); ax[3].tick_params(axis="x", labelsize=8, rotation=90)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.close(fig)

dec_all = {}
for k, (col, lab, c) in TARGETS.items():
    d = decompose(series(col)); dec_all[k] = d
    plot_decomp(d, f"Декомпозиция: {lab}", lab.split("(")[-1].rstrip(")"), c, f"decomp_{k}.png")
for k, col, lab, c in [("sec2_export", "sec2_export_kusd", "Раздел II, экспорт, стоимость (тыс. долл.)", ORANGE),
                       ("sec2_import", "sec2_import_kusd", "Раздел II, импорт, стоимость (тыс. долл.)", BLUE)]:
    d = decompose(series(col)); dec_all[k] = d
    plot_decomp(d, f"Декомпозиция: {lab}", "тыс. долл.", c, f"decomp_{k}.png")
pd.concat({k: v for k, v in dec_all.items()}, names=["series", "year"]).round(2).to_csv(OUT / "decomposition.csv")


def f_naive(y, h, **kw):
    return np.repeat(y.iloc[-1], h), None

def f_drift(y, h, **kw):
    slope = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
    return y.iloc[-1] + slope * np.arange(1, h + 1), None

def f_ets(y, h, alpha=0.2, **kw):
    try:
        res = ETSModel(y.astype(float).reset_index(drop=True), error="add", trend="add", damped_trend=True).fit(disp=False)
        fc = res.forecast(h).values
        sf = res.get_prediction(start=len(y), end=len(y) + h - 1).summary_frame(alpha=alpha)
        return fc, (sf["pi_lower"].values, sf["pi_upper"].values)
    except Exception:
        return f_naive(y, h)

def _best_sarimax(y, exog=None):
    best = None
    for p in (0, 1):
        for d in (0, 1):
            for q in (0, 1):
                for tr in (["c"] if d == 0 else ["n", "t"]):
                    if p == q == d == 0 and tr == "n": continue
                    try:
                        m = SARIMAX(y.astype(float).values, exog=exog, order=(p, d, q), trend=tr,
                                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                        if best is None or m.aic < best[0]:
                            best = (m.aic, m, (p, d, q, tr))
                    except Exception:
                        pass
    return best

def f_arima(y, h, alpha=0.2, use_fx=False, **kw):
        try:
        ex = ex_f = None
        if use_fx:
            ex = lfx.loc[y.index].values.reshape(-1, 1)
            ex_f = lfx.loc[range(y.index[-1] + 1, y.index[-1] + 1 + h)].values.reshape(-1, 1)
        aic, m, order = _best_sarimax(y, ex)
        pr = m.get_forecast(h, exog=ex_f)
        ci = pr.conf_int(alpha=alpha)
        return pr.predicted_mean, (ci[:, 0], ci[:, 1])
    except Exception:
        return f_naive(y, h)

def f_arimax(y, h, **kw):
    return f_arima(y, h, use_fx=True, **kw)

def _ml_forecast(model_fn, y, h):
    """Таблица признаков: y(t-1), прирост y(t-1)-y(t-2), курс y(t) и его прирост. Цель: прирост y(t)-y(t-1)."""
    yv = y.astype(float)
    yrs = list(yv.index)
    X, T = [], []
    for i in range(2, len(yv)):
        t = yrs[i]
        X.append([yv.iloc[i-1], yv.iloc[i-1] - yv.iloc[i-2], lfx[t], lfx[t] - lfx[t-1]])
        T.append(yv.iloc[i] - yv.iloc[i-1])
    X, T = np.array(X), np.array(T)
    m = model_fn().fit(X, T)
    hist = list(yv.values); out = []
    t = yrs[-1]
    for _ in range(h):
        t += 1
        x = [hist[-1], hist[-1] - hist[-2], lfx[t], lfx[t] - lfx[t-1]]
        nxt = max(0.0, hist[-1] + m.predict(np.array([x]))[0])
        out.append(nxt); hist.append(nxt)
    return np.array(out), None

def f_gbr(y, h, **kw):
    return _ml_forecast(lambda: GradientBoostingRegressor(n_estimators=150, learning_rate=.05, max_depth=2,
                                                          subsample=.8, random_state=0), y, h)

def f_rf(y, h, **kw):
    return _ml_forecast(lambda: RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=2, random_state=0), y, h)

MODELS = {  
    "Naive (последнее значение)": (f_naive, "baseline"),
    "Drift":                      (f_drift, "baseline"),
    "ETS (Holt, затухающий тренд)": (f_ets, "stat"),
    "ARIMA (подбор по AIC)":      (f_arima, "stat"),
    "ARIMAX (+ курс BYN/USD)":    (f_arimax, "stat"),
    "Gradient Boosting":          (f_gbr, "ml"),
    "Random Forest":              (f_rf, "ml"),
}

def smape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    den = (np.abs(a) + np.abs(f)); r = np.where(den == 0, 0, 2 * np.abs(a - f) / np.where(den == 0, 1, den))
    return 100 * r.mean()


ORIGINS = [2014, 2015, 2016, 2017, 2018]   
cv_rows, cv_err = [], []
for tk, (col, lab, _) in TARGETS.items():
    y_full = series(col)
    for name, (fn, grp) in MODELS.items():
        A, F, per_h = [], [], {1: [], 2: [], 3: []}
        for e in ORIGINS:
            tr = y_full.loc[:e]; te = y_full.loc[e + 1:e + H]
            hh = len(te)
            fc, _ = fn(tr, hh)
            fc = np.maximum(np.asarray(fc, float), 0)
            for j, (yr, a) in enumerate(te.items()):
                A.append(a); F.append(fc[j]); per_h[j + 1].append(a - fc[j])
                cv_err.append((tk, name, e, yr, j + 1, a, fc[j]))
        A, F = np.array(A), np.array(F)
        cv_rows.append(dict(target=tk, model=name, group=grp, MAE=np.abs(A - F).mean(),
                            RMSE=np.sqrt(((A - F) ** 2).mean()), sMAPE=smape(A, F),
                            WAPE=100 * np.abs(A - F).sum() / max(np.abs(A).sum(), 1e-9), n=len(A)))
cv = pd.DataFrame(cv_rows)
naive_rmse = cv[cv.model.str.startswith("Naive")].set_index("target")["RMSE"]
cv["RMSE_vs_naive"] = cv.apply(lambda r: r.RMSE / naive_rmse[r.target], axis=1)
cv.round(3).to_csv(OUT / "cv_metrics.csv", index=False)
pd.DataFrame(cv_err, columns=["target", "model", "origin", "year", "h", "actual", "forecast"]).round(3).to_csv(OUT / "cv_forecasts.csv", index=False)


best = {}
for tk in TARGETS:
    sub = cv[cv.target == tk].sort_values("RMSE")
    best[tk] = dict(overall=sub.iloc[0].model,
                    stat=sub[sub.group.isin(["stat"])].iloc[0].model,
                    ml=sub[sub.group == "ml"].iloc[0].model)
print(cv.pivot(index="model", columns="target", values="RMSE").round(1).to_string())
print(best)


errs = pd.DataFrame(cv_err, columns=["target", "model", "origin", "year", "h", "actual", "forecast"])
errs["e"] = errs.actual - errs.forecast

def emp_interval(tk, name, fc, z=1.2816):
    s = errs[(errs.target == tk) & (errs.model == name)].groupby("h")["e"].apply(lambda v: np.sqrt((v ** 2).mean()))
    s = s.reindex([1, 2, 3]).ffill().cummax().values[:len(fc)]
    return fc - z * s, fc + z * s


fc_rows = []
fc_store = {}
for tk, (col, lab, _) in TARGETS.items():
    y = series(col)
    for name, (fn, grp) in MODELS.items():
        fc, ci = fn(y, H, alpha=0.2)
        fc = np.maximum(np.asarray(fc, float), 0)
        if ci is None:
            lo, hi = emp_interval(tk, name, fc)
        else:
            lo, hi = np.asarray(ci[0], float), np.asarray(ci[1], float)
        lo = np.maximum(lo, 0)
        fc_store[(tk, name)] = (fc, lo, hi)
        for i, yr in enumerate(FUT):
            fc_rows.append(dict(target=tk, model=name, group=grp, year=yr, forecast=fc[i], lo80=lo[i], hi80=hi[i],
                                is_best=(name == best[tk]["overall"])))
fcdf = pd.DataFrame(fc_rows)
fcdf.round(2).to_csv(OUT / "forecasts_all_models.csv", index=False)
fcdf[fcdf.is_best].round(2).to_csv(OUT / "forecasts_best_model.csv", index=False)
print(fcdf[fcdf.is_best][["target", "model", "year", "forecast", "lo80", "hi80"]].round(1).to_string(index=False))


fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, (tk, (col, lab, c)) in zip(axes.flat, TARGETS.items()):
    y = series(col)
    ax.plot(y.index, y.values, color=c, lw=2, marker="o", ms=4, label="Факт")
    b = best[tk]["overall"]
    fc, lo, hi = fc_store[(tk, b)]
    xs = [y.index[-1]] + FUT
    ax.fill_between(FUT, lo, hi, color=c, alpha=.15, label="80% интервал (лучшая модель)")
    ax.plot(xs, [y.iloc[-1], *fc], color=c, lw=2, ls="--", marker="s", ms=5, label=f"Лучшая: {b}")
    for grp, ls, col2, nm in [("stat", ":", INK, best[tk]["stat"]), ("ml", ":", AQUA, best[tk]["ml"])]:
        if nm != b:
            f2 = fc_store[(tk, nm)][0]
            ax.plot(xs, [y.iloc[-1], *f2], color=col2, lw=1.4, ls=ls, label=f"{'Стат.' if grp=='stat' else 'ML'}: {nm}")
    ax.axvline(2021.5, color=GREY, lw=1)
    ax.set_title(lab, loc="left", fontsize=10); ax.legend(frameon=False, fontsize=7, loc="upper left"); ax.tick_params(labelsize=8)
fig.suptitle("Прогноз 2022–2024 гг.: арбузы свежие (ТН ВЭД 080711)", x=0.01, ha="left", fontsize=13)
fig.tight_layout(); fig.savefig(FIG / "forecast_all.png"); plt.close(fig)


piv = cv.pivot(index="model", columns="target", values="RMSE_vs_naive").loc[list(MODELS)]
fig, ax = plt.subplots(figsize=(11, 4.6))
w = .2
cols = [BLUE, "#b5d1a4", ORANGE, "#f3b4c5"]
for i, tk in enumerate(TARGETS):
    ax.bar(np.arange(len(piv)) + i * w, piv[tk].clip(upper=4), width=w, color=cols[i], label=TARGETS[tk][1].split(" (")[0].replace(" арбузов, ", ", "))
ax.axhline(1, color=INK, lw=1, ls="--"); ax.text(len(piv) - .5, 1.04, "naive = 1", fontsize=8, color=INK2, ha="right")
ax.set_xticks(np.arange(len(piv)) + 1.5 * w); ax.set_xticklabels([m.replace(" (", "\n(") for m in piv.index], fontsize=8)
ax.set_ylabel("RMSE / RMSE naive (меньше = лучше)"); ax.legend(frameon=False, fontsize=8, ncol=2)
ax.set_title("Качество моделей на скользящей валидации (горизонт 3 года)", loc="left")
fig.tight_layout(); fig.savefig(FIG / "model_comparison.png"); plt.close(fig)


fig, axes = plt.subplots(2, 2, figsize=(12, 6.2), sharex=True)
for ax, (tk, (col, lab, c)) in zip(axes.flat, TARGETS.items()):
    ax.plot(df.loc[2008:2021, col].index, df.loc[2008:2021, col].values, color=c, lw=2, marker="o", ms=4)
    if tk.startswith("import"):
        ax.scatter([2012], [df.loc[2012, col]], color="white", edgecolor=INK, s=55, zorder=4, label="2012: интерполяция")
        ax.legend(frameon=False, fontsize=8)
    ax.set_title(lab, loc="left", fontsize=10); ax.tick_params(labelsize=8)
fig.suptitle("Исходные данные: арбузы свежие (080711), 2008–2021", x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(FIG / "raw_data.png"); plt.close(fig)


corr_rows = []
cols_c = {"Импорт арбузов, стоимость": "import_value_kusd", "Импорт арбузов, количество": "import_qty_t",
          "Экспорт арбузов, стоимость": "export_value_kusd", "Экспорт арбузов, количество": "export_qty_t",
          "Раздел II, импорт, стоимость": "sec2_import_kusd", "Раздел II, экспорт, стоимость": "sec2_export_kusd"}
for lab, col in cols_c.items():
    y = df.loc[2008:2021, col].dropna()
    for tr_name, yy, xx in [("уровни", y, fx.loc[y.index]),
                            ("приросты", y.diff().dropna(), fx.diff().loc[y.diff().dropna().index]),
                            ("уровни, курс t-1", y.iloc[1:], fx.shift(1).loc[y.index[1:]])]:
        ok = yy.notna() & xx.notna()
        if ok.sum() >= 4:
            r, p = stats.pearsonr(xx[ok], yy[ok]); rs, ps = stats.spearmanr(xx[ok], yy[ok])
            corr_rows.append(dict(ряд=lab, вариант=tr_name, n=int(ok.sum()), pearson_r=r, pearson_p=p, spearman_rho=rs, spearman_p=ps))
corr = pd.DataFrame(corr_rows)
corr.round(3).to_csv(OUT / "correlations.csv", index=False)
print(corr.round(3).to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, (lab, col) in zip(axes.flat, cols_c.items()):
    y = df.loc[2008:2021, col].dropna(); x = fx.loc[y.index]
    ax.scatter(x, y, color=ORANGE if "Экспорт" in lab else BLUE, s=45, edgecolor="white", linewidth=1.2, zorder=3)
    for xi, yi, yr in zip(x, y, y.index):
        ax.annotate(str(yr)[2:], (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points", color=INK2)
    r, p = stats.pearsonr(x, y)
    ax.set_title(f"{lab}\nr = {r:.2f}, p = {p:.2f}", loc="left", fontsize=9)
    ax.set_xlabel("BYN за 1 USD (среднегодовой)", fontsize=8); ax.tick_params(labelsize=8)
fig.suptitle("Связь стоимости торговли с курсом BYN/USD (подписи — год)", x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(FIG / "correlation_fx.png"); plt.close(fig)


fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(fx.index, fx.values, color=VIOLET, lw=2, marker="o", ms=4)
ax.set_title("Среднегодовой официальный курс, BYN за 1 USD (после деноминации 2016 г.)", loc="left", fontsize=10)
ax.axvspan(2021.5, 2024.5, color="#f6ece7"); ax.text(2023, fx.min(), "период прогноза", ha="center", fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(FIG / "fx_rate.png"); plt.close(fig)
print("done")
