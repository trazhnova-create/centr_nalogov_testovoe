"""Шаг 2. Кластеризация стран по структуре торговли (K-means, DBSCAN), профили, PCA.
Запуск (после make_matrix.py):  python src/clustering.py
"""
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
PROC, FIG, RES = ROOT / "data" / "processed", ROOT / "figures", ROOT / "results"
SEED = 0

GREEN, PINK, SAGE, VIOLET, PEACH, TEAL = "#5f9a3c", "#e4678a", "#a9c79a", "#7a4f9a", "#e8a87c", "#6fa8a0"
GREY, INK, INK2, BG = "#d9cfc9", "#3b3434", "#6b6060", "#fdf8f5"
CL_COL = [GREEN, PINK, VIOLET, PEACH, TEAL, SAGE]
HEAT = LinearSegmentedColormap.from_list("matcha", ["#fdf8f5", "#cfe0c3", "#5f9a3c"])
DIV = LinearSegmentedColormap.from_list("div", ["#e4678a", "#fdf8f5", "#5f9a3c"])
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": GREY, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})

X = pd.read_csv(PROC / "country_share_matrix_SYNTHETIC.csv", index_col=0)
arch = pd.read_csv(PROC / "country_archetype_SYNTHETIC.csv", index_col=0).iloc[:, 0]
GROUPS = list(X.columns)

Z = StandardScaler().fit_transform(X)

ks = list(range(2, 9))
rows = []
for k in ks:
    km = KMeans(k, n_init=30, random_state=SEED).fit(Z)
    rows.append((k, km.inertia_, silhouette_score(Z, km.labels_), davies_bouldin_score(Z, km.labels_),
                 int(np.bincount(km.labels_).min())))
kdf = pd.DataFrame(rows, columns=["k", "inertia", "silhouette", "davies_bouldin", "min_размер_кластера"])
kdf.round(3).to_csv(RES / "kmeans_choose_k.csv", index=False, encoding="utf-8-sig")
valid = kdf[kdf["min_размер_кластера"] >= 3]
K = int(valid.loc[valid.silhouette.idxmax(), "k"])
print(kdf.round(3).to_string(index=False), f"\n-> выбрано k = {K}")

fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.3))
for a, col, t in zip(ax, ["inertia", "silhouette", "davies_bouldin"],
                     ["Инерция (метод локтя)", "Silhouette (больше — лучше)", "Davies–Bouldin (меньше — лучше)"]):
    a.plot(kdf.k, kdf[col], marker="o", color=GREEN, lw=2)
    a.axvline(K, color=PINK, ls="--", lw=1.2); a.set_title(t); a.set_xlabel("число кластеров k")
fig.suptitle(f"Выбор числа кластеров K-means: выбрано k = {K}", x=0.01, ha="left", fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(FIG / "01_choose_k.png", dpi=150); plt.close(fig)

# K-means 
km = KMeans(K, n_init=50, random_state=SEED).fit(Z)

#DBSCAN
MIN_S = 3
d = np.sort(NearestNeighbors(n_neighbors=MIN_S).fit(Z).kneighbors(Z)[0][:, -1])
fig, ax = plt.subplots(figsize=(5.5, 3.3))
ax.plot(d, color=GREEN, lw=2); ax.set_title(f"DBSCAN: расстояние до {MIN_S}-го соседа")
ax.set_xlabel("страны, отсортированные по расстоянию"); ax.set_ylabel("расстояние")
fig.tight_layout(); fig.savefig(FIG / "02_dbscan_kdistance.png", dpi=150); plt.close(fig)

grid = []
for eps in np.round(np.arange(1.5, 4.01, 0.25), 2):
    lab = DBSCAN(eps=eps, min_samples=MIN_S).fit_predict(Z)
    ncl, noise = len(set(lab)) - (1 if -1 in lab else 0), int((lab == -1).sum())
    mask = lab != -1
    sil = silhouette_score(Z[mask], lab[mask]) if ncl > 1 and mask.sum() > ncl else np.nan
    grid.append((eps, ncl, noise, sil))
gdf = pd.DataFrame(grid, columns=["eps", "кластеров", "шум", "silhouette"])
gdf.round(3).to_csv(RES / "dbscan_grid.csv", index=False, encoding="utf-8-sig")
print("\n", gdf.round(3).to_string(index=False))

ok = gdf[(gdf["кластеров"] >= 2) & (gdf["шум"] <= 0.15 * len(X))].dropna()
EPS = float(ok.loc[ok.silhouette.idxmax(), "eps"])
db_lab = DBSCAN(eps=EPS, min_samples=MIN_S).fit_predict(Z)
print(f"-> eps = {EPS}")

# Сравнение алгоритмов
ag_lab = AgglomerativeClustering(K, linkage="ward").fit_predict(Z)
m = db_lab != -1
cmp = pd.DataFrame({
    "алгоритм": ["K-means", "Ward (контроль)", f"DBSCAN (eps={EPS}, min={MIN_S})"],
    "кластеров": [K, K, len(set(db_lab)) - (1 if -1 in db_lab else 0)],
    "стран в шуме": [0, 0, int((~m).sum())],
    "silhouette": [silhouette_score(Z, km.labels_), silhouette_score(Z, ag_lab),
                   silhouette_score(Z[m], db_lab[m]) if len(set(db_lab[m])) > 1 else np.nan],
    "ARI с K-means": [1.0, adjusted_rand_score(km.labels_, ag_lab), adjusted_rand_score(km.labels_[m], db_lab[m])],
}).round(3)
cmp.to_csv(RES / "algorithm_comparison.csv", index=False, encoding="utf-8-sig")
print("\n", cmp.to_string(index=False))

#Итоговые кластеры, имена и профили 
res = X.copy()
res["cluster_raw"] = km.labels_
mean_all = X.mean()

order = res.cluster_raw.value_counts().index.tolist()
res["Кластер"] = res.cluster_raw.map({c: i + 1 for i, c in enumerate(order)})
prof = res.groupby("Кластер")[GROUPS].mean() * 100            
lift = prof / (mean_all * 100)                                 
names = {}
for c in prof.index:
    
    diff = (prof.loc[c] - mean_all * 100).sort_values(ascending=False)
    top = [g for g in diff.index[:2] if diff[g] >= 5] or [diff.index[0]]   
    names[c] = f"К{c}: " + " + ".join(top)
res["Профиль"] = res["Кластер"].map(names)
res["DBSCAN"] = db_lab
res["заложенный_архетип"] = arch
res.drop(columns="cluster_raw").round(4).to_csv(RES / "country_clusters_SYNTHETIC.csv", encoding="utf-8-sig", index_label="Страна")
prof.round(1).to_csv(RES / "cluster_profiles.csv", encoding="utf-8-sig")
lift.round(2).to_csv(RES / "cluster_lift.csv", encoding="utf-8-sig")
for c in prof.index:
    print(f"\n{names[c]}  (стран: {(res.Кластер == c).sum()})")
    print("  топ-3 доли:", ", ".join(f"{g} {prof.loc[c, g]:.0f}%" for g in prof.loc[c].sort_values(ascending=False).index[:3]))
    print("  страны:", ", ".join(res.index[res.Кластер == c]))
print("\nDBSCAN-шум:", list(res.index[res.DBSCAN == -1]))
print("ARI K-means vs заложенные архетипы:", round(adjusted_rand_score(arch, res.Кластер), 3))

# Графики
cc = {c: CL_COL[i % len(CL_COL)] for i, c in enumerate(prof.index)}

#тепловая карта профилей (средние доли, %)
fig, ax = plt.subplots(figsize=(12, 0.7 * len(prof) + 2.4))
im = ax.imshow(prof.values, cmap=HEAT, aspect="auto")
ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([textwrap.fill(g, 15, break_long_words=False) for g in GROUPS], fontsize=8.5)
ax.set_yticks(range(len(prof))); ax.set_yticklabels([f"{names[c]}\n(n={(res.Кластер == c).sum()})" for c in prof.index], fontsize=8.5)
for i in range(prof.shape[0]):
    for j in range(prof.shape[1]):
        v = prof.values[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 28 else INK)
ax.set_title("Профили кластеров: средняя доля товарной группы в торговле, %")
for s in ax.spines.values(): s.set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
fig.tight_layout(); fig.savefig(FIG / "03_cluster_profiles_heatmap.png", dpi=150); plt.close(fig)


fig, axes = plt.subplots(1, len(prof), figsize=(3.0 * len(prof), 4.2), sharey=True)
axes = np.atleast_1d(axes)
for a, c in zip(axes, prof.index):
    v = (lift.loc[c] - 1) * 100
    a.barh(range(len(GROUPS)), v.values, color=[cc[c] if x > 0 else GREY for x in v.values])
    a.axvline(0, color=INK2, lw=0.8); a.set_title(names[c].split(":")[0] + f" (n={(res.Кластер == c).sum()})", fontsize=10)
    a.set_yticks(range(len(GROUPS))); a.set_yticklabels(GROUPS if a is axes[0] else [], fontsize=8.5)
    a.invert_yaxis(); a.set_xlabel("% к среднему по всем странам", fontsize=8)
fig.suptitle("Чем кластер отличается от «средней страны»", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "04_cluster_vs_average.png", dpi=150); plt.close(fig)

#PCA
pca = PCA(n_components=2, random_state=SEED).fit(Z)
P = pca.transform(Z); ev = pca.explained_variance_ratio_
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
a = ax[0]
for c in prof.index:
    idx = (res.Кластер == c).values
    a.scatter(P[idx, 0], P[idx, 1], s=70, color=cc[c], edgecolor="white", lw=0.8, label=names[c], zorder=3)
for i, n in enumerate(X.index):
    a.annotate(n.replace(" (синт.)", "*"), (P[i, 0], P[i, 1]), fontsize=6.8, color=INK2, xytext=(3, 3), textcoords="offset points")
a.set_xlabel(f"PC1 ({ev[0]*100:.0f}% дисперсии)"); a.set_ylabel(f"PC2 ({ev[1]*100:.0f}% дисперсии)")
a.set_title("Страны в пространстве главных компонент"); a.legend(fontsize=8, frameon=False, loc="best"); a.grid(color=GREY, lw=0.4, alpha=0.6)
a = ax[1]
load = pd.DataFrame(pca.components_.T, index=GROUPS, columns=["PC1", "PC2"])
for g in GROUPS:
    a.arrow(0, 0, load.loc[g, "PC1"], load.loc[g, "PC2"], color=GREEN, head_width=0.02, lw=1.4, length_includes_head=True)
    a.text(load.loc[g, "PC1"] * 1.12, load.loc[g, "PC2"] * 1.12, g, fontsize=7.5, ha="center", color=INK)
a.set_xlim(-0.85, 0.85); a.set_ylim(-0.85, 0.85); a.axhline(0, color=GREY, lw=0.6); a.axvline(0, color=GREY, lw=0.6)
a.set_title("Вклад товарных групп в компоненты"); a.set_xlabel("PC1"); a.set_ylabel("PC2")
fig.tight_layout(); fig.savefig(FIG / "05_pca_clusters.png", dpi=150); plt.close(fig)
load.round(3).to_csv(RES / "pca_loadings.csv", encoding="utf-8-sig")
print(f"\nPCA: PC1 {ev[0]*100:.1f}%, PC2 {ev[1]*100:.1f}%, всего {ev.sum()*100:.1f}%")

# K-means vs DBSCAN на PCA
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True, sharey=True)
for a, lab, t in [(ax[0], res.Кластер.values, f"K-means (k={K})"), (ax[1], db_lab, f"DBSCAN (eps={EPS})")]:
    for l in sorted(set(lab)):
        idx = lab == l
        col = GREY if l == -1 else CL_COL[(l - 1 if t.startswith("K") else l) % len(CL_COL)]
        a.scatter(P[idx, 0], P[idx, 1], s=60, color=col, lw=0.8, **({} if l == -1 else {"edgecolor": "white"}),
                  marker="x" if l == -1 else "o", label="шум" if l == -1 else f"кластер {l}")
    a.set_title(t); a.legend(fontsize=8, frameon=False); a.set_xlabel("PC1")
ax[0].set_ylabel("PC2")
fig.tight_layout(); fig.savefig(FIG / "06_kmeans_vs_dbscan.png", dpi=150); plt.close(fig)

# стековые доли
fig, ax = plt.subplots(figsize=(11, 0.55 * len(res) * 0.45 + 3))
srt = res.sort_values(["Кластер"] + [GROUPS[0]])
bottoms = np.zeros(len(srt)); pal = [GREEN, PINK, SAGE, VIOLET, PEACH, TEAL, "#c9a14a", GREY]
for g, col in zip(GROUPS, pal):
    ax.barh(range(len(srt)), srt[g] * 100, left=bottoms, color=col, label=g, edgecolor=BG, lw=0.4); bottoms += srt[g].values * 100
ax.set_yticks(range(len(srt))); ax.set_yticklabels([f"{n}  [К{c}]" for n, c in zip(srt.index, srt.Кластер)], fontsize=7)
ax.invert_yaxis(); ax.set_xlabel("доля в торговле, %"); ax.set_title("Структура торговли каждой страны (сортировка по кластерам)")
ax.legend(ncol=4, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
fig.tight_layout(); fig.savefig(FIG / "07_country_structure.png", dpi=150); plt.close(fig)
