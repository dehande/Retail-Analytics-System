"""
====================================================================
Fashion Retail Store - Satış Analizi Görselleştirme
====================================================================
Kullanım:
    python 02_visualization.py

Gereksinimler:
    pip install pandas matplotlib seaborn
====================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ─── PATH ───────────────────────────────────────────────────────
TRANSACTIONS_PATH = "D:/AI_Lab/02_ml/datasets/transactions_features.csv"
# ────────────────────────────────────────────────────────────────

df = pd.read_csv(TRANSACTIONS_PATH)

MONTH_LABELS = {
    "2025-10": "Eki 25",
    "2025-11": "Kas 25",
    "2025-12": "Ara 25",
    "2026-01": "Oca 26",
}
df["month_label"] = df["month"].map(MONTH_LABELS)
month_order = list(MONTH_LABELS.values())

# ─── RENK PALETİ ────────────────────────────────────────────────
C_BLUE   = "#185FA5"
C_LIGHT  = "#B5D4F4"
C_GREEN  = "#1D9E75"
C_AMBER  = "#EF9F27"
C_CORAL  = "#D85A30"
C_RED    = "#A32D2D"
C_PURPLE = "#534AB7"
C_GRAY   = "#888780"

CAT_COLORS = [C_BLUE, C_GREEN, C_PURPLE, C_CORAL, C_AMBER,
              "#0F6E56", "#993556", "#3B6D11", "#854F0B", C_GRAY]

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":False,
    "axes.grid":         True,
    "axes.grid.axis":    "x",
    "grid.color":        "#ebebeb",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.titlepad":     10,
    "xtick.bottom":      False,
    "ytick.left":        False,
})

# ─── VERİ HAZIRLIK ──────────────────────────────────────────────

monthly = (df.groupby("month_label")
           .agg(gelir=("revenue_rub","sum"), ort_indirim=("discount_pct","mean"),
                adet=("quantity","sum"))
           .reindex(month_order))

cat_top10 = (df.groupby("category")
             .agg(gelir=("revenue_rub","sum"), ort_indirim=("discount_pct","mean"))
             .sort_values("gelir", ascending=False).head(10))

mgr = (df.groupby("manager")
       .agg(gelir=("revenue_rub","sum"), ort_indirim=("discount_pct","mean"))
       .sort_values("gelir"))

# İndirim bant dağılımı (aylık)
disc_bands = (df.groupby(["month_label","discount_band"])
              .size().unstack(fill_value=0)
              .reindex(month_order))
band_order  = ["full_price","light","moderate","deep","clearance"]
band_labels = {"full_price":"Tam fiyat","light":"Hafif %1-15",
               "moderate":"Orta %16-35","deep":"Derin %36-55","clearance":"Tasfiye %56+"}
band_colors = [C_BLUE, C_GREEN, C_AMBER, C_CORAL, C_RED]
disc_bands  = disc_bands.reindex(columns=[b for b in band_order if b in disc_bands.columns])

# Kategori indirim oranı (top 10)
cat_disc = cat_top10[["ort_indirim"]].copy()
cat_disc["tam_fiyat"] = 100 - cat_disc["ort_indirim"]
cat_disc = cat_disc.sort_values("ort_indirim")

# ─── FİGÜR ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle("Fashion Retail Store — Satış Analizi", fontsize=14, fontweight="bold", y=0.98)

gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.35)
ax1 = fig.add_subplot(gs[0, 0])   # Aylık gelir trendi
ax2 = fig.add_subplot(gs[0, 1])   # Kategori bazlı gelir
ax3 = fig.add_subplot(gs[1, 0])   # Satıcı performansı
ax4 = fig.add_subplot(gs[1, 1])   # Kategori indirim oranı
ax5 = fig.add_subplot(gs[2, :])   # İndirim bant dağılımı

# ── 1. Aylık gelir trendi (bar + çizgi) ──────────────────────────
bars = ax1.bar(month_order, monthly["gelir"] / 1e6, color=C_LIGHT, width=0.5, zorder=2)
ax1.set_title("Aylık gelir trendi")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₽{x:.1f}M"))
ax1.set_axisbelow(True)
ax1.grid(axis="y", color="#ebebeb", linewidth=0.6)
ax1.grid(axis="x", visible=False)

ax1b = ax1.twinx()
ax1b.plot(month_order, monthly["ort_indirim"], color=C_AMBER,
          marker="o", linewidth=2, markersize=5, zorder=3)
ax1b.set_ylabel("Ort. indirim %", color=C_AMBER, fontsize=9)
ax1b.tick_params(axis="y", labelcolor=C_AMBER, labelsize=9)
ax1b.spines["right"].set_visible(False)
ax1b.spines["top"].set_visible(False)
ax1b.set_ylim(0, 70)

for bar, val in zip(bars, monthly["gelir"]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"₽{val/1e6:.2f}M", ha="center", va="bottom", fontsize=8.5, color=C_BLUE, fontweight="bold")

# ── 2. Kategori bazlı gelir ───────────────────────────────────────
cat_names = [c[:12] for c in cat_top10.index]
ax2.barh(cat_names, cat_top10["gelir"] / 1e6, color=CAT_COLORS[:len(cat_top10)],
         height=0.6, zorder=2)
ax2.set_title("Kategori bazlı gelir (top 10)")
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₽{x:.1f}M"))
ax2.grid(axis="x", color="#ebebeb", linewidth=0.6)
ax2.grid(axis="y", visible=False)
ax2.set_axisbelow(True)

# ── 3. Satıcı performansı ─────────────────────────────────────────
mgr_short = [m.split()[0].capitalize() for m in mgr.index]
colors_mgr = [C_CORAL if d > 40 else C_BLUE for d in mgr["ort_indirim"]]
ax3.barh(mgr_short, mgr["gelir"] / 1e6, color=colors_mgr, height=0.6, zorder=2)
ax3.set_title("Satıcı performansı — toplam gelir\n(turuncu = ort. indirim >%40)")
ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₽{x:.1f}M"))
ax3.grid(axis="x", color="#ebebeb", linewidth=0.6)
ax3.grid(axis="y", visible=False)
ax3.set_axisbelow(True)

# ── 4. Kategori indirim oranı (yığılı yatay bar) ─────────────────
cat_disc_sorted = cat_disc.sort_values("ort_indirim")
cat_names_disc  = [c[:12] for c in cat_disc_sorted.index]
ax4.barh(cat_names_disc, cat_disc_sorted["tam_fiyat"], color=C_LIGHT,
         height=0.6, label="Tam fiyat", zorder=2)
ax4.barh(cat_names_disc, cat_disc_sorted["ort_indirim"],
         left=cat_disc_sorted["tam_fiyat"],
         color=C_CORAL, height=0.6, label="İndirimli", zorder=2)
ax4.set_title("Kategori indirim oranı")
ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"%{x:.0f}"))
ax4.set_xlim(0, 100)
ax4.grid(axis="x", color="#ebebeb", linewidth=0.6)
ax4.grid(axis="y", visible=False)
ax4.set_axisbelow(True)
ax4.legend(fontsize=8, loc="lower right", frameon=False)

# ── 5. İndirim bant dağılımı (aylık yığılı bar) ──────────────────
bottom = [0] * len(month_order)
for band, color in zip(band_order, band_colors):
    if band not in disc_bands.columns:
        continue
    vals = disc_bands[band].values
    ax5.bar(month_order, vals, bottom=bottom, color=color,
            label=band_labels[band], width=0.4, zorder=2)
    bottom = [b + v for b, v in zip(bottom, vals)]

ax5.set_title("İndirim bant dağılımı — aylık")
ax5.set_ylabel("İşlem sayısı")
ax5.legend(ncol=5, fontsize=8, loc="upper center",
           bbox_to_anchor=(0.5, -0.12), frameon=False)
ax5.grid(axis="y", color="#ebebeb", linewidth=0.6)
ax5.grid(axis="x", visible=False)
ax5.set_axisbelow(True)

plt.savefig("D:/AI_Lab/02_ml/datasets/satis_analizi.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
print("Grafik kaydedildi: satis_analizi.png")
