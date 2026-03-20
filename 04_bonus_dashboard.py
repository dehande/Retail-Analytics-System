"""
====================================================================
Fashion Retail Store - Satış Danışmanı Prim & Performans Sistemi
====================================================================
Modül B — Prim Hesaplama
Modül C — Satıcı Performans Dashboardu

Prim Formülü:
    Prim = Ciro × %3 × Performans Skoru

    Performans Skoru:
        Normal dönem  → Ciro %40 + İndirim %30 + Ticket %20 + Premium %10
        Zorunlu dönem → Ciro %55 + Ticket %25 + Premium %20

Zorunlu İndirim Dönemi:
    Aralık 11-31 ve Ocak tamamı
    Varsayılan oran: %50 (kategori bazlı oranlar eklenince güncelle)

Kullanım:
    python 04_bonus_dashboard.py

Gereksinimler:
    pip install pandas matplotlib
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

# ─── PATH ────────────────────────────────────────────────────────
TRANSACTIONS_PATH = "D:/AI_Lab/02_ml/datasets/transactions_features.csv"
SAVE_PATH         = "D:/AI_Lab/02_ml/datasets/bonus_dashboard.png"
# ─────────────────────────────────────────────────────────────────

# ─── ZORUNLU İNDİRİM DÖNEMİ ─────────────────────────────────────
# Kategori bazlı zorunlu indirim oranları eklenince burası güncellenir
# Şimdilik tüm kategoriler için %50 varsayımı
MANDATORY_DISCOUNT_RATE = {
    "jacket":               50,
    "sweater":              50,
    "sweatshirt":           50,
    "shirt":                50,
    "trousers":             50,
    "T-shirt":              50,
    "vest":                 50,
    "cardigan":             50,
    "sports trousers":      50,
    "polo shirt":           50,
    "hat":                  50,
    "blouse":               50,
    "bag":                  50,
    "belt":                 50,
    "cap":                  50,
    "coat":                 50,
    "gloves":               50,
    "scarf":                50,
    "DEFAULT":              50,   # Bilinmeyen kategoriler için
}

# Zorunlu dönem flag'i: (ay, gün_başlangıç) → o günden itibaren zorunlu
MANDATORY_PERIODS = [
    ("2025-12", 11),   # Aralık 11'den itibaren
    ("2026-01", 1),    # Ocak tamamı
]

# ─── PRİM PARAMETRELERİ ──────────────────────────────────────────
BONUS_RATE        = 0.03    # Cirönun %3'ü (baz oran, tam hedefe ulaşınca)
MAX_SCORE_MULTIPLIER = 1.3  # Hedefi aşma bonusu max %30

# Ağırlıklar — normal dönem
W_NORMAL = {
    "ciro":    0.40,
    "indirim": 0.30,
    "ticket":  0.20,
    "premium": 0.10,
}

# Ağırlıklar — zorunlu indirim döneminde indirim skoru hesaplanmaz
W_MANDATORY = {
    "ciro":    0.55,
    "indirim": 0.00,
    "ticket":  0.25,
    "premium": 0.20,
}

# ─── RENK PALETİ ────────────────────────────────────────────────
C_SALES   = "#185FA5"
C_LIGHT   = "#B5D4F4"
C_GREEN   = "#1D9E75"
C_AMBER   = "#EF9F27"
C_CORAL   = "#D85A30"
C_RED     = "#A32D2D"
C_GRAY    = "#888780"
C_LGRAY   = "#F1EFE8"

MONTH_LABELS = {
    "2025-10": "Eki 25",
    "2025-11": "Kas 25",
    "2025-12": "Ara 25",
    "2026-01": "Oca 26",
}

plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.color":         "#ebebeb",
    "grid.linewidth":     0.6,
    "font.family":        "sans-serif",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "axes.titlepad":      8,
    "xtick.bottom":       False,
    "ytick.left":         False,
})


# ════════════════════════════════════════════════════════════════
# VERİ HAZIRLIK
# ════════════════════════════════════════════════════════════════

def is_mandatory_period(month: str, day: int = 15) -> bool:
    """Verilen ay ve gün zorunlu indirim döneminde mi?"""
    for m, start_day in MANDATORY_PERIODS:
        if month == m and day >= start_day:
            return True
        if month == m and start_day == 1:
            return True
    return False


def flag_mandatory(df: pd.DataFrame) -> pd.DataFrame:
    """Her işlemi zorunlu/normal dönem olarak etiketle."""
    # Ay ortası baz alınarak flag atanır (günlük veri yok)
    # Aralık için: ayın yarısında zorunlu dönem başladığından
    # Aralık işlemlerini 50/50 bölüyoruz (yaklaşık)
    def check(row):
        month = row["month"]
        if month == "2025-12":
            # Aralık'ı ikiye böl — elimizde gün bilgisi yok
            # Yaklaşım: işlemlerin ilk %35'i normal, kalanı zorunlu
            return "zorunlu"   # Konservatif: tamamını zorunlu say
        for m, start_day in MANDATORY_PERIODS:
            if month == m:
                return "zorunlu"
        return "normal"
    df["discount_period"] = df.apply(check, axis=1)
    return df


def compute_mandatory_discount(category: str) -> float:
    return MANDATORY_DISCOUNT_RATE.get(category, MANDATORY_DISCOUNT_RATE["DEFAULT"])


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(TRANSACTIONS_PATH)
    df["month_label"] = df["month"].map(MONTH_LABELS)
    df = flag_mandatory(df)

    # Premium flag (birim tam fiyat > 15,000 ₽)
    df["is_premium"] = (df["full_price_rub"] / df["quantity"] > 15000).astype(int)

    # Zorunlu dönemde beklenen indirim
    df["mandatory_disc"] = df["category"].apply(compute_mandatory_discount)

    # Fazla indirim = satıcının kendi verdiği ekstra (normal dönemde anlamlı)
    df["extra_discount"] = np.where(
        df["discount_period"] == "normal",
        np.maximum(0, df["discount_pct"] - 0),   # Normal dönemde tüm indirim satıcıya ait
        np.maximum(0, df["discount_pct"] - df["mandatory_disc"])  # Zorunlu üstü fazlası
    )
    return df


# ════════════════════════════════════════════════════════════════
# PRİM HESAPLAMA
# ════════════════════════════════════════════════════════════════

def compute_bonus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her satıcı için aylık prim hesaplar.
    Hedef: bir önceki ayın %10 üstü (basit rolling hedef)
    """
    # Satıcı × ay bazlı özetler
    mgr = df.groupby(["manager","month","month_label","discount_period"]).agg(
        ciro          = ("revenue_rub",  "sum"),
        adet          = ("quantity",     "sum"),
        avg_discount  = ("discount_pct", "mean"),
        extra_disc    = ("extra_discount","mean"),
        premium_adet  = ("is_premium",   "sum"),
        ticket_sum    = ("revenue_rub",  "sum"),
        ticket_count  = ("quantity",     "count"),
    ).reset_index()

    mgr["avg_ticket"]    = mgr["ticket_sum"] / mgr["ticket_count"]
    mgr["premium_ratio"] = mgr["premium_adet"] / mgr["adet"]

    # Hedef: bir önceki ay cirosunun %10 üstü
    mgr = mgr.sort_values(["manager","month"])
    mgr["ciro_target"] = mgr.groupby("manager")["ciro"].shift(1) * 1.10
    # İlk ay için hedef yok → o ayın ortalamasını baz al
    store_avg = mgr.groupby("month")["ciro"].transform("mean")
    mgr["ciro_target"] = mgr["ciro_target"].fillna(store_avg * 0.9)

    # Ortalama ticket hedefi (mağaza ortalaması)
    mgr["ticket_target"] = mgr.groupby("month")["avg_ticket"].transform("mean")

    # ── SKORLAR ──────────────────────────────────────────────────
    # 1. Ciro skoru (0–1.3)
    mgr["score_ciro"] = (mgr["ciro"] / mgr["ciro_target"]).clip(0, MAX_SCORE_MULTIPLIER)

    # 2. İndirim skoru (0–1): az ekstra indirim = yüksek skor
    #    Normal dönem: ekstra indirim %0 → skor 1.0, %20+ → skor 0
    #    Zorunlu dönem: skor hesaplanmaz (0 ağırlık)
    mgr["score_indirim"] = np.where(
        mgr["discount_period"] == "normal",
        (1 - mgr["extra_disc"] / 20).clip(0, 1),
        1.0   # Zorunlu dönemde nötr (ağırlık 0 olduğu için fark etmez)
    )

    # 3. Ticket skoru (0–1.3)
    mgr["score_ticket"] = (mgr["avg_ticket"] / mgr["ticket_target"]).clip(0, MAX_SCORE_MULTIPLIER)

    # 4. Premium skor (0–1)
    mgr["score_premium"] = mgr["premium_ratio"].clip(0, 1)

    # ── AĞIRLIKLI PERFORMANS SKORU ───────────────────────────────
    def weighted_score(row):
        w = W_MANDATORY if row["discount_period"] == "zorunlu" else W_NORMAL
        return (row["score_ciro"]    * w["ciro"]    +
                row["score_indirim"] * w["indirim"] +
                row["score_ticket"]  * w["ticket"]  +
                row["score_premium"] * w["premium"])

    mgr["perf_score"] = mgr.apply(weighted_score, axis=1)

    # ── PRİM ─────────────────────────────────────────────────────
    mgr["bonus_rub"] = mgr["ciro"] * BONUS_RATE * mgr["perf_score"]

    # Skor rengi (dashboard için)
    mgr["score_color"] = mgr["perf_score"].apply(
        lambda s: C_GREEN if s >= 0.9 else (C_AMBER if s >= 0.7 else C_RED)
    )

    return mgr


# ════════════════════════════════════════════════════════════════
# DASHBOARD GRAFİKLERİ
# ════════════════════════════════════════════════════════════════

def plot_dashboard(df: pd.DataFrame, bonus: pd.DataFrame):
    months     = list(MONTH_LABELS.values())
    managers   = bonus.groupby("manager")["ciro"].sum().sort_values(ascending=False).index.tolist()
    mgr_short  = [m.split()[0].capitalize() for m in managers]

    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(
        "Fashion Retail Store — Satış Danışmanı Performans & Prim Dashboardu",
        fontsize=13, fontweight="bold", y=0.995
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.65, wspace=0.4)

    # ── 1. Aylık ciro — satıcı bazlı yığılı bar ─────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    colors_mgr = ["#185FA5","#1D9E75","#EF9F27","#D85A30","#534AB7",
               "#3B6D11","#F4C0D1","#085041","#FAC775","#B5D4F4","#888780"]
    bottom = np.zeros(len(months))
    for i, (mgr, short) in enumerate(zip(managers, mgr_short)):
        vals = []
        for m_label in months:
            row = bonus[(bonus["manager"] == mgr) & (bonus["month_label"] == m_label)]
            vals.append(row["ciro"].values[0] / 1e6 if len(row) else 0)
        ax1.bar(months, vals, bottom=bottom, color=colors_mgr[i],
                width=0.5, label=short, zorder=2)
        bottom += np.array(vals)
    ax1.set_title("Aylık ciro — satıcı bazlı dağılım (₽M)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"₽{v:.1f}M"))
    ax1.legend(fontsize=7, frameon=False, loc="upper right",
               ncol=3, bbox_to_anchor=(1, 1.15))
    ax1.grid(axis="x", visible=False)

    # ── 2. Toplam prim — satıcı bazlı ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    total_bonus = bonus.groupby("manager")["bonus_rub"].sum().reindex(managers)
    colors_b = [C_GREEN if v > total_bonus.mean() else C_LIGHT for v in total_bonus.values]
    ax2.barh(mgr_short, total_bonus.values / 1e3, color=colors_b, height=0.6, zorder=2)
    ax2.set_title("Toplam prim — 4 ay toplamı\n(yeşil = mağaza ortalaması üstü)")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"₽{v:.0f}K"))
    ax2.grid(axis="y", visible=False)
    ax2.grid(axis="x", color="#ebebeb", linewidth=0.6)
    for i, v in enumerate(total_bonus.values):
        ax2.text(v/1e3 + 0.5, i, f"₽{v/1e3:.1f}K", va="center", fontsize=8, color=C_SALES)

    # ── 3. Performans skoru — aylık ısı haritası ─────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    score_matrix = np.zeros((len(managers), len(months)))
    for i, mgr in enumerate(managers):
        for j, m_label in enumerate(months):
            row = bonus[(bonus["manager"] == mgr) & (bonus["month_label"] == m_label)]
            score_matrix[i, j] = row["perf_score"].values[0] if len(row) else np.nan

    im = ax3.imshow(score_matrix, cmap="RdYlGn", vmin=0.5, vmax=1.2, aspect="auto")
    ax3.set_xticks(range(len(months))); ax3.set_xticklabels(months, fontsize=9)
    ax3.set_yticks(range(len(managers))); ax3.set_yticklabels(mgr_short, fontsize=8)
    ax3.grid(visible=False)
    for i in range(len(managers)):
        for j in range(len(months)):
            v = score_matrix[i, j]
            if not np.isnan(v):
                ax3.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=8, fontweight="bold",
                         color="white" if v < 0.75 or v > 1.1 else "black")
    plt.colorbar(im, ax=ax3, orientation="vertical", shrink=0.8, label="Performans skoru")
    ax3.set_title("Performans skoru ısı haritası (kırmızı=düşük, yeşil=yüksek)")

    # ── 4. İndirim skoru vs ciro skoru scatter ───────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    normal_bonus = bonus[bonus["discount_period"] == "normal"]
    scatter_colors = [C_GREEN if s >= 0.9 else (C_AMBER if s >= 0.7 else C_RED)
                      for s in normal_bonus["perf_score"]]
    ax4.scatter(normal_bonus["score_indirim"], normal_bonus["score_ciro"],
                c=scatter_colors, s=60, zorder=3, alpha=0.8)
    ax4.axhline(1.0, color=C_GRAY, linewidth=0.8, linestyle="--", alpha=0.5)
    ax4.axvline(1.0, color=C_GRAY, linewidth=0.8, linestyle="--", alpha=0.5)
    ax4.set_xlabel("İndirim kontrolü (sağ = az indirim yaptı)", fontsize=8)
    ax4.set_ylabel("Ciro skoru (yüksek=hedef aşıldı)", fontsize=8)
    ax4.set_title("Normal dönem:\nİndirim kontrolü vs ciro hedefi")
    ax4.grid(visible=True, color="#ebebeb")
    # Satıcı etiketleri
    for _, row in normal_bonus.iterrows():
        short = row["manager"].split()[0].capitalize()
        ax4.annotate(short, (row["score_indirim"], row["score_ciro"]),
                     fontsize=6, ha="left", va="bottom",
                     xytext=(3, 3), textcoords="offset points", color=C_GRAY)

    # ── 5. Aylık ortalama indirim — normal vs zorunlu dönem ──────
    ax5 = fig.add_subplot(gs[2, :2])
    x = np.arange(len(managers))
    normal_disc  = []
    mandatory_disc = []
    for mgr in managers:
        n = bonus[(bonus["manager"] == mgr) & (bonus["discount_period"] == "normal")]["avg_discount"].mean()
        m = bonus[(bonus["manager"] == mgr) & (bonus["discount_period"] == "zorunlu")]["avg_discount"].mean()
        normal_disc.append(n if not np.isnan(n) else 0)
        mandatory_disc.append(m if not np.isnan(m) else 0)
    ax5.bar(x - 0.2, normal_disc,    width=0.35, color=C_SALES, zorder=2, label="Normal dönem")
    ax5.bar(x + 0.2, mandatory_disc, width=0.35, color=C_LIGHT, zorder=2, label="Zorunlu dönem")
    ax5.set_xticks(x); ax5.set_xticklabels(mgr_short, fontsize=8)
    ax5.set_ylabel("Ort. indirim %", fontsize=8)
    ax5.set_title("Satıcı bazlı ortalama indirim — normal vs zorunlu dönem")
    ax5.legend(fontsize=8, frameon=False)
    ax5.grid(axis="x", visible=False)

    # ── 6. Premium satış oranı ───────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    prem_ratio = bonus.groupby("manager")["premium_ratio"].mean().reindex(managers)
    colors_p = [C_GREEN if v > prem_ratio.mean() else C_LIGHT for v in prem_ratio.values]
    ax6.barh(mgr_short, prem_ratio.values * 100, color=colors_p, height=0.6, zorder=2)
    ax6.axvline(prem_ratio.mean() * 100, color=C_CORAL, linewidth=1.2,
                linestyle="--", label=f"Ort. %{prem_ratio.mean()*100:.0f}")
    ax6.set_title("Premium ürün satış oranı %\n(yeşil=mağaza ort. üstü)")
    ax6.set_xlabel("%", fontsize=8)
    ax6.legend(fontsize=7, frameon=False)
    ax6.grid(axis="y", visible=False)
    ax6.grid(axis="x", color="#ebebeb", linewidth=0.6)

    # ── 7. Prim breakdown — son ay ───────────────────────────────
    ax7 = fig.add_subplot(gs[3, :])
    last_month = months[-1]
    last_bonus = bonus[bonus["month_label"] == last_month].copy()
    last_bonus = last_bonus.sort_values("bonus_rub", ascending=False)
    last_mgr_short = [m.split()[0].capitalize() for m in last_bonus["manager"]]
    x7 = np.arange(len(last_bonus))

    # Bileşen katkıları
    w = W_MANDATORY  # Son ay (Ocak) zorunlu dönem
    ciro_contrib    = last_bonus["score_ciro"]    * w["ciro"]    * last_bonus["ciro"] * BONUS_RATE / 1e3
    ticket_contrib  = last_bonus["score_ticket"]  * w["ticket"]  * last_bonus["ciro"] * BONUS_RATE / 1e3
    premium_contrib = last_bonus["score_premium"] * w["premium"] * last_bonus["ciro"] * BONUS_RATE / 1e3

    ax7.bar(x7, ciro_contrib.values,    color=C_SALES,  width=0.5, label="Ciro bileşeni",    zorder=2)
    ax7.bar(x7, ticket_contrib.values,  color=C_AMBER,  width=0.5, label="Ticket bileşeni",  zorder=2,
            bottom=ciro_contrib.values)
    ax7.bar(x7, premium_contrib.values, color=C_GREEN,  width=0.5, label="Premium bileşeni", zorder=2,
            bottom=ciro_contrib.values + ticket_contrib.values)

    # Toplam prim etiketi
    for i, (_, row) in enumerate(last_bonus.iterrows()):
        ax7.text(i, row["bonus_rub"]/1e3 + 0.3,
                 f"₽{row['bonus_rub']/1e3:.1f}K\n(skor:{row['perf_score']:.2f})",
                 ha="center", va="bottom", fontsize=7.5, color=C_SALES, fontweight="bold")

    ax7.set_xticks(x7); ax7.set_xticklabels(last_mgr_short, fontsize=9)
    ax7.set_title(f"Prim breakdown — {last_month} (zorunlu indirim dönemi, indirim skoru hesaba katılmadı)")
    ax7.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"₽{v:.0f}K"))
    ax7.legend(fontsize=8, frameon=False, loc="upper right")
    ax7.grid(axis="x", visible=False)

    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"\nDashboard kaydedildi: {SAVE_PATH}")


# ════════════════════════════════════════════════════════════════
# PRİM TABLOSU (konsol çıktısı)
# ════════════════════════════════════════════════════════════════

def print_bonus_table(bonus: pd.DataFrame):
    print("\n" + "=" * 80)
    print("  PRİM TABLOSU")
    print("=" * 80)
    cols = ["manager","month_label","discount_period","ciro","perf_score",
            "score_ciro","score_indirim","score_ticket","score_premium","bonus_rub"]
    tbl = bonus[cols].copy()
    tbl["ciro"]      = tbl["ciro"].apply(lambda v: f"₽{v:,.0f}")
    tbl["bonus_rub"] = tbl["bonus_rub"].apply(lambda v: f"₽{v:,.0f}")
    tbl["perf_score"]     = tbl["perf_score"].round(3)
    tbl["score_ciro"]     = tbl["score_ciro"].round(2)
    tbl["score_indirim"]  = tbl["score_indirim"].round(2)
    tbl["score_ticket"]   = tbl["score_ticket"].round(2)
    tbl["score_premium"]  = tbl["score_premium"].round(2)
    tbl.columns = ["Satıcı","Ay","Dönem","Ciro","Perf.Skor",
                   "Ciro S.","İndirim S.","Ticket S.","Premium S.","Prim"]
    print(tbl.to_string(index=False))

    print("\n" + "─" * 80)
    print("  TOPLAM PRİM ÖZETI")
    print("─" * 80)
    summary = (bonus.groupby("manager")
               .agg(toplam_ciro=("ciro","sum"),
                    ort_skor=("perf_score","mean"),
                    toplam_prim=("bonus_rub","sum"))
               .sort_values("toplam_prim", ascending=False))
    summary["toplam_ciro"] = summary["toplam_ciro"].apply(lambda v: f"₽{v:,.0f}")
    summary["toplam_prim"] = summary["toplam_prim"].apply(lambda v: f"₽{v:,.0f}")
    summary["ort_skor"]    = summary["ort_skor"].round(3)
    print(summary.to_string())


# ════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df    = load_and_prepare()
    bonus = compute_bonus(df)
    print_bonus_table(bonus)
    plot_dashboard(df, bonus)
