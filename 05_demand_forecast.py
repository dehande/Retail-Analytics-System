"""
====================================================================
Fashion Retail Store - Talep Tahmini Modeli
====================================================================
Modeller:
    1. Naive baseline    — geçen ayın değeri
    2. Linear regression — yorumlanabilir, az veriyle çalışır
    3. LightGBM          — dış etkenler eklenince güçlenir

Granularite:
    - Kategori bazlı toplam ciro
    - Kategori + cinsiyet bazlı ciro

Değerlendirme:
    Leave-one-out CV (4 aylık veriyle tek geçerli yöntem)

Çıktı:
    - Konsol: model performans tablosu
    - Grafik: tahmin vs gerçek + feature importance

Kullanım:
    python 05_demand_forecast.py

Gereksinimler:
    pip install pandas numpy matplotlib scikit-learn lightgbm
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠  LightGBM bulunamadı — pip install lightgbm. Sadece baseline + lineer çalışacak.")

# ─── PATH ────────────────────────────────────────────────────────
TRANSACTIONS_PATH = "D:/AI_Lab/02_ml/datasets/transactions_features.csv"
SAVE_PATH         = "D:/AI_Lab/02_ml/datasets/demand_forecast.png"
# ─────────────────────────────────────────────────────────────────

MONTH_LABELS = {
    "2025-10": "Eki 25",
    "2025-11": "Kas 25",
    "2025-12": "Ara 25",
    "2026-01": "Oca 26",
}
MONTH_ORDER = list(MONTH_LABELS.values())

C_SALES  = "#185FA5"
C_LIGHT  = "#B5D4F4"
C_AMBER  = "#EF9F27"
C_CORAL  = "#D85A30"
C_GREEN  = "#1D9E75"
C_RED    = "#A32D2D"
C_GRAY   = "#888780"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.color":        "#ebebeb",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.titlepad":     8,
    "xtick.bottom":      False,
    "ytick.left":        False,
})


# ════════════════════════════════════════════════════════════════
# VERİ HAZIRLIK
# ════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(TRANSACTIONS_PATH)
    df["month_label"] = df["month"].map(MONTH_LABELS)
    return df


def build_category_monthly(df):
    """Kategori bazlı aylık feature matrisi."""
    agg = (df.groupby(["month", "category"])
           .agg(
               total_rev      = ("revenue_rub",    "sum"),
               total_qty      = ("quantity",        "sum"),
               avg_discount   = ("discount_pct",    "mean"),
               avg_ticket     = ("avg_ticket_rub",  "mean"),
               winter_sens    = ("winter_sensitivity", "first"),
               holiday_imp    = ("holiday_importance", "first"),
               month_index    = ("month_index",     "first"),
               month_num      = ("month_num",       "first"),
               is_accessory   = ("is_accessory",    "first"),
           )
           .reset_index()
           .sort_values(["category", "month"]))

    # Lag features
    agg["rev_lag1"]    = agg.groupby("category")["total_rev"].shift(1)
    agg["rev_lag2"]    = agg.groupby("category")["total_rev"].shift(2)
    agg["qty_lag1"]    = agg.groupby("category")["total_qty"].shift(1)
    agg["rev_growth"]  = agg["total_rev"] / agg["rev_lag1"].replace(0, np.nan) - 1

    # Hedef: bir sonraki ay cirosu
    agg["target"] = agg.groupby("category")["total_rev"].shift(-1)

    return agg.dropna(subset=["rev_lag1"])


def build_gender_monthly(df):
    """Kategori + cinsiyet bazlı aylık feature matrisi."""
    agg = (df.groupby(["month", "category", "gender"])
           .agg(
               total_rev      = ("revenue_rub",    "sum"),
               total_qty      = ("quantity",        "sum"),
               avg_discount   = ("discount_pct",    "mean"),
               avg_ticket     = ("avg_ticket_rub",  "mean"),
               winter_sens    = ("winter_sensitivity", "first"),
               holiday_imp    = ("holiday_importance", "first"),
               month_index    = ("month_index",     "first"),
               month_num      = ("month_num",       "first"),
           )
           .reset_index()
           .sort_values(["category", "gender", "month"]))

    agg["rev_lag1"]   = agg.groupby(["category","gender"])["total_rev"].shift(1)
    agg["rev_lag2"]   = agg.groupby(["category","gender"])["total_rev"].shift(2)
    agg["qty_lag1"]   = agg.groupby(["category","gender"])["total_qty"].shift(1)
    agg["rev_growth"] = agg["total_rev"] / agg["rev_lag1"].replace(0, np.nan) - 1
    agg["target"]     = agg.groupby(["category","gender"])["total_rev"].shift(-1)
    agg["gender_enc"] = (agg["gender"] == "Women").astype(int)

    return agg.dropna(subset=["rev_lag1"])


FEATURES = [
    "rev_lag1", "rev_lag2", "qty_lag1", "rev_growth",
    "avg_discount", "avg_ticket", "winter_sens",
    "holiday_imp", "month_index", "month_num", "is_accessory",
]

FEATURES_GENDER = FEATURES + ["gender_enc"]


# ════════════════════════════════════════════════════════════════
# MODELLER
# ════════════════════════════════════════════════════════════════

def naive_baseline(data, group_col):
    """Geçen ayın değerini tahmin olarak kullan."""
    return data["rev_lag1"].values


def ridge_model(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = Ridge(alpha=10.0)
    model.fit(X_tr, y_train)
    return model.predict(X_te), model, scaler


def lgbm_model(X_train, y_train, X_test, feature_names):
    if not HAS_LGB:
        return None, None
    model = lgb.LGBMRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        min_child_samples=2,
        subsample=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        verbose=-1,
        random_state=42,
    )
    model.fit(X_train, y_train,
              feature_name=feature_names,
              callbacks=[lgb.early_stopping(10, verbose=False),
                         lgb.log_evaluation(-1)] if len(X_train) > 5 else [lgb.log_evaluation(-1)])
    return model.predict(X_test), model


# ════════════════════════════════════════════════════════════════
# LEAVE-ONE-OUT CV
# ════════════════════════════════════════════════════════════════

def leave_one_out_cv(data, features, label="kategori"):
    """
    4 aylık veriyle LOO-CV.
    Her seferinde 1 ayı test, kalanı train olarak kullan.
    Target NaN olan satırlar (son ay) test dışı bırakılır.
    """
    data = data.dropna(subset=["target"]).copy()
    months = sorted(data["month"].unique())

    results = []
    ridge_coefs = None
    lgbm_importances = None
    ridge_model_last = None
    scaler_last = None

    for test_month in months:
        train = data[data["month"] != test_month]
        test  = data[data["month"] == test_month]

        if len(train) < 3 or len(test) < 1:
            continue

        X_train = train[features].fillna(0).values
        y_train = train["target"].values
        X_test  = test[features].fillna(0).values
        y_test  = test["target"].values

        # Naive
        naive_pred = test["rev_lag1"].fillna(0).values

        # Ridge
        ridge_pred, ridge_m, scaler = ridge_model(X_train, y_train, X_test)
        ridge_model_last = ridge_m
        scaler_last = scaler

        # LightGBM
        if HAS_LGB and len(X_train) >= 5:
            lgbm_pred, lgbm_m = lgbm_model(X_train, y_train, X_test, features)
            if lgbm_m is not None:
                lgbm_importances = lgbm_m.feature_importances_
        else:
            lgbm_pred = None

        for i, (_, row) in enumerate(test.iterrows()):
            r = {
                "month":      test_month,
                "month_label": MONTH_LABELS.get(test_month, test_month),
                "actual":     y_test[i],
                "naive":      naive_pred[i],
                "ridge":      max(0, ridge_pred[i]),
            }
            if lgbm_pred is not None:
                r["lgbm"] = max(0, lgbm_pred[i])
            results.append(r)

    results_df = pd.DataFrame(results)

    # Ridge katsayıları
    if ridge_model_last is not None:
        ridge_coefs = pd.Series(
            ridge_model_last.coef_,
            index=features
        ).sort_values(key=abs, ascending=False)

    return results_df, ridge_coefs, lgbm_importances


# ════════════════════════════════════════════════════════════════
# PERFORMANS METRİKLERİ
# ════════════════════════════════════════════════════════════════

def compute_metrics(results_df):
    models = ["naive", "ridge"]
    if "lgbm" in results_df.columns:
        models.append("lgbm")

    metrics = []
    for m in models:
        valid = results_df.dropna(subset=[m, "actual"])
        mae  = mean_absolute_error(valid["actual"], valid[m])
        mape = mean_absolute_percentage_error(valid["actual"], valid[m]) * 100
        metrics.append({"model": m.upper(), "MAE (₽)": mae, "MAPE (%)": round(mape, 1)})

    return pd.DataFrame(metrics)


# ════════════════════════════════════════════════════════════════
# GRAFİKLER
# ════════════════════════════════════════════════════════════════

def plot_results(results_cat, results_gen, metrics_cat, metrics_gen,
                 ridge_coefs, lgbm_imp, features):

    fig = plt.figure(figsize=(17, 20))
    fig.suptitle("Fashion Retail Store — Talep Tahmini Modeli",
                 fontsize=13, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.6, wspace=0.38)

    # ── 1. Kategori bazlı: tahmin vs gerçek (top 6 kategori) ────
    ax1 = fig.add_subplot(gs[0, :])
    top6 = (results_cat.groupby("category")["actual"].sum()
            .sort_values(ascending=False).head(6).index.tolist()
            if "category" in results_cat.columns
            else results_cat.head(6).index.tolist())

    # Aylık toplam karşılaştırma
    monthly_actual = results_cat.groupby("month_label")["actual"].sum().reindex(MONTH_ORDER[:-1])
    monthly_naive  = results_cat.groupby("month_label")["naive"].sum().reindex(MONTH_ORDER[:-1])
    monthly_ridge  = results_cat.groupby("month_label")["ridge"].sum().reindex(MONTH_ORDER[:-1])

    x = np.arange(len(monthly_actual))
    w = 0.25
    ax1.bar(x - w, monthly_actual.values/1e6, width=w, color=C_SALES,  label="Gerçek",  zorder=2, borderRadius=0)
    ax1.bar(x,     monthly_naive.values/1e6,  width=w, color=C_GRAY,   label="Naive",   zorder=2)
    ax1.bar(x + w, monthly_ridge.values/1e6,  width=w, color=C_AMBER,  label="Ridge",   zorder=2)
    if "lgbm" in results_cat.columns:
        monthly_lgbm = results_cat.groupby("month_label")["lgbm"].sum().reindex(MONTH_ORDER[:-1])
        ax1.bar(x + w*2, monthly_lgbm.values/1e6, width=w, color=C_GREEN, label="LightGBM", zorder=2)

    ax1.set_xticks(x); ax1.set_xticklabels(monthly_actual.index)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"₽{v:.1f}M"))
    ax1.set_title("Aylık toplam ciro — gerçek vs tahmin (kategori bazlı)")
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(axis="x", visible=False)

    # ── 2. Cinsiyet bazlı: tahmin vs gerçek ─────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    monthly_actual_g = results_gen.groupby("month_label")["actual"].sum().reindex(MONTH_ORDER[:-1])
    monthly_ridge_g  = results_gen.groupby("month_label")["ridge"].sum().reindex(MONTH_ORDER[:-1])

    ax2.plot(MONTH_ORDER[:-1], monthly_actual_g.values/1e6,
             color=C_SALES, marker="o", linewidth=2, markersize=6, label="Gerçek")
    ax2.plot(MONTH_ORDER[:-1], monthly_ridge_g.values/1e6,
             color=C_AMBER, marker="s", linewidth=2, markersize=5,
             linestyle="--", label="Ridge tahmini")
    if "lgbm" in results_gen.columns:
        monthly_lgbm_g = results_gen.groupby("month_label")["lgbm"].sum().reindex(MONTH_ORDER[:-1])
        ax2.plot(MONTH_ORDER[:-1], monthly_lgbm_g.values/1e6,
                 color=C_GREEN, marker="^", linewidth=2, markersize=5,
                 linestyle="--", label="LightGBM tahmini")

    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"₽{v:.1f}M"))
    ax2.set_title("Aylık toplam ciro — gerçek vs tahmin (kategori + cinsiyet bazlı)")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(axis="x", visible=False)

    # ── 3. Model performans karşılaştırması — kategori ──────────
    ax3 = fig.add_subplot(gs[2, 0])
    models = metrics_cat["model"].tolist()
    mapes  = metrics_cat["MAPE (%)"].tolist()
    colors = [C_GRAY if m == "NAIVE" else (C_AMBER if m == "RIDGE" else C_GREEN) for m in models]
    bars = ax3.bar(models, mapes, color=colors, width=0.4, zorder=2)
    for bar, val in zip(bars, mapes):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"%{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax3.set_title("MAPE % — kategori bazlı\n(düşük = iyi)")
    ax3.set_ylabel("MAPE %", fontsize=8)
    ax3.grid(axis="x", visible=False)

    # ── 4. Model performans karşılaştırması — cinsiyet ──────────
    ax4 = fig.add_subplot(gs[2, 1])
    models_g = metrics_gen["model"].tolist()
    mapes_g  = metrics_gen["MAPE (%)"].tolist()
    colors_g = [C_GRAY if m == "NAIVE" else (C_AMBER if m == "RIDGE" else C_GREEN) for m in models_g]
    bars_g = ax4.bar(models_g, mapes_g, color=colors_g, width=0.4, zorder=2)
    for bar, val in zip(bars_g, mapes_g):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"%{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax4.set_title("MAPE % — kategori + cinsiyet bazlı\n(düşük = iyi)")
    ax4.set_ylabel("MAPE %", fontsize=8)
    ax4.grid(axis="x", visible=False)

    # ── 5. Ridge feature katsayıları ────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    if ridge_coefs is not None:
        top_coefs = ridge_coefs.head(8)
        colors_c = [C_GREEN if v > 0 else C_CORAL for v in top_coefs.values]
        ax5.barh(top_coefs.index, top_coefs.values, color=colors_c, height=0.5, zorder=2)
        ax5.axvline(0, color=C_GRAY, linewidth=0.8)
        ax5.set_title("Ridge — feature etkileri\n(yeşil=pozitif, mercan=negatif)")
        ax5.grid(axis="y", visible=False)
        ax5.grid(axis="x", color="#ebebeb", linewidth=0.6)

    # ── 6. LightGBM feature importance ──────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    if lgbm_imp is not None and HAS_LGB:
        imp_series = pd.Series(lgbm_imp, index=features).sort_values(ascending=True).tail(8)
        ax6.barh(imp_series.index, imp_series.values, color=C_SALES, height=0.5, zorder=2)
        ax6.set_title("LightGBM — feature importance\n(yüksek = daha etkili)")
        ax6.grid(axis="y", visible=False)
        ax6.grid(axis="x", color="#ebebeb", linewidth=0.6)
    elif not HAS_LGB:
        ax6.text(0.5, 0.5, "LightGBM kurulu değil\npip install lightgbm",
                 ha="center", va="center", transform=ax6.transAxes,
                 color=C_GRAY, fontsize=10)
        ax6.set_title("LightGBM — feature importance")

    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"\nGrafik kaydedildi: {SAVE_PATH}")


# ════════════════════════════════════════════════════════════════
# KONSOL ÇIKTISI
# ════════════════════════════════════════════════════════════════

def print_results(metrics_cat, metrics_gen, results_cat):
    print("\n" + "=" * 60)
    print("  MODEL PERFORMANS TABLOSU")
    print("=" * 60)

    print("\n📊 Kategori bazlı:")
    print(metrics_cat.to_string(index=False))

    print("\n📊 Kategori + cinsiyet bazlı:")
    print(metrics_gen.to_string(index=False))

    print("\n" + "─" * 60)
    print("  KATEGORİ BAZLI TAHMİN DETAYI (son CV adımı)")
    print("─" * 60)
    last_month = results_cat["month"].max()
    last = results_cat[results_cat["month"] == last_month].copy()
    last = last.sort_values("actual", ascending=False)
    cols = ["category","actual","naive","ridge"]
    if "lgbm" in last.columns:
        cols.append("lgbm")
    for col in ["actual","naive","ridge","lgbm"]:
        if col in last.columns:
            last[col] = last[col].apply(lambda v: f"₽{v:,.0f}")
    last.columns = [c.upper() for c in last.columns]
    print(last[["CATEGORY","ACTUAL","NAIVE","RIDGE"] +
               (["LGBM"] if "LGBM" in last.columns else [])].to_string(index=False))


# ════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Fashion Retail Store — Talep Tahmini Modeli")
    print("=" * 60 + "\n")

    df = load_data()

    print("▶ Kategori bazlı veri hazırlanıyor...")
    cat_data = build_category_monthly(df)
    print(f"   {len(cat_data)} satır, {cat_data['category'].nunique()} kategori")

    print("▶ Kategori + cinsiyet bazlı veri hazırlanıyor...")
    gen_data = build_gender_monthly(df)
    print(f"   {len(gen_data)} satır")

    print("▶ Leave-one-out CV çalışıyor — kategori bazlı...")
    results_cat, ridge_coefs, lgbm_imp = leave_one_out_cv(
        cat_data, FEATURES, label="kategori"
    )

    print("▶ Leave-one-out CV çalışıyor — cinsiyet bazlı...")
    results_gen, _, _ = leave_one_out_cv(
        gen_data, FEATURES_GENDER, label="cinsiyet"
    )

    metrics_cat = compute_metrics(results_cat)
    metrics_gen = compute_metrics(results_gen)

    print_results(metrics_cat, metrics_gen, results_cat)
    plot_results(results_cat, results_gen, metrics_cat, metrics_gen,
                 ridge_coefs, lgbm_imp, FEATURES)
