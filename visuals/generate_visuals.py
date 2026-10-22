"""
generate_visuals.py — Pattupetti Graph Suite
=============================================
Generates all EDA + evaluation charts styled to match the
Pattupetti colour palette:

  Background  : #e2e2e0  (concrete grey)
  Surface     : #ededeb
  Alt BG      : #d4d5d8
  Text Primary: #161719  (near-black)
  Text Muted  : #8a8e96
  Accent      : #e52421  (signal vermilion red)
  Dark chip   : #18191b  (matte black hardware)

Charts produced
---------------
1. class_balance.png         — Genre class distribution bar chart
2. feature_boxplot.png       — Feature distributions per genre (key features)
3. feature_correlation.png   — Feature correlation heat-map
4. model_comparison.png      — Accuracy vs Macro-F1 per model
5. confusion_matrix.png      — Normalised confusion matrix (best model)
6. pre_genre_radar.png       — Pre-genre insights radar (per-genre mean features)

Run from the project root:
    python visuals/generate_visuals.py
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -- Project root & output dir -----------------------------------------------
ROOT   = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUTPUT = Path(__file__).resolve().parent   # saves inside visuals/
OUTPUT.mkdir(parents=True, exist_ok=True)

# -- Pattupetti Palette -------------------------------------------------------
BG         = "#e2e2e0"
BG_ALT     = "#d4d5d8"
SURFACE    = "#ededeb"
BORDER     = "#c4c7cc"
TEXT_PRI   = "#161719"
TEXT_SEC   = "#585b62"
TEXT_MUTED = "#8a8e96"
ACCENT     = "#e52421"
DARK_CHIP  = "#18191b"

# 8-genre palette: vermilion anchor + harmonious greys/charcoals
GENRE_COLORS = [
    "#e52421",  # Electronic   — signal red
    "#18191b",  # Experimental — matte black
    "#585b62",  # Folk         — medium grey
    "#8a8e96",  # Hip-Hop      — stone grey
    "#c4c7cc",  # Instrumental — light border grey
    "#b04040",  # International— deep rose-red
    "#d4d5d8",  # Pop          — pale grey
    "#a0a4ab",  # Rock         — mid grey
]


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    SURFACE,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT_PRI,
        "axes.titlecolor":   TEXT_PRI,
        "axes.titleweight":  "bold",
        "axes.titlesize":    13,
        "axes.labelsize":    10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        BORDER,
        "grid.linewidth":    0.6,
        "grid.alpha":        0.55,
        "xtick.color":       TEXT_SEC,
        "ytick.color":       TEXT_SEC,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "text.color":        TEXT_PRI,
        "legend.framealpha": 0.92,
        "legend.edgecolor":  BORDER,
        "legend.facecolor":  SURFACE,
        "legend.fontsize":   9,
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DM Sans", "Inter", "Segoe UI", "Arial"],
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.25,
    })


apply_style()


def footer(fig, text):
    fig.text(0.98, 0.015, text, ha="right", va="bottom",
             fontsize=7, color=TEXT_MUTED, style="italic")


def save(fig, name):
    path = OUTPUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  OK  Saved -> {path.relative_to(ROOT)}")


# ============================================================================
# 1.  CLASS BALANCE
# ============================================================================
def plot_class_balance(df):
    counts = df["genre"].value_counts().sort_values(ascending=False)
    genres = counts.index.tolist()
    colors = GENRE_COLORS[:len(genres)]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.14)

    bars = ax.bar(genres, counts.values, color=colors, width=0.62,
                  edgecolor=DARK_CHIP, linewidth=0.6, zorder=3)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 14,
                f"{val:,}", ha="center", va="bottom",
                fontsize=9, fontweight="600", color=TEXT_PRI)

    # highlight the tallest bar
    bars[counts.values.argmax()].set_edgecolor(ACCENT)
    bars[counts.values.argmax()].set_linewidth(2.0)

    mean_val = counts.mean()
    ax.axhline(mean_val, color=ACCENT, linewidth=1.2,
               linestyle="--", alpha=0.7, zorder=2)
    ax.text(len(genres) - 0.5, mean_val + 12,
            f"avg {int(mean_val):,}", color=ACCENT, fontsize=8, ha="right")

    ax.set_title("Class Balance — Genre Distribution", pad=14)
    ax.set_xlabel("Music Genre", labelpad=8)
    ax.set_ylabel("Track Count", labelpad=8)
    ax.set_ylim(0, counts.max() * 1.14)
    ax.tick_params(axis="x", rotation=25)

    footer(fig, "pattupetti · EDA · class_balance")
    save(fig, "class_balance.png")


# ============================================================================
# 2.  FEATURE BOX-PLOT
# ============================================================================
def plot_feature_boxplot(df):
    key_features = [
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "zcr_mean",
        "rms_mean",
        "tempo",
        "mfcc_1_mean",
    ]
    feat_labels = [
        "Spectral Centroid",
        "Spectral Bandwidth",
        "Zero-Crossing Rate",
        "RMS Energy",
        "Tempo (BPM)",
        "MFCC-1",
    ]
    genres = sorted(df["genre"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Feature Distributions per Genre", fontsize=14,
                 fontweight="bold", color=TEXT_PRI, y=1.01)
    axes = axes.flatten()

    for idx, (feat, label) in enumerate(zip(key_features, feat_labels)):
        ax = axes[idx]
        data = [df.loc[df["genre"] == g, feat].dropna().values for g in genres]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            notch=False,
            widths=0.55,
            medianprops=dict(color=ACCENT, linewidth=2),
            whiskerprops=dict(color=TEXT_MUTED, linewidth=0.9),
            capprops=dict(color=TEXT_MUTED, linewidth=0.9),
            flierprops=dict(marker="o", color=ACCENT, alpha=0.25,
                            markersize=3, linestyle="none"),
        )
        for patch, color in zip(bp["boxes"], GENRE_COLORS[:len(genres)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)
            patch.set_edgecolor(DARK_CHIP)
            patch.set_linewidth(0.7)

        ax.set_title(label, fontsize=10, fontweight="600", color=TEXT_PRI, pad=6)
        ax.set_xticks(range(1, len(genres) + 1))
        ax.set_xticklabels([g[:4] for g in genres],
                           fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Value", fontsize=8, labelpad=4)

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    footer(fig, "pattupetti · EDA · feature_boxplot")
    save(fig, "feature_boxplot.png")


# ============================================================================
# 3.  FEATURE CORRELATION HEAT-MAP
# ============================================================================
def plot_feature_correlation(df):
    drop_cols = ["track_id", "genre"]
    num_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    mean_cols = [c for c in num_df.columns
                 if c.endswith("_mean") or c == "tempo"]
    corr = num_df[mean_cols].corr()

    short = {
        c: c.replace("_mean", "")
             .replace("spectral_", "s.")
             .replace("chroma_", "chr.")
             .replace("mfcc_", "mfcc")
        for c in mean_cols
    }
    corr.index   = [short[c] for c in corr.index]
    corr.columns = [short[c] for c in corr.columns]

    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = sns.diverging_palette(h_neg=5, h_pos=230, s=70, l=50, as_cmap=True)

    sns.heatmap(
        corr, ax=ax, cmap=cmap, center=0, vmin=-1, vmax=1,
        linewidths=0.3, linecolor=BG_ALT, annot=False,
        cbar_kws={"shrink": 0.75, "label": "Pearson r", "pad": 0.02},
    )
    ax.set_title("Feature Correlation Matrix (Mean Features)", pad=14)
    ax.tick_params(axis="x", rotation=55, labelsize=7.5)
    ax.tick_params(axis="y", rotation=0,  labelsize=7.5)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(TEXT_SEC)
    cbar.ax.tick_params(labelcolor=TEXT_SEC, labelsize=8)

    fig.tight_layout()
    footer(fig, "pattupetti · EDA · feature_correlation")
    save(fig, "feature_correlation.png")


# ============================================================================
# 4.  MODEL COMPARISON BAR CHART
# ============================================================================
def plot_model_comparison():
    comp_csv = ROOT / "models" / "model_comparison.csv"
    if not comp_csv.exists():
        print("  WARN  models/model_comparison.csv not found — skipping.")
        return

    df = pd.read_csv(comp_csv).sort_values("Test Macro F1", ascending=False)
    x, width = np.arange(len(df)), 0.34

    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.18)

    bars_acc = ax.bar(x - width / 2, df["Test Accuracy"], width,
                      color=DARK_CHIP, edgecolor=DARK_CHIP, linewidth=0.5,
                      label="Test Accuracy", zorder=3, alpha=0.88)
    bars_f1  = ax.bar(x + width / 2, df["Test Macro F1"], width,
                      color=ACCENT, edgecolor=DARK_CHIP, linewidth=0.5,
                      label="Test Macro F1", zorder=3, alpha=0.88)

    for bars in (bars_acc, bars_f1):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.008, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="600", color=TEXT_PRI)

    best_i = df["Test Macro F1"].values.argmax()
    ax.annotate("* best",
                xy=(best_i + width / 2,
                    df["Test Macro F1"].iloc[best_i] + 0.03),
                ha="center", fontsize=8, color=ACCENT, fontweight="700")

    ax.set_title("Model Performance Comparison — FMA Small (8 Genres)", pad=14)
    ax.set_xlabel("Machine Learning Model", labelpad=8)
    ax.set_ylabel("Score (0–1)", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=28, ha="right", fontsize=9)
    ax.set_ylim(0, 0.78)
    ax.legend(loc="upper right")

    footer(fig, "pattupetti · evaluation · model_comparison")
    save(fig, "model_comparison.png")


# ============================================================================
# 5.  CONFUSION MATRIX  (normalised)
# ============================================================================
def plot_confusion_matrix_chart(df):
    models_dir  = ROOT / "models"
    model_path  = models_dir / "best_model.joblib"
    scaler_path = models_dir / "scaler.joblib"
    feat_path   = models_dir / "feature_names.joblib"
    name_path   = models_dir / "best_model_name.joblib"

    if not (model_path.exists() and scaler_path.exists() and feat_path.exists()):
        print("  WARN  Trained model artefacts not found — skipping confusion matrix.")
        return

    model         = joblib.load(model_path)
    scaler        = joblib.load(scaler_path)
    feature_names = joblib.load(feat_path)
    model_name    = joblib.load(name_path) if name_path.exists() else "Best Model"

    drop_cols = ["track_id", "genre", "file_path"]
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = df["genre"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_scaled = scaler.transform(X_test[feature_names])
    y_pred = model.predict(X_test_scaled)

    labels  = sorted(y.unique())
    cm      = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "patti_red",
        [BG, BG_ALT, "#e0b0b0", "#c06060", ACCENT, "#9e1a18"],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for r in range(len(labels)):
        for c in range(len(labels)):
            val   = cm_norm[r, c]
            color = "white" if val > 0.55 else TEXT_PRI
            ax.text(c, r, f"{val:.0%}", ha="center", va="center",
                    fontsize=8.5, fontweight="600", color=color)

    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("Normalised Recall", color=TEXT_SEC, fontsize=9)
    cb.ax.tick_params(labelcolor=TEXT_SEC, labelsize=8)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted Genre", labelpad=10)
    ax.set_ylabel("True Genre", labelpad=10)
    ax.set_title(f"Confusion Matrix — {model_name} (Normalised)", pad=14)

    for i in range(len(labels) + 1):
        ax.axhline(i - 0.5, color=BG, linewidth=0.8)
        ax.axvline(i - 0.5, color=BG, linewidth=0.8)

    fig.tight_layout()
    footer(fig, "pattupetti · evaluation · confusion_matrix")
    save(fig, "confusion_matrix.png")


# ============================================================================
# 6.  PRE-GENRE INSIGHTS RADAR
# ============================================================================
def plot_pre_genre_radar(df):
    key_features = [
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "zcr_mean",
        "rms_mean",
        "tempo",
        "mfcc_1_mean",
    ]
    feat_labels = [
        "Spec.\nCentroid",
        "Spec.\nBandwidth",
        "ZCR",
        "RMS",
        "Tempo",
        "MFCC-1",
    ]
    genres  = sorted(df["genre"].unique())
    n_feats = len(key_features)

    genre_means = df.groupby("genre")[key_features].mean()
    normed = (genre_means - genre_means.min()) / (
        genre_means.max() - genre_means.min() + 1e-9
    )

    angles  = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(13, 10))
    fig.patch.set_facecolor(BG)

    for g_idx, genre in enumerate(genres):
        ax = fig.add_subplot(2, 4, g_idx + 1, polar=True)
        ax.set_facecolor(SURFACE)

        values = normed.loc[genre].tolist()
        values += values[:1]

        color = GENRE_COLORS[g_idx % len(GENRE_COLORS)]
        ax.plot(angles, values, color=color, linewidth=2, zorder=3)
        ax.fill(angles, values, alpha=0.25, color=color, zorder=2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feat_labels, fontsize=7, color=TEXT_SEC)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["0.25", "0.5", "0.75"], fontsize=5.5, color=TEXT_MUTED)
        ax.set_ylim(0, 1)
        ax.spines["polar"].set_color(BORDER)
        ax.grid(color=BORDER, linewidth=0.5, alpha=0.6)
        ax.set_title(genre, fontsize=9.5, fontweight="700",
                     color=TEXT_PRI, pad=10)

    fig.suptitle("Pre-Genre Insights — Normalised Feature Radar",
                 fontsize=14, fontweight="bold", color=TEXT_PRI, y=1.01)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    footer(fig, "pattupetti · EDA · pre_genre_radar")
    save(fig, "pre_genre_radar.png")


# ============================================================================
# MAIN
# ============================================================================
def main():
    csv_path = ROOT / "data" / "processed" / "features_30s.csv"
    if not csv_path.exists():
        print(f"[Error] Feature CSV not found at {csv_path}")
        sys.exit(1)

    print("\n" + "=" * 58)
    print("  Pattupetti Visual Suite — generating graphs ...")
    print("  Output folder : visuals/")
    print("=" * 58 + "\n")

    df = pd.read_csv(csv_path)

    print("[1/6] Class Balance ...")
    plot_class_balance(df)

    print("[2/6] Feature Box-plot ...")
    plot_feature_boxplot(df)

    print("[3/6] Feature Correlation ...")
    plot_feature_correlation(df)

    print("[4/6] Model Comparison ...")
    plot_model_comparison()

    print("[5/6] Confusion Matrix ...")
    plot_confusion_matrix_chart(df)

    print("[6/6] Pre-Genre Radar ...")
    plot_pre_genre_radar(df)

    print("\n" + "=" * 58)
    print("  All charts saved to visuals/")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    main()
