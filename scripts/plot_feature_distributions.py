"""
Generate individual per-feature distribution plots for the VBF event classifier project.
C1 (SM, cvv=1) vs C_not1 (everything else).
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = "ml_framework/data/large/matched_events_all_no_dup.csv"
OUT_DIR   = "docs/figures/dist"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Features to plot ───────────────────────────────────────────────────────────
FEATURES = [
    "vbf_q1_pt", "vbf_q1_eta", "vbf_q1_jet_pt", "vbf_q1_jet_eta", "vbf_q1_jet_m",
    "vbf_q2_pt", "vbf_q2_eta", "vbf_q2_jet_pt", "vbf_q2_jet_eta", "vbf_q2_jet_m",
    "w_qq_pt", "w_qq_eta", "w_qq_e", "w_qq_jet_pt", "w_qq_jet_eta", "w_qq_jet_e", "w_qq_jet_m",
    "w_qq_q1_pt", "w_qq_q1_e", "w_qq_q1_m", "w_qq_q2_pt", "w_qq_q2_e", "w_qq_q2_m",
    "w_lep_l_pt", "w_lep_l_eta", "w_lep_l_e", "w_lep_l_m",
    "w_lep_v_pt", "w_lep_v_eta", "w_lep_v_e", "w_lep_v_m",
    "h_bb_pt", "h_bb_e", "h_bb_jet_pt", "h_bb_jet_e", "h_bb_jet_m",
    "h_bb_b1_pt", "h_bb_b1_e", "h_bb_b2_pt", "h_bb_b2_e",
    "hbb_wqq_m", "hbb_wqq_pt", "hbb_lep_m", "hbb_lep_pt", "hbb_lep_eta",
    "wqq_lep_m", "wqq_lep_pt", "wqq_lep_eta", "q1_q2_m", "q1_q2_pt", "q1_q2_eta",
]

# ── Style ──────────────────────────────────────────────────────────────────────
C1_COLOR    = "#e34234"
CNOT1_COLOR = "#2196F3"
ALPHA       = 0.45
LW          = 1.5

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"  Total rows: {len(df):,}")

# Sample if large
MAX_ROWS = 200_000
if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42)
    print(f"  Sampled down to {MAX_ROWS:,} rows")

# ── Split classes ──────────────────────────────────────────────────────────────
mask_c1   = df["cvv"] == 1.0
df_c1     = df[mask_c1]
df_cnot1  = df[~mask_c1]
print(f"  C1 (cvv=1): {len(df_c1):,}   C_not1: {len(df_cnot1):,}")

# ── Helper: KDE density plot ───────────────────────────────────────────────────
def plot_kde(ax, data, color, label, alpha=ALPHA, lw=LW):
    data = data.dropna()
    if len(data) < 2:
        return
    # Clip extreme outliers for nicer plots (1st–99th percentile range extended by 5%)
    lo, hi = data.quantile(0.001), data.quantile(0.999)
    span = hi - lo
    if span == 0:
        span = 1.0
    lo_ext = lo - 0.05 * span
    hi_ext = hi + 0.05 * span
    data_clipped = data.clip(lo_ext, hi_ext)
    xs = np.linspace(lo_ext, hi_ext, 500)
    try:
        kde = gaussian_kde(data_clipped, bw_method="scott")
        ys  = kde(xs)
    except Exception:
        return
    ax.fill_between(xs, ys, alpha=alpha, color=color, label=label)
    ax.plot(xs, ys, color=color, linewidth=lw)

# ── Generate plots ─────────────────────────────────────────────────────────────
generated = []
skipped   = []

for feat in FEATURES:
    if feat not in df.columns:
        print(f"  [SKIP] Column not found: {feat}")
        skipped.append(feat)
        continue

    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plot_kde(ax, df_c1[feat],   color=C1_COLOR,    label=r"$C_1$ (SM, cvv=1)")
    plot_kde(ax, df_cnot1[feat], color=CNOT1_COLOR, label=r"$C_{\mathrm{not}1}$")

    ax.set_xlabel(feat, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(feat, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{feat}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    generated.append(out_path)
    print(f"  Saved: {out_path}")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\nDone. Generated {len(generated)} plots, skipped {len(skipped)}.")
if skipped:
    print(f"  Skipped features: {skipped}")
