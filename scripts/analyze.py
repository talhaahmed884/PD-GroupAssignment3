#!/usr/bin/env python3


import sys
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("NOTE: seaborn not installed — efficiency heatmap will be skipped.")
    print("      Install with: pip install seaborn")

# ============================================================
# CONFIG
# ============================================================
PLOT_DIR = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

ALGO_COLORS = {
    "ser": "#555555",
    "1d": "#ff7f0e",
    "2d": "#2ca02c",
}
ALGO_LABELS = {
    "ser": "MM-ser (serial)",
    "1d": "MM-1D (MPI row-strip)",
    "2d": "MM-2D (MPI 2D grid)",
}
PARALLEL_ALGOS = ["1d", "2d"]


# ============================================================
# DATA LOADING & CLEANING
# ============================================================
def load_csv(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Drop error rows
    df = df[df["time_seconds"] != "ERROR"].copy()
    df["time_seconds"] = pd.to_numeric(df["time_seconds"], errors="coerce")
    df.dropna(subset=["time_seconds"], inplace=True)
    df = df[df["time_seconds"] > 0].copy()

    print(f"Loaded {len(df)} valid timing rows from {len(paths)} file(s).")

    # Aggregate repetitions: median + variance stats per config
    agg = (
        df.groupby(["algorithm", "m", "n", "q", "P"])["time_seconds"]
        .agg(time_median="median", time_min="min", time_max="max", time_std="std")
        .reset_index()
    )
    return df, agg


# ============================================================
# SPEEDUP, COST, EFFICIENCY
# ============================================================
def add_metrics(agg):
    # T_serial = time of algo="ser" at P=1 for each (m, n, q)
    serial = agg[(agg["algorithm"] == "ser") & (agg["P"] == 1)][
        ["m", "n", "q", "time_median"]
    ].rename(columns={"time_median": "T_serial"})

    result = agg.merge(serial, on=["m", "n", "q"], how="left")
    result["speedup"] = result["T_serial"] / result["time_median"]
    result["cost"] = result["P"] * result["time_median"]
    result["efficiency"] = result["speedup"] / result["P"]
    return result


def square_sizes(df):
    sq = df[(df["m"] == df["n"]) & (df["n"] == df["q"])]
    return sorted(sq["m"].unique())


# ============================================================
# PLOT 1: Speedup vs P
# Assignment Q2: "How does P affect speedup?"
# ============================================================
def plot_speedup_vs_P(df):
    print("\n[Plot 1] Speedup vs P (answers: How does P affect speedup?)")
    for sz in square_sizes(df):
        fig, ax = plt.subplots(figsize=(7, 5))
        subset = df[(df["m"] == sz) & (df["n"] == sz) & (df["q"] == sz)]

        P_all = sorted(df["P"].unique())
        ax.plot(P_all, P_all, "k--", lw=1, label="Ideal (linear speedup)", zorder=1)

        for algo in PARALLEL_ALGOS:
            s = subset[subset["algorithm"] == algo].sort_values("P")
            if s.empty:
                continue
            ax.plot(
                s["P"],
                s["speedup"],
                marker="o",
                color=ALGO_COLORS.get(algo, "gray"),
                label=ALGO_LABELS.get(algo, algo),
                zorder=2,
            )
            # Shaded variance band
            speedup_lo = s["T_serial"] / s["time_max"]
            speedup_hi = s["T_serial"] / s["time_min"]
            ax.fill_between(
                s["P"],
                speedup_lo,
                speedup_hi,
                color=ALGO_COLORS.get(algo, "gray"),
                alpha=0.12,
            )

        ax.set_xlabel("Number of Processes P")
        ax.set_ylabel("Speedup  (T_serial / T_par)")
        ax.set_title(
            f"Speedup vs P  |  m=n=q={sz}\n" f"(Q2: How does P affect speedup?)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        fname = os.path.join(PLOT_DIR, f"speedup_vs_P_size{sz}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ============================================================
# PLOT 2: Runtime vs Matrix Size (log-log)
# Assignment Q2: "How does matrix size affect runtime?"
# ============================================================
def plot_runtime_vs_size(df):
    print("\n[Plot 2] Runtime vs Size (answers: How does matrix size affect runtime?)")
    P_values = sorted([p for p in df["P"].unique() if p in [1, 4, 16, 36]])

    for P in P_values:
        fig, ax = plt.subplots(figsize=(7, 5))
        sq = df[(df["m"] == df["n"]) & (df["n"] == df["q"])]

        for algo in ["ser", "1d", "2d"]:
            if algo == "ser":
                s = sq[sq["algorithm"] == "ser"].sort_values("m")
            else:
                s = sq[(sq["algorithm"] == algo) & (sq["P"] == P)].sort_values("m")
            if s.empty:
                continue
            label = ALGO_LABELS.get(algo, algo)
            if algo != "ser":
                label += f" (P={P})"
            ax.plot(
                s["m"],
                s["time_median"],
                marker="o",
                color=ALGO_COLORS.get(algo, "gray"),
                label=label,
            )

        ax.set_xlabel("Matrix Dimension  (m = n = q)")
        ax.set_ylabel("Wall-clock Time (seconds)")
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.set_title(
            f"Runtime vs Size  |  P={P}\n" f"(Q2: How does matrix size affect runtime?)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

        fname = os.path.join(PLOT_DIR, f"runtime_vs_size_P{P}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ============================================================
# PLOT 3: Shape Comparison (bar chart)
# Assignment Q2: "Do certain matrix shapes work better for certain algorithms?"
# ============================================================
def plot_shape_comparison(df):
    print("\n[Plot 3] Shape Comparison (answers: Do certain shapes work better?)")

    # Compare three shapes at the same n=512 baseline, fixed P=16 (or nearest available)
    shapes = {
        "square\n(512×512×512)": (512, 512, 512),
        "thin-tall A\n(512×128×512)": (512, 128, 512),
        "wide A\n(128×512×128)": (128, 512, 128),
    }
    P_TARGET = 16

    # Find closest available P
    available_P = sorted(df["P"].unique())
    P_use = min(available_P, key=lambda p: abs(p - P_TARGET))

    fig, axes = plt.subplots(1, len(PARALLEL_ALGOS), figsize=(14, 5), sharey=False)
    fig.suptitle(
        f"Shape Sensitivity  |  P={P_use}\n"
        f"(Q2: Do certain matrix shapes work better for certain algorithms?)",
        fontsize=11,
    )

    for ax, algo in zip(axes, PARALLEL_ALGOS):
        times = []
        labels = []
        for shape_label, (m, n, q) in shapes.items():
            row = df[
                (df["algorithm"] == algo)
                & (df["m"] == m)
                & (df["n"] == n)
                & (df["q"] == q)
                & (df["P"] == P_use)
            ]
            if row.empty:
                times.append(0)
            else:
                times.append(row["time_median"].iloc[0])
            labels.append(shape_label)

        bars = ax.bar(labels, times, color=ALGO_COLORS.get(algo, "gray"), alpha=0.8)
        ax.set_title(ALGO_LABELS.get(algo, algo))
        ax.set_ylabel("Time (seconds)" if algo == PARALLEL_ALGOS[0] else "")
        ax.set_ylim(bottom=0)
        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{t:.3f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    fig.tight_layout()
    fname = os.path.join(PLOT_DIR, "shape_comparison.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ============================================================
# PLOT 4: Cost vs P
# Assignment Q2: "Is the algorithm cost-optimal?"
# ============================================================
def plot_cost_vs_P(df):
    print("\n[Plot 4] Cost vs P (answers: Is the algorithm cost-optimal?)")
    for sz in square_sizes(df):
        fig, ax = plt.subplots(figsize=(7, 5))
        subset = df[(df["m"] == sz) & (df["n"] == sz) & (df["q"] == sz)]

        # Serial cost = T_serial * 1 (flat reference line)
        ser = subset[subset["algorithm"] == "ser"]
        if not ser.empty:
            T_ser = ser["time_median"].iloc[0]
            ax.axhline(
                T_ser,
                color=ALGO_COLORS["ser"],
                lw=1.5,
                linestyle="--",
                label=f"T_serial = {T_ser:.4f}s  (cost-optimal target)",
            )

        for algo in PARALLEL_ALGOS:
            s = subset[subset["algorithm"] == algo].sort_values("P")
            if s.empty:
                continue
            ax.plot(
                s["P"],
                s["cost"],
                marker="s",
                color=ALGO_COLORS.get(algo, "gray"),
                label=ALGO_LABELS.get(algo, algo),
            )

        ax.set_xlabel("Number of Processes P")
        ax.set_ylabel("Cost = P × T_par  (process-seconds)")
        ax.set_title(
            f"Cost vs P  |  m=n=q={sz}\n"
            f"(Q2: Is the algorithm cost-optimal? Cost ≈ T_serial → yes)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        fname = os.path.join(PLOT_DIR, f"cost_vs_P_size{sz}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ============================================================
# PLOT 5: Efficiency Heatmap
# Assignment Q2: "Do experimental results contradict asymptotic analysis?"
# ============================================================
def plot_efficiency_heatmap(df):
    if not HAS_SEABORN:
        print("\n[Plot 5] Efficiency heatmap — SKIPPED (seaborn not installed)")
        return

    print(
        "\n[Plot 5] Efficiency heatmap (answers: Do results match asymptotic analysis?)"
    )
    sq = df[(df["m"] == df["n"]) & (df["n"] == df["q"]) & (df["algorithm"] != "ser")]

    for sz in square_sizes(df):
        subset = sq[sq["m"] == sz]
        if subset.empty:
            continue

        pivot = subset.pivot_table(index="algorithm", columns="P", values="efficiency")

        fig, ax = plt.subplots(figsize=(9, 3))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlGn",
            vmin=0,
            vmax=1,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Efficiency (speedup/P)"},
        )
        ax.set_title(
            f"Parallel Efficiency  |  m=n=q={sz}\n"
            f"(Q2: Asymptotic analysis vs reality — ideal = 1.0)"
        )
        ax.set_xlabel("P (processes)")
        ax.set_ylabel("Algorithm")

        fname = os.path.join(PLOT_DIR, f"efficiency_heatmap_size{sz}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ============================================================
# SUMMARY TABLE
# ============================================================
def save_summary_table(df):
    print("\n[Table] Summary table")
    sq = df[(df["m"] == df["n"]) & (df["n"] == df["q"])].copy()
    cols = [
        c
        for c in [
            "algorithm",
            "m",
            "n",
            "q",
            "P",
            "time_median",
            "time_std",
            "speedup",
            "efficiency",
            "cost",
        ]
        if c in sq.columns
    ]
    sq_sorted = sq.sort_values(["m", "algorithm", "P"])[cols]

    path = os.path.join(PLOT_DIR, "summary_table.csv")
    sq_sorted.to_csv(path, index=False, float_format="%.6f")
    print(f"  Saved: {path}")

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 120)
    print(sq_sorted.to_string(index=False))


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/analyze.py results/raw/results_*.csv")
        sys.exit(1)

    csv_paths = sys.argv[1:]
    raw_df, agg_df = load_csv(csv_paths)
    enriched = add_metrics(agg_df)

    plot_speedup_vs_P(enriched)
    plot_runtime_vs_size(enriched)
    plot_shape_comparison(enriched)
    plot_cost_vs_P(enriched)
    plot_efficiency_heatmap(enriched)
    save_summary_table(enriched)

    print(f"\nAll outputs written to: {PLOT_DIR}/")
    print("Copy plots and summary_table.csv into your PowerPoint.")
