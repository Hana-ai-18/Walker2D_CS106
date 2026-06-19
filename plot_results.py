"""
Plot results - fixed version.
Generates learning curves và summary table giống paper.
"""

import os, sys, glob, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

ALGO_COLORS = {
    "PPO":  "#2ca02c",
    "TD3":  "#1f77b4",
    "SAC":  "#d62728",
    "MTD3": "#9467bd",
    "MSAC": "#ff7f0e",
}
ALGO_ORDER = ["PPO", "TD3", "SAC", "MTD3", "MSAC"]

ENV_LABELS = {
    "MDP": "Original MDP",
    "RN":  "POMDP-RN (Noise)",
    "FLK": "POMDP-FLK (Flickering)",
    "RSM": "POMDP-RSM (Sensor Missing)",
    "RV":  "POMDP-RV (Remove Velocity)",
}
ENV_ORDER = ["MDP", "FLK", "RN", "RSM", "RV"]


def load_results(results_dir):
    csvs = glob.glob(os.path.join(results_dir, "*.csv"))
    csvs = [f for f in csvs if "all_results" not in os.path.basename(f)]
    if not csvs:
        # fallback: dùng all_results.csv
        all_f = os.path.join(results_dir, "all_results.csv")
        if os.path.exists(all_f):
            csvs = [all_f]
        else:
            print("Không tìm thấy CSV nào!")
            return pd.DataFrame()
    dfs = []
    for f in csvs:
        try:
            dfs.append(pd.read_csv(f))
        except:
            pass
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates()
    print(f"Loaded {len(df)} rows từ {len(csvs)} files")
    print(f"  Algorithms: {sorted(df['algorithm'].unique())}")
    print(f"  Envs: {sorted(df['env_type'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    print(f"  Timestep range: {df['timestep'].min():,} - {df['timestep'].max():,}")
    return df


def plot_learning_curves(df, output_dir):
    """Learning curve mỗi env — giống Figure 1/4 trong paper."""
    env_types = [e for e in ENV_ORDER if e in df["env_type"].unique()]

    for env_type in env_types:
        env_df = df[df["env_type"] == env_type]
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False

        for algo in ALGO_ORDER:
            algo_df = env_df[env_df["algorithm"] == algo]
            if algo_df.empty:
                continue

            grouped = algo_df.groupby("timestep")["mean_return"]
            mean_vals = grouped.mean()
            std_vals  = grouped.std().fillna(0)

            steps = mean_vals.index.values
            means = mean_vals.values
            stds  = std_vals.values

            if len(means) < 2:
                continue

            sigma = min(20, max(1, len(means)//5))
            means_s = gaussian_filter1d(means, sigma=sigma)
            stds_s  = gaussian_filter1d(stds,  sigma=sigma)

            color = ALGO_COLORS.get(algo, "gray")
            ax.plot(steps, means_s, color=color, label=algo, linewidth=2)
            ax.fill_between(steps,
                            means_s - 0.5*stds_s,
                            means_s + 0.5*stds_s,
                            alpha=0.2, color=color)
            plotted = True

        if not plotted:
            plt.close()
            continue

        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Average Return", fontsize=12)
        ax.set_title(f"Walker2D - {ENV_LABELS.get(env_type, env_type)}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        max_step = df["timestep"].max()
        if max_step > 0:
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
            )

        plt.tight_layout()
        fname = os.path.join(output_dir, f"learning_curve_{env_type}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


def plot_bar_chart(df, output_dir):
    """Bar chart so sánh — giống Figure 3 trong paper."""
    env_types = [e for e in ENV_ORDER if e in df["env_type"].unique()]
    algos     = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]

    if not env_types or not algos:
        return

    n_envs  = len(env_types)
    n_algos = len(algos)
    fig, ax = plt.subplots(figsize=(max(10, n_envs*2.5), 6))
    x = np.arange(n_envs)
    width = 0.8 / n_algos

    for i, algo in enumerate(algos):
        means, stds = [], []
        for env_type in env_types:
            sub = df[(df["algorithm"]==algo) & (df["env_type"]==env_type)]
            if sub.empty:
                means.append(0); stds.append(0)
                continue
            per_seed = sub.groupby("seed")["mean_return"].max()
            means.append(per_seed.mean())
            stds.append(per_seed.std() if len(per_seed)>1 else 0)

        offset = (i - n_algos/2 + 0.5) * width
        ax.bar(x + offset, means, width*0.9,
               label=algo,
               color=ALGO_COLORS.get(algo, "gray"),
               yerr=stds, capsize=3,
               error_kw={"elinewidth":1.5})

    ax.set_xlabel("Environment", fontsize=12)
    ax.set_ylabel("Max Average Return", fontsize=12)
    ax.set_title("Walker2D: Algorithm Comparison across POMDP Variants", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS.get(e,e) for e in env_types],
                       rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fname = os.path.join(output_dir, "bar_chart_comparison.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def print_and_save_table(df, output_dir):
    """
    Tạo summary table giống Table 4 trong paper.
    Lưu ra CSV và in ra terminal.
    """
    env_types = [e for e in ENV_ORDER if e in df["env_type"].unique()]
    algos     = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]

    rows = []
    print(f"\n{'='*75}")
    print("SUMMARY TABLE: Max Average Return (mean ± std over seeds)")
    print("Walker2D — giống Table 4 trong paper")
    print(f"{'='*75}")
    header = f"{'Env':<10}" + "".join(f"{a:>14}" for a in algos)
    print(header)
    print("-"*75)

    for env_type in env_types:
        row = {"env_type": f"Walker2D-{env_type}"}
        line = f"{row['env_type']:<10}"
        for algo in algos:
            sub = df[(df["algorithm"]==algo) & (df["env_type"]==env_type)]
            if sub.empty:
                line += f"{'N/A':>14}"
                row[algo] = "N/A"
                continue
            per_seed = sub.groupby("seed")["mean_return"].max()
            m = per_seed.mean()
            s = per_seed.std() if len(per_seed)>1 else 0
            line += f"{m:>8.1f}±{s:<4.1f}"
            row[algo] = f"{m:.1f}±{s:.1f}"
        rows.append(row)
        print(line)

    print(f"{'='*75}\n")

    # Lưu table ra CSV
    table_df = pd.DataFrame(rows)
    table_path = os.path.join(output_dir, "summary_table.csv")
    table_df.to_csv(table_path, index=False)
    print(f"Table saved: {table_path}")
    return table_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir",  type=str, default="plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_results(args.results_dir)
    if df.empty:
        print("Không có dữ liệu!")
        sys.exit(0)

    print_and_save_table(df, args.output_dir)
    plot_learning_curves(df, args.output_dir)
    plot_bar_chart(df, args.output_dir)

    print(f"\n✓ Tất cả plots đã lưu vào: {args.output_dir}/")
