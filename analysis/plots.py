import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def apply_style():
    plt.rcParams.update({
        "figure.figsize": (8, 5.5),
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
    })


def safe_corr(x, y):
    vals = pd.concat([x, y], axis=1).dropna()
    if len(vals) < 3 or vals.iloc[:, 0].nunique() < 2 or vals.iloc[:, 1].nunique() < 2:
        return None
    return float(np.corrcoef(vals.iloc[:, 0], vals.iloc[:, 1])[0, 1])


def add_best_fit(ax, x, y):
    if len(x) < 3 or len(np.unique(x)) < 2:
        return
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, m * xs + b, linewidth=1.8, linestyle="--", color="tab:red", label="Best fit")


def save_fig(fig, out_path):
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Saved: " + str(out_path))


def plot_hexbin(df, x, y, title, out_path, xlabel=None, ylabel=None, gridsize=25):
    sub = df[[x, y]].dropna()
    if sub.empty:
        log.warning("hexbin {}: no data".format(out_path.name))
        return

    fig, ax = plt.subplots()
    hb = ax.hexbin(sub[x], sub[y], gridsize=gridsize, mincnt=1, cmap="YlOrRd")
    fig.colorbar(hb, ax=ax, label="Count")
    add_best_fit(ax, sub[x].to_numpy(float), sub[y].to_numpy(float))

    corr = safe_corr(sub[x], sub[y])
    note = f"n={len(sub)}"
    if corr is not None:
        note += f"  |  r = {corr:.3f}"
    ax.text(0.02, 0.97, note, transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.12))
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_weighted_scatter(df, x, y, title, out_path, xlabel=None, ylabel=None,
                          size_col="maps", label_col="label", min_size=20.0, max_size=200.0):
    needed = [c for c in [x, y, size_col, label_col] if c]
    sub = df[needed].dropna().copy()
    if sub.empty:
        log.warning("scatter {}: no data".format(out_path.name))
        return

    fig, ax = plt.subplots()

    if size_col and size_col in sub.columns:
        sv = sub[size_col].astype(float)
        rng = sv.max() - sv.min()
        if rng > 0:
            sizes = min_size + (sv - sv.min()) / rng * (max_size - min_size)
        else:
            sizes = np.full(len(sub), (min_size + max_size) / 2)
    else:
        sizes = np.full(len(sub), 60.0)

    ax.scatter(sub[x], sub[y], s=sizes, alpha=0.6, edgecolors="k", linewidths=0.35)
    add_best_fit(ax, sub[x].to_numpy(float), sub[y].to_numpy(float))

    corr = safe_corr(sub[x], sub[y])
    note = f"n={len(sub)}"
    if corr is not None:
        note += f"  |  r = {corr:.3f}"
    ax.text(0.02, 0.97, note, transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.12))

    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_rank_shift_bump(shift_df, metric_rank_col, rapm_rank_col, label_col, out_path, top_n=25,
                         title="Ranking Shift: Public Rating vs RAPM"):
    sub = shift_df[[label_col, metric_rank_col, rapm_rank_col, "maps"]].dropna().copy()
    if sub.empty:
        log.warning("rank shift plot: no data")
        return

    sub["abs_shift"] = (sub[metric_rank_col] - sub[rapm_rank_col]).abs()
    sub = sub.sort_values("abs_shift", ascending=False).head(top_n).sort_values(metric_rank_col)

    fig, ax = plt.subplots(figsize=(9, max(7, top_n * 0.36)))
    y_pos = np.arange(len(sub))

    for i, (_, row) in enumerate(sub.iterrows()):
        r_rank = row[metric_rank_col]
        a_rank = row[rapm_rank_col]
        color = "tab:green" if a_rank < r_rank else "tab:orange"
        ax.plot([r_rank, a_rank], [i, i], color=color, linewidth=2.0, alpha=0.75)
        ax.scatter(r_rank, i, color="tab:blue", s=60, zorder=3, label="Rating rank" if i == 0 else "")
        ax.scatter(a_rank, i, color="tab:red", s=60, zorder=3, label="RAPM rank" if i == 0 else "")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub[label_col], fontsize=8)
    ax.invert_xaxis()
    ax.set_xlabel("Rank (lower = better)")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right")
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_top_bottom_bar(df, metric, label_col, out_path, n=15, title=None, xlabel=None):
    sub = df[[label_col, metric]].dropna().copy()
    if sub.empty:
        log.warning("bar chart {}: no data".format(out_path.name))
        return

    top = sub.nlargest(n, metric)
    bot = sub.nsmallest(n, metric)
    combined = pd.concat([top, bot]).drop_duplicates(label_col).sort_values(metric)
    colors = ["tab:green" if v >= 0 else "tab:red" for v in combined[metric]]

    fig, ax = plt.subplots(figsize=(9, max(7, len(combined) * 0.38)))
    ax.barh(combined[label_col], combined[metric], color=colors, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.axvline(0, linewidth=1.0, color="k")
    ax.set_title(title or f"Top and Bottom {n} by {metric}")
    ax.set_xlabel(xlabel or metric)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_coefficient_distribution(values, title, out_path, xlabel="Coefficient Value", bins=50):
    vals = pd.Series(values).dropna().astype(float)
    if vals.empty:
        log.warning("distribution plot {}: no data".format(out_path.name))
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=bins, color="tab:blue", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(vals.mean(), color="k", linewidth=1.2, linestyle="--", label=f"Mean ({vals.mean():.3f})")
    ax.legend()

    try:
        from scipy.stats import gaussian_kde
        if vals.nunique() > 5 and vals.std() > 0:
            kde = gaussian_kde(vals)
            x_range = np.linspace(vals.min(), vals.max(), 200)
            ax2 = ax.twinx()
            ax2.plot(x_range, kde(x_range), color="tab:red", linewidth=2.0, alpha=0.8)
            ax2.set_yticks([])
    except ImportError:
        pass

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")

    zeros = int((vals.abs() < 1e-4).sum())
    stats_text = "n = {}\nmean = {:.4f}\nstd = {:.4f}\napprox zeros = {} ({:.1f}%)".format(
        len(vals), vals.mean(), vals.std(), zeros, (zeros / len(vals)) * 100
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_context_coefficients(coeff_df, out_path):
    if coeff_df.empty:
        log.warning("context coefficients: no data")
        return

    metrics = coeff_df["metric"].tolist()
    tm_betas = coeff_df["coef_tm_pre"].tolist()
    opp_betas = coeff_df["coef_opp_pre"].tolist()

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, tm_betas, width, label="Teammate beta", color="tab:blue", alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, opp_betas, width, label="Opponent beta", color="tab:orange", alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Coefficient (standardised, SD units)")
    ax.set_title("Context Regression Coefficients by Outcome")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_context_coeff_heatmap(coeff_df, out_path):
    if coeff_df.empty:
        return

    metrics = coeff_df["metric"].tolist() if "metric" in coeff_df.columns else []
    coef_cols = ["coef_own_pre", "coef_tm_pre", "coef_opp_pre"]
    present = [c for c in coef_cols if c in coeff_df.columns]
    if not present or not metrics:
        return

    data = coeff_df.set_index("metric")[present]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    cax = ax.imshow(data.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, label="Coefficient")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([c.replace("coef_", "").replace("_pre", "").replace("_", " ") for c in present], fontsize=10)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=10)
    ax.set_title("Context Regression Coefficients")
    for i in range(len(metrics)):
        for j in range(len(present)):
            val = data.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                        color="black" if abs(val) < 0.5 else "white")
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_iti_top_bottom(iti_df, out_path, n=15, title="Indirect Teammate Impact (ITI): Top and Bottom Players", role_df=None):
    required = {"player_a", "iti", "n_distinct_teammates", "iti_t_stat"}
    if not required.issubset(iti_df.columns):
        log.warning("ITI plot: required columns missing")
        return

    sub = iti_df[
        (iti_df["n_distinct_teammates"] >= 4) &
        iti_df["iti_t_stat"].notna() &
        (iti_df["iti_p_value"] < 0.05)
    ].copy()

    if sub.empty:
        log.warning("ITI plot: no valid rows after filtering")
        return

    top = sub.nlargest(n, "iti")
    bot = sub.nsmallest(n, "iti")
    combined = pd.concat([top, bot]).drop_duplicates("player_a_id").sort_values("iti")

    role_lookup = {}
    if role_df is not None and not role_df.empty:
        role_lookup = role_df.set_index("hltv_player_id")["role"].to_dict()

    def label_with_role(row):
        name = row["player_a"]
        role = role_lookup.get(row.get("player_a_id"), "")
        return f"{name} ({role})" if role else name

    combined["display_label"] = combined.apply(label_with_role, axis=1)
    colors = ["tab:green" if v >= 0 else "tab:red" for v in combined["iti"]]

    err_lo = err_hi = None
    if {"iti_ci_lower", "iti_ci_upper"}.issubset(combined.columns):
        ci_lo = combined["iti_ci_lower"].fillna(combined["iti"])
        ci_hi = combined["iti_ci_upper"].fillna(combined["iti"])
        err_lo = (combined["iti"] - ci_lo).abs().clip(lower=0).to_numpy()
        err_hi = (ci_hi - combined["iti"]).abs().clip(lower=0).to_numpy()

    fig, ax = plt.subplots(figsize=(9.5, max(8, len(combined) * 0.4)))
    bar_kwargs = dict(color=colors, alpha=0.8, edgecolor="k", linewidth=0.3)
    if err_lo is not None:
        bar_kwargs["xerr"] = [err_lo, err_hi]
        bar_kwargs["error_kw"] = dict(ecolor="black", capsize=3, alpha=0.55, elinewidth=0.9)

    ax.barh(combined["display_label"], combined["iti"], **bar_kwargs)
    ax.axvline(0, linewidth=1.0, color="k")
    ax.set_title(title)
    ax.set_xlabel("ITI: mean teammate residual rating delta (with A - without A)")
    ax.text(0.98, 0.02, "Error bars: 95% CI", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, alpha=0.7)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_roster_change_before_after(events_df, out_path, n=10, title="Roster-Change Event Study: Retained Teammate Residuals"):
    if events_df.empty:
        log.warning("roster change plot: no data")
        return

    top = pd.concat([
        events_df.nlargest(n // 2, "delta"),
        events_df.nsmallest(n // 2, "delta")
    ]).drop_duplicates().sort_values("delta")

    labels = [f"{r['team_name']}: {r['player_a']}" for _, r in top.iterrows()]
    pre_vals = top["mean_pre"].tolist()
    post_vals = top["mean_post"].tolist()
    deltas = top["delta"].tolist()

    y = np.arange(len(top))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.6)))
    ax.barh(y + height / 2, pre_vals, height, label="Before (mean residual)", color="#aaaaaa", alpha=0.85, edgecolor="k", linewidth=0.3)
    ax.barh(y - height / 2, post_vals, height, label="After (mean residual)",
            color=["tab:green" if d >= 0 else "tab:red" for d in deltas], alpha=0.85, edgecolor="k", linewidth=0.3)

    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean retained teammate residual rating")
    ax.set_title(title)
    ax.legend(fontsize=9)

    for i, (pre, post, delta, p) in enumerate(zip(pre_vals, post_vals, deltas, top["p_value"].tolist())):
        sig = "*" if p < 0.05 else ""
        ax.text(max(pre, post, 0) + 0.005, i, f"delta={delta:+.3f}{sig}", va="center", fontsize=8)

    fig.tight_layout()
    save_fig(fig, out_path)


def plot_placebo_comparison(placebo_dist_df, real_uplift_df, out_path):
    if "outcome" in real_uplift_df.columns and "t_stat" in real_uplift_df.columns:
        rating_rows = real_uplift_df[real_uplift_df["outcome"] == "rating"]
        real_abs_t = rating_rows["t_stat"].abs().dropna().astype(float)
    elif "abs_t" in real_uplift_df.columns:
        real_abs_t = real_uplift_df["abs_t"].dropna().astype(float)
    elif "t_stat" in real_uplift_df.columns:
        real_abs_t = real_uplift_df["t_stat"].abs().dropna().astype(float)
    else:
        log.warning("placebo comparison: no t_stat column found")
        return

    placebo_abs_t = placebo_dist_df["abs_t"].dropna().astype(float)

    if real_abs_t.empty or placebo_abs_t.empty:
        log.warning("placebo comparison: empty data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    try:
        from scipy.stats import gaussian_kde
        x_range = np.linspace(0, max(real_abs_t.quantile(0.99), placebo_abs_t.quantile(0.99)), 300)

        if real_abs_t.std() > 0:
            kde_real = gaussian_kde(real_abs_t)
            ax.fill_between(x_range, kde_real(x_range), alpha=0.4, color="tab:blue", label="Real pairs")
            ax.plot(x_range, kde_real(x_range), color="tab:blue", linewidth=1.8)

        if placebo_abs_t.std() > 0:
            kde_placebo = gaussian_kde(placebo_abs_t)
            ax.fill_between(x_range, kde_placebo(x_range), alpha=0.4, color="tab:orange", label="Placebo")
            ax.plot(x_range, kde_placebo(x_range), color="tab:orange", linewidth=1.8)

    except ImportError:
        bins = np.linspace(0, max(real_abs_t.quantile(0.99), placebo_abs_t.quantile(0.99)), 50)
        ax.hist(real_abs_t, bins=bins, alpha=0.5, color="tab:blue", label="Real pairs", density=True)
        ax.hist(placebo_abs_t, bins=bins, alpha=0.5, color="tab:orange", label="Placebo", density=True)

    p95_real = real_abs_t.quantile(0.95)
    p95_placebo = placebo_abs_t.quantile(0.95)
    ax.axvline(p95_real, color="tab:blue", linewidth=1.2, linestyle="--", alpha=0.8, label=f"Real 95th ({p95_real:.2f})")
    ax.axvline(p95_placebo, color="tab:orange", linewidth=1.2, linestyle="--", alpha=0.8, label=f"Placebo 95th ({p95_placebo:.2f})")

    ax.set_xlabel("|t| statistic")
    ax.set_ylabel("Density")
    ax.set_title("Real vs Placebo |t| Distribution")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_prediction_model_comparison(metrics_df, out_path):
    if metrics_df.empty:
        log.warning("model comparison plot: no data")
        return

    required = {"model", "mae", "rmse", "win_loss_accuracy"}
    if not required.issubset(metrics_df.columns):
        log.warning("model comparison plot: missing columns " + str(required - set(metrics_df.columns)))
        return

    models = metrics_df["model"].tolist()

    # If I don't shorten the labels, the bar is too large. very cool.
    short_labels = []
    for m in models:
        if "Zero" in m:
            short_labels.append("Zero\nBaseline")
        elif "Model 1" in m or "Individual" in m:  # TODO: Remove the or and only use one thing
            short_labels.append("Model 1\n(Individual RAPM)")
        elif "Model 2" in m or "Joint" in m:
            short_labels.append("Model 2\n(Joint RAPM\n+ Pairs)")
        else:
            short_labels.append(m)

    metrics_info = [
        ("mae", "Mean Absolute Error", "MAE (rounds)", True),
        ("rmse", "Root Mean Squared Error", "RMSE (rounds)", True),
        ("win_loss_accuracy", "Win/Loss Accuracy", "Accuracy", False),
    ]

    colors = ["#aaaaaa", "tab:blue", "tab:orange"]
    edgecolor = ["#777777", "#1a4d7a", "#8a3e10"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    fig.suptitle("Predictive Validation: Model Comparison", fontsize=14, fontweight="bold")

    for ax, (col, panel_title, ylabel, lower_is_better) in zip(axes, metrics_info):
        vals = metrics_df[col].tolist()
        bars = ax.bar(short_labels, vals, color=colors, edgecolor=edgecolor, linewidth=1.1, width=0.55)
        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=7)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis="x", labelsize=8.5)

        for bar, v in zip(bars, vals):
            fmt = f"{v:.1%}" if col == "win_loss_accuracy" else f"{v:.4f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004 * max(vals),
                fmt, ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

        span = max(vals) - min(vals)
        ax.set_ylim(min(vals) - span * 3, max(vals) + span * 1.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, out_path)


def plot_prediction_pair_coef_dist(pair_coef_df, out_path):
    if pair_coef_df.empty or "pair_coef" not in pair_coef_df.columns:
        log.warning("pair coef distribution: no data")
        return

    vals = pair_coef_df["pair_coef"].dropna().astype(float)
    if vals.empty:
        return

    threshold = 0.1
    positive = vals[vals > threshold]
    near_zero = vals[(vals >= -threshold) & (vals <= threshold)]
    negative = vals[vals < -threshold]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(vals.min() - 0.05, vals.max() + 0.05, 42)

    ax.hist(positive, bins=bins, color="tab:blue", alpha=0.80, label=f"Positive synergy  (n = {len(positive)})")
    ax.hist(near_zero, bins=bins, color="#cccccc", alpha=0.80, edgecolor="#aaaaaa", label=f"Near zero  (n = {len(near_zero)})")
    ax.hist(negative, bins=bins, color="tab:red", alpha=0.80, label=f"Negative synergy  (n = {len(negative)})")

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.axvline(threshold, color="#888888", linewidth=0.8, linestyle=":")
    ax.axvline(-threshold, color="#888888", linewidth=0.8, linestyle=":")

    ax.set_xlabel("Pair coefficient (additional round differential contribution)", fontsize=11)
    ax.set_ylabel("Number of pairs", fontsize=11)
    ax.set_title("Distribution of Pair Synergy Coefficients (Joint RAPM Model)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_prediction_bootstrap_ci(bs_raw, out_path):
    raw = bs_raw.get("raw", {})
    if not raw:
        log.warning("bootstrap CI plot: no raw samples")
        return

    labels = ["MAE", "RMSE", "Accuracy"]
    samples = [raw.get("mae", []), raw.get("rmse", []), raw.get("accuracy", [])]

    if not any(samples):
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.boxplot(
        samples, tick_labels=labels, vert=False, patch_artist=True, widths=0.45, showfliers=False,
        boxprops=dict(facecolor="tab:blue", alpha=0.6),
        whiskerprops=dict(linewidth=1.2, color="#444444"),
        capprops=dict(linewidth=1.2, color="#444444"),
    )
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7, label="Zero (no difference)")
    ax.set_xlabel("Improvement of Model 2 over Model 1  (positive = Model 2 better)", fontsize=10)
    ax.set_title("Bootstrap Distribution of Model 2 Improvement over Model 1\n(1,000 iterations)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_pair_survival(pair_survival, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_total = len(pair_survival)
    n_survived = int(pair_survival["appeared_in_test"].sum())
    n_absent = n_total - n_survived

    survived = pair_survival[pair_survival["appeared_in_test"]]

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], figure=fig)

    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(
        ["Appeared\nin test", "Absent\nfrom test"],
        [n_survived, n_absent],
        color=["#4C9BE8", "#CCCCCC"], width=0.5, edgecolor="none",
    )
    ax1.set_ylabel("Number of pairs")
    ax1.set_title("Pair survival into test period")
    for bar, val in zip(bars, [n_survived, n_absent]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(val), ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax1.set_ylim(0, n_total * 1.15)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[1])
    ax2.hist(
        survived["test_maps"],
        bins=range(1, int(survived["test_maps"].max()) + 2),
        color="#4C9BE8", edgecolor="white", linewidth=0.5
    )
    ax2.set_xlabel("Test maps played together")
    ax2.set_ylabel("Number of pairs")
    ax2.set_title("Maps played together in test period\n(surviving pairs only)")
    ax2.spines[["top", "right"]].set_visible(False)

    median_maps = survived["test_maps"].median()
    ax2.axvline(median_maps, color="#E84C4C", linestyle="--", linewidth=1.2, label=f"Median: {int(median_maps)} maps")
    ax2.legend(frameon=False)

    fig.suptitle(
        f"{n_survived} of {n_total} trained pairs appeared in the test period ({100*n_survived/n_total:.0f}%)",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved pair survival plot -> {out_path}")