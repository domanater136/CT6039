import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _pval_stars(p):
    if p is None or pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "."
    return ""


def write_data_report(audit, out_dir):
    path = out_dir / "report_data_audit.md"

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Data Audit Report\n\n")
        f.write("## Sample Summary\n\n")
        f.write("| Statistic | Value |\n|---|---|\n")
        f.write(f"| Total raw rows | {audit.get('total_rows', 'n/a'):,} |\n")
        f.write(f"| Matches | {audit.get('total_matches', 'n/a'):,} |\n")
        f.write(f"| Maps (raw) | {audit.get('total_maps', 'n/a'):,} |\n")
        f.write(f"| Players | {audit.get('total_players', 'n/a'):,} |\n")
        f.write(f"| Maps excluded (wrong player count) | {audit.get('maps_not_10_players', 0):,} |\n")
        f.write(f"| Maps excluded (null metrics) | {audit.get('maps_with_null_metrics', 0):,} |\n")
        f.write(f"| Maps excluded (bad round totals) | {audit.get('maps_bad_rounds', 0):,} |\n")
        f.write(f"| Maps retained | {audit.get('maps_retained', 'n/a'):,} |\n")

    log.info("data audit written to " + str(path))


def write_context_report(coeff_df, out_dir):
    path = out_dir / "report_context_regression.md"

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Context-Sensitivity Regression Report\n\n")

        if coeff_df is None or coeff_df.empty:
            f.write("_No results available._\n")
            return

        f.write("## Coefficients\n\n")
        f.write("| Metric | N | Adj R2 | Own pre (b) | Teammate pre (b) | sig | Opponent pre (b) | sig |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")

        for _, row in coeff_df.iterrows():
            metric = row.get("metric", "")
            n = int(row["n_obs"]) if pd.notna(row.get("n_obs")) else ""
            r2 = "{:.4f}".format(row["r2_adj"]) if pd.notna(row.get("r2_adj")) else ""
            own = "{:.4f}".format(row["coef_own_pre"]) if pd.notna(row.get("coef_own_pre")) else ""
            tm = "{:.4f}".format(row["coef_tm_pre"]) if pd.notna(row.get("coef_tm_pre")) else ""
            tm_sig = _pval_stars(row.get("coef_tm_pre_pval"))
            opp = "{:.4f}".format(row["coef_opp_pre"]) if pd.notna(row.get("coef_opp_pre")) else ""
            opp_sig = _pval_stars(row.get("coef_opp_pre_pval"))
            f.write(f"| {metric} | {n} | {r2} | {own} | {tm} | {tm_sig} | {opp} | {opp_sig} |\n")

    log.info("context report written")


def write_rapm_report(rapm_scores, shift_df, out_dir, role_df=None):
    path = out_dir / "report_rapm.md"

    role_lookup = {}
    if role_df is not None and not role_df.empty:
        role_lookup = dict(zip(role_df["hltv_player_id"], role_df["role"]))

    best_alpha = rapm_scores.attrs.get("best_alpha", "n/a")
    n_maps = rapm_scores.attrs.get("n_maps", "n/a")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# RAPM Report\n\n")
        f.write("## Config\n\n")
        f.write(f"- Best alpha: **{best_alpha}**\n")
        f.write(f"- Maps: **{n_maps}**\n")
        f.write(f"- Active players: **{len(rapm_scores)}**\n\n")

        if rapm_scores.empty:
            f.write("_No RAPM results._\n")
            return

        if not shift_df.empty and "avg_rating" in shift_df.columns:
            enriched = rapm_scores.merge(
                shift_df[["hltv_player_id", "avg_rating", "maps"]],
                on="hltv_player_id", how="left"
            )
        else:
            enriched = rapm_scores.copy()
            enriched["avg_rating"] = np.nan
            enriched["maps"] = np.nan

        total = len(enriched)
        enriched["rapm_rank"] = enriched["rapm_score"].rank(ascending=False, method="min").astype(int)
        enriched["rating_rank"] = enriched["avg_rating"].rank(ascending=False, method="min").astype(int)
        enriched["role"] = enriched["hltv_player_id"].map(role_lookup).fillna("-")

        def write_player_table(f, rows):
            f.write("| Player | Role | Maps | RAPM | Rating |\n|---|---|---|---|---|\n")
            for _, row in rows.iterrows():
                rapm_str = f"{row['rapm_score']:.3f} (#{int(row['rapm_rank'])}/{total})"
                if pd.notna(row.get("avg_rating")):
                    rat_str = f"{row['avg_rating']:.3f} (#{int(row['rating_rank'])}/{total})"
                else:
                    rat_str = "n/a"
                maps_str = str(int(row["maps"])) if pd.notna(row.get("maps")) else "n/a"
                f.write(f"| {row['label']} | {row['role']} | {maps_str} | {rapm_str} | {rat_str} |\n")

        f.write("## Top 25 by RAPM\n\n")
        write_player_table(f, enriched.head(25))

        f.write("\n## Bottom 25 by RAPM\n\n")
        write_player_table(f, enriched.tail(25))

        if not shift_df.empty and "delta_rating" in shift_df.columns:
            f.write("\n## Rank Shift - RAPM higher than raw rating (top 20)\n\n")
            f.write("| Player | Role | Maps | RAPM | Rating | Delta |\n|---|---|---|---|---|---|\n")

            top_shift = shift_df.nlargest(20, "delta_rating")
            for _, row in top_shift.iterrows():
                pid = row["hltv_player_id"]
                role = role_lookup.get(pid, "-")
                match = enriched[enriched["hltv_player_id"] == pid]
                if match.empty:
                    continue
                m = match.iloc[0]
                rapm_str = "{:.3f} (#{}/{})".format(m["rapm_score"], int(m["rapm_rank"]), total)
                rat_str = "{:.3f} (#{}/{})".format(m["avg_rating"], int(m["rating_rank"]), total) if pd.notna(m.get("avg_rating")) else "n/a"
                maps_str = str(int(m["maps"])) if pd.notna(m.get("maps")) else "n/a"
                delta_val = row["delta_rating"]
                delta = "+{}".format(int(delta_val)) if delta_val > 0 else str(int(delta_val))
                f.write(f"| {row['label']} | {role} | {maps_str} | {rapm_str} | {rat_str} | {delta} |\n")

    log.info("RAPM report written")


def write_prediction_report(prediction_results, out_dir):
    path = out_dir / "report_prediction.md"

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Predictive Validation Report\n\n")

        cutoff = prediction_results.get("cutoff_date", "n/a")
        train_n = prediction_results.get("train_maps", "n/a")
        test_n = prediction_results.get("test_maps", "n/a")
        n_players = prediction_results.get("n_players_train", "n/a")

        f.write("## Setup\n\n")
        f.write(f"- Cutoff: **{cutoff}**\n")
        if isinstance(train_n, int):
            f.write(f"- Training maps: **{train_n:,}**\n")
        else:
            f.write(f"- Training maps: **{train_n}**\n")
        if isinstance(test_n, int):
            f.write(f"- Test maps: **{test_n:,}**\n")
        else:
            f.write(f"- Test maps: **{test_n}**\n")
        f.write(f"- Players in training: **{n_players}**\n\n")

        metrics = prediction_results.get("metrics_table", pd.DataFrame())
        if not metrics.empty:
            f.write("## Metrics\n\n")
            f.write(metrics.to_markdown(index=False))
            f.write("\n\n")

        tests = prediction_results.get("statistical_tests", {})
        mcn = tests.get("mcnemar_m1_vs_m2", {})

        if mcn:
            f.write("## McNemar Test\n\n")
            f.write(f"- M1 correct / M2 wrong: **{mcn.get('b', 'n/a')}**\n")
            f.write(f"- M2 correct / M1 wrong: **{mcn.get('c', 'n/a')}**\n")
            f.write(f"- Mid-p: **{mcn.get('p_value_midp', 'n/a')}**\n")
            f.write(f"- Significant: **{mcn.get('significant', 'n/a')}**\n\n")

        boot = tests.get("bootstrap_m1_vs_m2", {})
        if boot:
            f.write("## Bootstrap CIs\n\n")
            f.write("| Metric | Improvement | 95% CI | Excludes zero |\n|---|---|---|---|\n")
            for m in ["mae", "rmse", "accuracy"]:
                r = boot.get(m, {})
                if not r:
                    continue
                f.write("| {} | {} | [{}, {}] | {} |\n".format(
                    m.upper(), r.get("observed", ""), r.get("ci_lower", ""),
                    r.get("ci_upper", ""), r.get("excludes_zero", "")
                ))
            f.write("\n")

        pair_coef_table = prediction_results.get("pair_coef_table", pd.DataFrame())
        if not pair_coef_table.empty:
            pos = pair_coef_table[pair_coef_table["pair_coef"] > 0].nlargest(25, "pair_coef")
            neg = pair_coef_table[pair_coef_table["pair_coef"] < 0].nsmallest(25, "pair_coef")

            f.write("## Top Synergistic Pairs\n\n")
            f.write(pos.drop(columns=["abs_pair_coef"], errors="ignore").round(4).to_markdown(index=False))
            f.write("\n\n## Top Anti-Synergistic Pairs\n\n")
            f.write(neg.drop(columns=["abs_pair_coef"], errors="ignore").round(4).to_markdown(index=False))
            f.write("\n")

    log.info("prediction report written to " + str(path))


def write_summary_report(audit, rapm_scores, shift_df, coeff_df, out_dir):
    path = out_dir / "report_summary.md"
    best_alpha = rapm_scores.attrs.get("best_alpha", "n/a") if not rapm_scores.empty else "n/a"

    with open(path, "w", encoding="utf-8") as f:
        f.write("# CS2 Dissertation Analysis - Summary\n\n---\n\n")

        f.write("## 1. Dataset\n\n")
        f.write(f"- Maps retained: **{audit.get('maps_retained', 'n/a'):,}**\n")
        f.write("- Maps excluded: **{}** (count: {}, nulls: {}, bad rounds: {})\n".format(
            audit.get("maps_excluded_total", 0),
            audit.get("maps_not_10_players", 0),
            audit.get("maps_with_null_metrics", 0),
            audit.get("maps_bad_rounds", 0),
        ))
        f.write(f"- Players: **{audit.get('total_players', 'n/a'):,}**\n")
        f.write(f"- Matches: **{audit.get('total_matches', 'n/a'):,}**\n\n---\n\n")

        f.write("## 2. Context Regressions\n\n")
        if not coeff_df.empty and "coef_tm_pre" in coeff_df.columns:
            for _, row in coeff_df.iterrows():
                sig = _pval_stars(row.get("coef_tm_pre_pval"))
                f.write("- **{}**: teammate b = {:.4f} {} (adj R2 = {:.4f})\n".format(
                    row["metric"], row["coef_tm_pre"], sig,
                    row.get("r2_adj", float("nan"))
                ))
        else:
            f.write("_Context regressions not run._\n")

        f.write("\n---\n\n## 3. RAPM\n\n")
        f.write(f"- Ridge alpha: **{best_alpha}**\n")
        f.write(f"- Active players: **{len(rapm_scores):,}**\n\n")
        if not rapm_scores.empty:
            f.write(rapm_scores[["label", "rapm_score"]].head(10).round(4).to_markdown(index=False))
        else:
            f.write("_No results._")

        f.write("\n\n---\n\n## 4. Rank Shift\n\n")
        if not shift_df.empty:
            fh = int(shift_df["flag_high_rating_low_rapm"].sum()) if "flag_high_rating_low_rapm" in shift_df.columns else 0
            fl = int(shift_df["flag_low_rating_high_rapm"].sum()) if "flag_low_rating_high_rapm" in shift_df.columns else 0
            f.write(f"- Rating overestimates vs RAPM: **{fh}**\n")
            f.write(f"- Rating underestimates vs RAPM: **{fl}**\n")
        else:
            f.write("_Not available._\n")

        f.write("\n---\n\n## 5. Teammate Influence\n\n")
        f.write("Full results in `teammate_influence/report_teammate_influence.md`\n")

    log.info("summary report written")