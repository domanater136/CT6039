import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="CS2 dissertation analysis pipeline.")
    p.add_argument("--db", default="hltv_dissertation.db")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--min-maps", type=int, default=20)
    p.add_argument("--min-maps-rapm", type=int, default=50)
    p.add_argument("--cutoff", default="2025-07-01")
    p.add_argument("--min-pair-maps", type=int, default=20)
    p.add_argument("--placebo-iters", type=int, default=100)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--pre-window", type=int, default=10)
    p.add_argument("--post-window", type=int, default=10)
    p.add_argument("--skip-context", action="store_true")
    p.add_argument("--skip-rapm", action="store_true")
    p.add_argument("--skip-prediction", action="store_true")
    p.add_argument("--skip-influence", action="store_true")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    data_dir = Path("data")
    plot_dir = Path("plots")

    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "analysis.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    log.info("Starting analysis pipeline")

    # --- load data ---
    from analysis.data import build_analysis_tables
    conn = sqlite3.connect(args.db)
    tables = build_analysis_tables(conn)
    conn.close()

    conn = sqlite3.connect(args.db)
    role_df = pd.read_sql_query(
        "SELECT hltv_player_id, role FROM players WHERE role IS NOT NULL AND role != 'Unknown'",
        conn
    )
    conn.close()

    VALID_ROLES = {"Rifler", "AWPer", "IGL", "Entry", "Lurker", "Support"}
    ROLE_NORMALISE = {"In-game leader": "IGL"}

    # TODO: Uncomplicate this. Far to dififuclt to understand
    def clean_role(r_str):
        if not r_str or pd.isna(r_str):
            return None
        bits = [ROLE_NORMALISE.get(x.strip(), x.strip()) for x in r_str.split(",")]
        bits = [x for x in bits if x in VALID_ROLES]
        return ", ".join(bits) if bits else None

    role_df["role"] = role_df["role"].apply(clean_role)
    role_df = role_df[role_df["role"].notna()].copy()

    audit = tables["audit"]
    player_map = tables["player_map"]

    print("Building features...")
    from analysis.features import build_features
    player_map = build_features(player_map)

    bad = player_map[player_map["result_diff"].abs() > 20]
    if len(bad) > 0:
        log.warning(f"Found {len(bad)} rows with result_diff > 20. might be worth investigating")

    log.info("Building player summaries")
    summary = player_map.groupby("hltv_player_id").agg(
        maps=("map_id", "nunique"),
        label=("label", "first"),
        avg_rating=("rating", "mean"),
        avg_swing=("swing", "mean"),
        avg_adr=("adr", "mean"),
        avg_kd_log=("kd_log", "mean"),
        total_kills=("kills", "sum"),
        total_deaths=("deaths", "sum"),
        win_rate=("team_won", "mean"),
    ).reset_index()
    summary = summary[summary["maps"] >= args.min_maps].copy()
    summary["kill_diff"] = summary["total_kills"] - summary["total_deaths"]
    player_summary = summary.sort_values("avg_rating", ascending=False).reset_index(drop=True)
    player_summary.to_csv(data_dir / "player_summary.csv", index=False)

    tpm = player_map.groupby("hltv_player_id").agg(
        true_plus_minus=("result_diff", "mean"),
        pm_maps=("map_id", "nunique"),
        avg_rating=("rating", "mean"),
        win_rate=("team_won", "mean"),
        label=("label", "first"),
    ).reset_index()
    tpm = tpm[tpm["pm_maps"] >= args.min_maps_rapm].copy()
    true_pm = tpm.sort_values("true_plus_minus", ascending=False).reset_index(drop=True)
    true_pm.to_csv(data_dir / "true_plus_minus.csv", index=False)

    coeff_df = pd.DataFrame()
    context_results = {}
    if not args.skip_context:
        log.info("Running context regressions")
        from analysis.models_context import run_context_regressions, context_coeff_table
        try:
            context_results = run_context_regressions(player_map, min_maps_context=args.min_maps)
            if context_results:
                coeff_df = context_coeff_table(context_results)
                coeff_df.to_csv(data_dir / "context_regression.csv", index=False)
        except Exception as exc:
            log.error("Context regressions failed: " + str(exc))

    rapm_scores = pd.DataFrame()
    shift_df = pd.DataFrame()
    player_ref = pd.DataFrame()
    if not args.skip_rapm:
        log.info("Running RAPM")
        from analysis.models_rapm import build_rapm_matrix, fit_rapm, rank_shift_analysis
        try:
            rapm_matrix, p_cols, player_ref = build_rapm_matrix(player_map, min_maps=args.min_maps_rapm)
            rapm_scores = fit_rapm(rapm_matrix, p_cols, player_ref)
            rapm_scores.to_csv(data_dir / "rapm_scores.csv", index=False)
            shift_df = rank_shift_analysis(rapm_scores, player_summary)
            shift_df.to_csv(data_dir / "rank_shift.csv", index=False)
        except Exception as exc:
            log.error("RAPM failed", exc_info=True)

    prediction_results = {}
    if not args.skip_prediction:
        log.info("Running predictions")
        from analysis.models_prediction import run_prediction_evaluation
        ref = player_ref if not player_ref.empty else player_map[["hltv_player_id", "label"]].drop_duplicates("hltv_player_id")
        try:
            prediction_results = run_prediction_evaluation(player_map, ref, cutoff_date=args.cutoff, min_maps_rapm=args.min_maps_rapm)
            prediction_results["metrics_table"].to_csv(data_dir / "prediction_metrics.csv", index=False)
            prediction_results["player_coef_table"].to_csv(data_dir / "prediction_player_coefs.csv", index=False)
            if "pair_coef_table" in prediction_results:
                prediction_results["pair_coef_table"].to_csv(data_dir / "prediction_pair_coefs.csv", index=False)
        except Exception as exc:
            log.error("Predictions failed", exc_info=True)

    if not args.skip_influence:
        log.info("Running teammate influence")
        from analysis.teammate_influence import main as ti_main
        ti_args = [
            "--db", args.db, "--data-dir", str(data_dir), "--report-dir", str(out_dir),
            "--min-maps", str(args.min_maps_rapm), "--min-pair-maps", str(args.min_pair_maps),
            "--placebo-iters", str(args.placebo_iters), "--n-jobs", str(args.n_jobs),
            "--pre-window", str(args.pre_window), "--post-window", str(args.post_window),
        ]
        try:
            ti_main(ti_args)
        except Exception as exc:
            log.error("Teammate influence failed: " + str(exc))

    log.info("Generating plots")
    from analysis import plots as P
    P.apply_style()

    try:
        if not rapm_scores.empty and not true_pm.empty:
            merged = rapm_scores.merge(
                true_pm[["hltv_player_id", "true_plus_minus", "pm_maps"]],
                on="hltv_player_id", how="left"
            ).dropna()
            P.plot_weighted_scatter(
                merged, "true_plus_minus", "rapm_score", "True PM vs RAPM",
                plot_dir / "plot_true_plus_minus_vs_rapm.png",
                size_col="pm_maps", label_col="label"
            )
    except Exception:
        log.warning("scatter plot failed")

    if not rapm_scores.empty:
        P.plot_coefficient_distribution(rapm_scores["rapm_score"], "RAPM Scores", plot_dir / "plot_dist_rapm.png")

    if not coeff_df.empty:
        P.plot_context_coefficients(coeff_df, plot_dir / "plot_context_coefficients.png")

    iti_path = data_dir / "indirect_teammate_impact.csv"
    if iti_path.exists():
        iti = pd.read_csv(iti_path)
        P.plot_iti_top_bottom(iti, plot_dir / "plot_iti_top_bottom.png", role_df=role_df)

    rc_path = data_dir / "roster_change_events.csv"
    if rc_path.exists():
        rc = pd.read_csv(rc_path)
        P.plot_roster_change_before_after(rc, plot_dir / "plot_roster_change_before_after.png")

    log.info("Writing reports")
    from analysis import report as R
    try:
        R.write_data_report(audit, out_dir)
        R.write_context_report(coeff_df, out_dir)
        R.write_rapm_report(rapm_scores, shift_df, out_dir, role_df=role_df)
        R.write_summary_report(audit, rapm_scores, shift_df, coeff_df, out_dir)
        if prediction_results:
            R.write_prediction_report(prediction_results, out_dir)
    except Exception as exc:
        log.error("Report writing failed: " + str(exc))

    print("\n--- ANALYSIS COMPLETE ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())