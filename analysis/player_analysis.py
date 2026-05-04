import argparse
import logging
import sqlite3
import sys
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

VALID_ROLES = {"Rifler", "AWPer", "IGL", "Entry", "Lurker", "Support"}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="hltv_dissertation.db")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--min-maps", type=int, default=20)
    p.add_argument("--min-maps-rapm", type=int, default=50)
    p.add_argument("--cutoff", default="2025-07-01")
    p.add_argument("--min-pair-maps", type=int, default=20)
    p.add_argument("--placebo-iters", type=int, default=100)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--pre-window", type=int, default=30)
    p.add_argument("--post-window", type=int, default=30)
    p.add_argument("--skip-context", action="store_true")
    p.add_argument("--skip-rapm", action="store_true")
    p.add_argument("--skip-prediction", action="store_true")
    p.add_argument("--skip-influence", action="store_true")
    return p.parse_args()


def player_summary_from_map(player_map, min_maps):
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
    summary = summary[summary["maps"] >= min_maps].copy()
    summary["kill_diff"] = summary["total_kills"] - summary["total_deaths"]
    return summary.sort_values("avg_rating", ascending=False).reset_index(drop=True)


def true_plus_minus_from_map(player_map, min_maps):
    out = player_map.groupby("hltv_player_id").agg(
        true_plus_minus=("result_diff", "mean"),
        pm_maps=("map_id", "nunique"),
        avg_rating=("rating", "mean"),
        win_rate=("team_won", "mean"),
        label=("label", "first"),
    ).reset_index()
    out = out[out["pm_maps"] >= min_maps].copy()
    return out.sort_values("true_plus_minus", ascending=False).reset_index(drop=True)


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "analysis.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    print("Loading data from", args.db)
    from analysis.data import build_analysis_tables
    conn = sqlite3.connect(args.db)
    tables = build_analysis_tables(conn)
    conn.close()

    audit = tables["audit"]
    player_map = tables["player_map"]

    print("Building features...")
    from analysis.features import build_features
    player_map = build_features(player_map)

    if player_map["result_diff"].abs().max() > 20:
        n_bad = (player_map["result_diff"].abs() > 20).sum()
        log.warning(f"{n_bad} rows have result_diff > 20. check the data")

    player_summary = player_summary_from_map(player_map, args.min_maps)
    player_summary.to_csv(out_dir / "player_summary.csv", index=False)

    true_pm = true_plus_minus_from_map(player_map, args.min_maps_rapm)
    true_pm.to_csv(out_dir / "true_plus_minus.csv", index=False)

    coeff_df = pd.DataFrame()
    context_results = {}

    if not args.skip_context:
        print("Running context regressions...")
        from analysis.models_context import run_context_regressions, context_coeff_table
        try:
            context_results = run_context_regressions(player_map, min_maps_context=args.min_maps)
            if context_results:
                coeff_df = context_coeff_table(context_results)
        except Exception as exc:
            log.error("Context regressions failed: " + str(exc))

    rapm_scores = pd.DataFrame()
    shift_df = pd.DataFrame()
    player_ref = pd.DataFrame()

    if not args.skip_rapm:
        print("Running RAPM...")
        from analysis.models_rapm import build_rapm_matrix, fit_rapm, rank_shift_analysis
        try:
            rapm_matrix, player_cols, player_ref = build_rapm_matrix(player_map, min_maps=args.min_maps_rapm)
            rapm_scores = fit_rapm(rapm_matrix, player_cols, player_ref)
            rapm_scores.to_csv(out_dir / "rapm_scores.csv", index=False)
            shift_df = rank_shift_analysis(rapm_scores, player_summary)
        except Exception as exc:
            log.error("RAPM failed", exc_info=True)

    prediction_results = {}

    if not args.skip_prediction:
        print("Running predictions...")
        from analysis.models_prediction import run_prediction_evaluation
        try:
            ref = player_ref if not player_ref.empty else player_map[["hltv_player_id", "label"]].drop_duplicates("hltv_player_id")
            prediction_results = run_prediction_evaluation(
                player_map, ref,
                cutoff_date=args.cutoff,
                min_maps_rapm=args.min_maps_rapm,
            )
            prediction_results["metrics_table"].to_csv(out_dir / "prediction_metrics.csv", index=False)
            prediction_results["player_coef_table"].to_csv(out_dir / "prediction_player_coefs.csv", index=False)

            pair_coef_table = prediction_results.get("pair_coef_table", pd.DataFrame())
            if not pair_coef_table.empty:
                pair_coef_table.to_csv(out_dir / "prediction_pair_coefs.csv", index=False)

        except Exception as exc:
            log.error("Predictions failed: " + str(exc))

    if not args.skip_influence:
        print("Running teammate influence...")
        ti_out = out_dir / "teammate_influence"
        from analysis.teammate_influence import main as ti_main
        ti_args = [
            "--db", args.db,
            "--out-dir", str(ti_out),
            "--min-maps", str(args.min_maps_rapm),
            "--min-pair-maps", str(args.min_pair_maps),
            "--placebo-iters", str(args.placebo_iters),
            "--n-jobs", str(args.n_jobs),
            "--pre-window", str(args.pre_window),
            "--post-window", str(args.post_window),
        ]
        try:
            ti_main(ti_args)
        except Exception as exc:
            log.error("Teammate influence failed: " + str(exc))

    # --- plots ---
    print("Generating plots...")
    from analysis import plots as P
    P.apply_style()

    try:
        if not rapm_scores.empty and not true_pm.empty:
            rapm_eval = rapm_scores.merge(
                true_pm[["hltv_player_id", "true_plus_minus", "pm_maps"]],
                on="hltv_player_id", how="left"
            ).dropna(subset=["true_plus_minus"])

            P.plot_weighted_scatter(
                rapm_eval, "true_plus_minus", "rapm_score",
                "True Plus-Minus vs RAPM", out_dir / "plot_true_pm_vs_rapm.png",
                xlabel="True PM", ylabel="RAPM",
                size_col="pm_maps", label_col="label"
            )
    except Exception as exc:
        log.warning("scatter plot failed: " + str(exc))

    if not rapm_scores.empty:
        try:
            P.plot_coefficient_distribution(
                rapm_scores["rapm_score"], "RAPM Distribution", out_dir / "plot_dist_rapm.png"
            )
        except Exception:
            pass

    if prediction_results:
        pct = prediction_results.get("pair_coef_table", pd.DataFrame())
        if not pct.empty:
            P.plot_coefficient_distribution(
                pct["pair_coef"], "Pair Coef Distribution", out_dir / "plot_dist_pairs.png"
            )

    ti_out = out_dir / "teammate_influence"
    iti_csv = ti_out / "indirect_teammate_impact.csv"
    if iti_csv.exists():
        import pandas as pd
        iti_df = pd.read_csv(iti_csv)
        if not iti_df.empty:
            P.plot_iti_top_bottom(iti_df, ti_out / "plot_iti_top_bottom.png", n=15)

    roster_csv = ti_out / "roster_change_events.csv"
    if roster_csv.exists():
        rc_df = pd.read_csv(roster_csv)
        if not rc_df.empty:
            P.plot_roster_change_before_after(rc_df, ti_out / "plot_roster_change.png", n=10)

    # --- reports ---
    print("Writing reports...")
    from analysis import report as R
    R.write_data_report(audit, out_dir)
    R.write_context_report(coeff_df, out_dir)
    R.write_rapm_report(rapm_scores, shift_df, out_dir)
    R.write_summary_report(audit, rapm_scores, shift_df, coeff_df, out_dir)
    if prediction_results:
        R.write_prediction_report(prediction_results, out_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())