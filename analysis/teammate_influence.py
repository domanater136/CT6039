import argparse
import logging
import sqlite3
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

log = logging.getLogger("teammate_influence")


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")


def load_player_map(db_path):
    print("Loading player data from " + str(db_path))
    conn = sqlite3.connect(db_path)

    # joins the three tables we need. God forgive me for my sins which is this SQL statement
    sql = """
        SELECT
            mp.map_id, mp.match_id, mp.hltv_player_id, mp.player_name,
            mp.team_slot, mp.team_name, mp.opponent_team, mp.team_won,
            mp.kills, mp.deaths, mp.rating, mp.adr, mp.kd_ratio, mp.lineup_hash,
            ma.team1_rounds, ma.team2_rounds, ma.map_name,
            m.datetime_utc, m.event_name, m.stars
        FROM map_players mp
        JOIN maps ma ON ma.map_id = mp.map_id
        JOIN matches m ON m.match_id = mp.match_id
        WHERE mp.rating IS NOT NULL AND mp.adr IS NOT NULL
            AND mp.kills IS NOT NULL AND mp.deaths IS NOT NULL
        ORDER BY m.datetime_utc, mp.map_id, mp.team_slot
    """
    df = pd.read_sql_query(sql, conn, parse_dates=["datetime_utc"])
    conn.close()

    # only keep maps with exactly 10 players
    valid_maps = df.groupby("map_id").size().loc[lambda s: s == 10].index
    df = df[df["map_id"].isin(valid_maps)].copy()

    df["kd_log"] = np.log(df["kd_ratio"].clip(lower=0.01))
    df["result_diff"] = np.where(
        df["team_slot"] == "A",
        df["team1_rounds"] - df["team2_rounds"],
        df["team2_rounds"] - df["team1_rounds"]
    )
    df["label"] = df["player_name"].fillna(df["hltv_player_id"].astype(str))

    log.info(f"Loaded {len(df)} rows, {df['map_id'].nunique()} maps, {df['hltv_player_id'].nunique()} players")
    return df


def add_pre_map_features(player_map):
    log.info("Computing rolling pre-map features")
    pm = player_map.sort_values(["hltv_player_id", "datetime_utc", "map_id"]).copy()

    for metric, pre_col in [("rating", "pre_rating"), ("adr", "pre_adr"), ("kd_log", "pre_kd_log")]:
        pm[pre_col] = pm.groupby("hltv_player_id")[metric].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

    opp_agg = pm.groupby(["map_id", "team_slot"])["pre_rating"].mean().reset_index()
    opp_agg["opp_slot"] = opp_agg["team_slot"].map({"A": "B", "B": "A"})
    opp_agg = opp_agg.rename(columns={"pre_rating": "opp_pre_rating"})
    pm = pm.merge(
        opp_agg[["map_id", "opp_slot", "opp_pre_rating"]].rename(columns={"opp_slot": "team_slot"}),
        on=["map_id", "team_slot"], how="left"
    )

    pm["time_quarter"] = pd.to_datetime(pm["datetime_utc"], errors="coerce").dt.to_period("Q").astype(str)
    return pm


def compute_residuals(player_map, outcome):
    pre_col = f"pre_{outcome}" if f"pre_{outcome}" in player_map.columns else "pre_rating"
    needed = [outcome, pre_col, "opp_pre_rating", "map_name", "time_quarter", "hltv_player_id", "map_id"]
    sub = player_map[needed].dropna().copy()

    if len(sub) < 30:
        log.warning(f"Not enough rows for {outcome} residuals ({len(sub)}), skipping")
        pm = player_map.copy()
        pm[f"resid_{outcome}"] = np.nan
        return pm

    for col in [outcome, pre_col, "opp_pre_rating"]:
        sub[f"dm_{col}"] = sub[col] - sub.groupby("hltv_player_id")[col].transform("mean")

    map_d = pd.get_dummies(sub["map_name"], prefix="map", drop_first=True)
    time_d = pd.get_dummies(sub["time_quarter"], prefix="tq", drop_first=True)

    X = pd.concat([sub[[f"dm_{pre_col}", "dm_opp_pre_rating"]], map_d, time_d], axis=1).fillna(0).astype(float)
    y = sub[f"dm_{outcome}"].to_numpy(dtype=float)
    X_arr = np.column_stack([np.ones(len(X)), X.to_numpy()])
    coef, _, _, _ = np.linalg.lstsq(X_arr, y, rcond=None)
    sub[f"resid_{outcome}"] = y - X_arr @ coef

    pm = player_map.merge(sub[["hltv_player_id", "map_id", f"resid_{outcome}"]], on=["hltv_player_id", "map_id"], how="left")
    return pm


def build_slot_lookup(pm, active):
    sub = pm[pm["hltv_player_id"].isin(active)][["map_id", "hltv_player_id", "team_slot"]]
    out = {}
    for mid, pid, slot in zip(sub["map_id"], sub["hltv_player_id"], sub["team_slot"]):
        out.setdefault(int(mid), {})[int(pid)] = slot
    return out


def build_resid_lookup(pm, active, resid_col):
    sub = pm[pm["hltv_player_id"].isin(active) & pm[resid_col].notna()][["hltv_player_id", "map_id", resid_col]]
    out = {}
    for pid, mid, val in zip(sub["hltv_player_id"], sub["map_id"], sub[resid_col]):
        out.setdefault(int(pid), {})[int(mid)] = float(val)
    return out


def run_multi_outcome_uplift(player_map, active_player_ids, outcomes, min_pair_maps=20, p_threshold=0.05):
    log.info("Running multi-outcome uplift for " + str(outcomes))
    label_lookup = player_map[["hltv_player_id", "label"]].drop_duplicates("hltv_player_id").set_index("hltv_player_id")["label"].to_dict()
    slot_lookup = build_slot_lookup(player_map, active_player_ids)

    resid_lookups = {}
    for outcome in outcomes:
        resid_col = f"resid_{outcome}"
        if resid_col in player_map.columns:
            resid_lookups[outcome] = build_resid_lookup(player_map, active_player_ids, resid_col)

    if not resid_lookups:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    eligible = sorted(active_player_ids)

    for pid_a, pid_b in combinations(eligible, 2):
        for a, b in [(pid_a, pid_b), (pid_b, pid_a)]:
            for outcome, resid_lookup in resid_lookups.items():
                b_maps = resid_lookup.get(b, {})
                if not b_maps:
                    continue

                with_a, without_a = [], []
                for map_id, val in b_maps.items():
                    slots = slot_lookup.get(map_id, {})
                    sa, sb = slots.get(a), slots.get(b)
                    if sb is None:
                        continue
                    if sa == sb:
                        with_a.append(val)
                    else:
                        without_a.append(val)

                if len(with_a) < min_pair_maps or len(without_a) < min_pair_maps:
                    continue

                t_stat, p_value = stats.ttest_ind(with_a, without_a, equal_var=False)

                n1, n2 = len(with_a), len(without_a)
                s1, s2 = float(np.std(with_a, ddof=1)), float(np.std(without_a, ddof=1))
                pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                cohens_d = float((np.mean(with_a) - np.mean(without_a)) / pooled) if pooled > 0 else 0.0

                rows.append({
                    "player_a_id": a, "player_b_id": b,
                    "player_a": label_lookup.get(a, str(a)), "player_b": label_lookup.get(b, str(b)),
                    "outcome": outcome, "n_with": len(with_a), "n_without": len(without_a),
                    "mean_with": float(np.mean(with_a)), "mean_without": float(np.mean(without_a)),
                    "delta": float(np.mean(with_a) - np.mean(without_a)), "cohens_d": cohens_d,
                    "t_stat": float(t_stat), "p_value": float(p_value),
                    "significant": bool(p_value < p_threshold),
                })

    full_df = pd.DataFrame(rows)
    if full_df.empty:
        return full_df, pd.DataFrame()

    # Benjamini-Hochberg FDR correction
    m = len(full_df)
    sorted_idx = full_df["p_value"].argsort().values
    ranks = np.empty(m, dtype=int)
    ranks[sorted_idx] = np.arange(1, m + 1)

    fdr_q = 0.05
    bh_threshold = (ranks / m) * fdr_q
    sorted_pvals = full_df["p_value"].values[sorted_idx]
    bh_thresholds = (np.arange(1, m + 1) / m) * fdr_q
    below = sorted_pvals <= bh_thresholds
    bh_cutoff = sorted_pvals[int(np.where(below)[0].max())] if below.any() else 0.0

    full_df["bh_significant"] = full_df["p_value"] <= bh_cutoff
    full_df["bh_rank"] = ranks
    full_df["bh_threshold"] = bh_threshold

    sig_base = full_df[full_df["significant"]].groupby(
        ["player_a_id", "player_b_id", "player_a", "player_b"]
    ).agg(
        n_sig_outcomes=("outcome", "nunique"),
        outcomes_significant=("outcome", lambda s: ", ".join(sorted(s.unique()))),
        max_t=("t_stat", lambda s: s.loc[s.abs().idxmax()]),
        min_p=("p_value", "min"),
    ).reset_index()

    bh_counts = full_df[full_df["bh_significant"]].groupby(
        ["player_a_id", "player_b_id"]
    ).agg(n_bh_sig_outcomes=("outcome", "nunique")).reset_index()

    sig_base = sig_base.merge(bh_counts, on=["player_a_id", "player_b_id"], how="left")
    sig_base["n_bh_sig_outcomes"] = sig_base["n_bh_sig_outcomes"].fillna(0).astype(int)

    per_outcome_delta = full_df.pivot_table(
        index=["player_a_id", "player_b_id"], columns="outcome", values="delta", aggfunc="first"
    ).reset_index()
    per_outcome_delta.columns = ["player_a_id", "player_b_id"] + [f"delta_{c}" for c in per_outcome_delta.columns[2:]]

    sig_counts = sig_base.merge(per_outcome_delta, on=["player_a_id", "player_b_id"], how="left").sort_values(
        ["n_sig_outcomes", "max_t"], ascending=[False, False]
    ).reset_index(drop=True)

    return full_df, sig_counts


def run_indirect_teammate_impact(player_map, active_player_ids, resid_col="resid_rating",
                                  min_with=30, min_per_teammate_with=5, min_per_teammate_without=5,
                                  min_distinct_teammates=4):
    log.info("Computing ITI")
    label_lookup = player_map[["hltv_player_id", "label"]].drop_duplicates("hltv_player_id").set_index("hltv_player_id")["label"].to_dict()
    slot_lookup = build_slot_lookup(player_map, active_player_ids)
    resid_lookup = build_resid_lookup(player_map, active_player_ids, resid_col)

    rows = []
    for a in sorted(active_player_ids):
        a_maps = [m for m, slots in slot_lookup.items() if a in slots]
        if len(a_maps) < min_with:
            continue

        teammate_with = {}
        for mid in a_maps:
            slots = slot_lookup.get(mid, {})
            a_slot = slots.get(a)
            if a_slot is None:
                continue
            for pid, slot in slots.items():
                if pid == a or slot != a_slot:
                    continue
                resid = resid_lookup.get(pid, {}).get(mid)
                if resid is not None and not np.isnan(resid):
                    teammate_with.setdefault(pid, []).append(resid)

        per_pair = []
        for b, with_vals in teammate_with.items():
            if len(with_vals) < min_per_teammate_with:
                continue

            b_maps = resid_lookup.get(b, {})
            without_vals = [
                resid for mid, resid in b_maps.items()
                if a not in slot_lookup.get(mid, {}) and not np.isnan(resid)
            ]

            if len(without_vals) < min_per_teammate_without:
                continue

            per_pair.append({
                "teammate_b": b, "n_with": len(with_vals), "n_without": len(without_vals),
                "mean_with": float(np.mean(with_vals)), "mean_without": float(np.mean(without_vals)),
                "pair_delta": float(np.mean(with_vals) - np.mean(without_vals)),
            })

        if len(per_pair) < min_distinct_teammates:
            continue

        deltas = [r["pair_delta"] for r in per_pair]

        if len(deltas) >= 3:
            t_stat, p_value = stats.ttest_1samp(deltas, popmean=0.0)
            se = float(np.std(deltas, ddof=1)) / np.sqrt(len(deltas))
            t_crit = stats.t.ppf(0.975, df=len(deltas) - 1)
            ci_lo = float(np.mean(deltas)) - t_crit * se
            ci_hi = float(np.mean(deltas)) + t_crit * se
            cohen = float(np.mean(deltas) / np.std(deltas, ddof=1)) if np.std(deltas, ddof=1) > 0 else 0.0
        else:
            t_stat, p_value = np.nan, np.nan
            ci_lo = ci_hi = np.nan
            cohen = np.nan

        own_with = [
            resid_lookup.get(a, {}).get(mid) for mid in a_maps
            if resid_lookup.get(a, {}).get(mid) is not None and not np.isnan(resid_lookup.get(a, {}).get(mid))
        ]
        own_residual = float(np.mean(own_with)) if own_with else np.nan

        rows.append({
            "player_a_id": a, "player_a": label_lookup.get(a, str(a)),
            "n_maps_with_a": len(a_maps), "n_distinct_teammates": len(per_pair),
            "own_residual": own_residual, "iti": float(np.mean(deltas)),
            "iti_ci_lower": ci_lo, "iti_ci_upper": ci_hi,
            "iti_cohens_d": cohen,
            "iti_t_stat": float(t_stat) if not np.isnan(t_stat) else np.nan,
            "iti_p_value": float(p_value) if not np.isnan(p_value) else np.nan,
            "mean_teammate_with": float(np.mean([r["mean_with"] for r in per_pair])),
            "mean_teammate_without": float(np.mean([r["mean_without"] for r in per_pair])),
        })

    return pd.DataFrame(rows).sort_values("iti", ascending=False).reset_index(drop=True)


def run_roster_change_event_study(player_map, active_player_ids, resid_col="resid_rating",
                                   pre_window=30, post_window=30, min_retained=3):
    log.info(f"Roster-change event study (pre={pre_window}, post={post_window})")
    label_lookup = player_map[["hltv_player_id", "label"]].drop_duplicates("hltv_player_id").set_index("hltv_player_id")["label"].to_dict()
    pm = player_map.dropna(subset=[resid_col, "team_name", "lineup_hash"]).sort_values(["team_name", "datetime_utc", "map_id"])

    events = []

    for team_name, team_rows in pm.groupby("team_name"):
        team_maps = (
            team_rows.groupby("map_id")
            .agg(datetime_utc=("datetime_utc", "first"), lineup_hash=("lineup_hash", "first"))
            .reset_index()
            .sort_values(["datetime_utc", "map_id"])
            .reset_index(drop=True)
        )
        if len(team_maps) < pre_window + post_window:
            continue

        per_map_roster = {
            int(mid): set(grp["hltv_player_id"].astype(int).tolist())
            for mid, grp in team_rows.groupby("map_id")
        }

        prev_roster = None
        for idx, row in team_maps.iterrows():
            mid = int(row["map_id"])
            roster = per_map_roster.get(mid, set())

            if prev_roster is None or roster == prev_roster:
                prev_roster = roster
                continue

            new_joiners = roster - prev_roster
            retained = roster & prev_roster

            if len(retained) < min_retained or idx < pre_window or (len(team_maps) - idx) < post_window:
                prev_roster = roster
                continue

            pre_map_ids = team_maps.iloc[idx - pre_window: idx]["map_id"].astype(int).tolist()
            post_map_ids = team_maps.iloc[idx: idx + post_window]["map_id"].astype(int).tolist()

            for player_a in new_joiners:
                if player_a not in active_player_ids:
                    continue

                pre_resids, post_resids, per_teammate = [], [], []

                for b in retained:
                    b_pre = team_rows[
                        (team_rows["hltv_player_id"] == b) & (team_rows["map_id"].isin(pre_map_ids))
                    ][resid_col].dropna().tolist()
                    b_post = team_rows[
                        (team_rows["hltv_player_id"] == b) & (team_rows["map_id"].isin(post_map_ids))
                    ][resid_col].dropna().tolist()

                    if not b_pre or not b_post:
                        continue

                    pre_resids.extend(b_pre)
                    post_resids.extend(b_post)
                    per_teammate.append({
                        "teammate_b": int(b), "n_pre": len(b_pre), "n_post": len(b_post),
                        "mean_pre": float(np.mean(b_pre)), "mean_post": float(np.mean(b_post)),
                        "delta": float(np.mean(b_post) - np.mean(b_pre)),
                    })

                if not pre_resids or not post_resids or not per_teammate:
                    continue

                t_stat, p_value = stats.ttest_ind(post_resids, pre_resids, equal_var=False)
                events.append({
                    "team_name": team_name, "event_map_id": mid, "event_date": row["datetime_utc"],
                    "player_a_id": int(player_a), "player_a": label_lookup.get(int(player_a), str(player_a)),
                    "n_retained": len(per_teammate), "n_pre_obs": len(pre_resids), "n_post_obs": len(post_resids),
                    "mean_pre": float(np.mean(pre_resids)), "mean_post": float(np.mean(post_resids)),
                    "delta": float(np.mean(post_resids) - np.mean(pre_resids)),
                    "t_stat": float(t_stat), "p_value": float(p_value),
                    "retained_teammate_ids": ",".join(str(t["teammate_b"]) for t in per_teammate),
                })

            prev_roster = roster

    events_df = pd.DataFrame(events).sort_values("delta", ascending=False).reset_index(drop=True)
    if events_df.empty:
        return events_df, pd.DataFrame()

    summary = events_df.groupby(["player_a_id", "player_a"]).agg(
        n_events=("event_map_id", "count"),
        mean_delta=("delta", "mean"),
        median_delta=("delta", "median"),
        mean_t=("t_stat", "mean"),
        min_p=("p_value", "min"),
    ).reset_index().sort_values("mean_delta", ascending=False)

    events_df = (
        events_df.sort_values("delta", key=abs, ascending=False)
        .drop_duplicates(subset=["team_name", "event_map_id"], keep="first")
        .sort_values("delta", ascending=False)
        .reset_index(drop=True)
    )
    return events_df, summary


def placebo_single_iter(it, map_pid_slots, resid_lookup, pair_keys, min_pair_maps, seed):
    rng = np.random.default_rng(seed + it)
    shuffled = {}
    for mid, pids, slots in map_pid_slots:
        shuffled[mid] = dict(zip(pids, rng.permutation(slots).tolist()))

    rows = []
    for pid_a, pid_b in pair_keys:
        b_maps = resid_lookup.get(pid_b, {})
        if not b_maps:
            continue
        with_a, without_a = [], []
        for mid, val in b_maps.items():
            sl = shuffled.get(mid, {})
            sa, sb = sl.get(pid_a), sl.get(pid_b)
            if sb is None:
                continue
            if sa == sb:
                with_a.append(val)
            else:
                without_a.append(val)

        if len(with_a) < min_pair_maps or len(without_a) < min_pair_maps:
            continue

        t_stat, p_value = stats.ttest_ind(with_a, without_a, equal_var=False)
        rows.append({
            "iteration": it, "player_a_id": pid_a, "player_b_id": pid_b,
            "n_with": len(with_a), "n_without": len(without_a),
            "delta": float(np.mean(with_a) - np.mean(without_a)),
            "t_stat": float(t_stat), "p_value": float(p_value),
            "abs_t": float(abs(t_stat)), "significant": bool(p_value < 0.05),
        })
    return rows


def run_placebo_test(player_map, active_player_ids, real_uplift_df, outcome="rating",
                     min_pair_maps=20, n_iterations=100, seed=42, n_jobs=-1):
    try:
        from joblib import Parallel, delayed
        has_joblib = True
    except ImportError:
        has_joblib = False

    if real_uplift_df.empty:
        return pd.DataFrame(), {}

    resid_col = f"resid_{outcome}"
    real_for_outcome = real_uplift_df[real_uplift_df["outcome"] == outcome].copy()
    if real_for_outcome.empty:
        return pd.DataFrame(), {}

    pair_keys = list({(int(a), int(b)) for a, b in zip(real_for_outcome["player_a_id"], real_for_outcome["player_b_id"])})

    pm = player_map.dropna(subset=[resid_col])
    pm = pm[pm["hltv_player_id"].isin(active_player_ids)]

    resid_lookup = build_resid_lookup(pm, active_player_ids, resid_col)

    map_pid_slots = []
    for mid, grp in pm.groupby("map_id", sort=False):
        map_pid_slots.append((int(mid), grp["hltv_player_id"].astype(int).tolist(), grp["team_slot"].tolist()))

    log.info(f"Placebo: {n_iterations} iterations on {len(pair_keys)} pairs")

    if has_joblib and n_jobs != 1:
        batches = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
            delayed(placebo_single_iter)(it, map_pid_slots, resid_lookup, pair_keys, min_pair_maps, seed)
            for it in range(n_iterations)
        )
        placebo_rows = [r for batch in batches for r in batch]
    else:
        placebo_rows = []
        for it in range(n_iterations):
            placebo_rows.extend(placebo_single_iter(it, map_pid_slots, resid_lookup, pair_keys, min_pair_maps, seed))

    placebo_df = pd.DataFrame(placebo_rows)
    if placebo_df.empty:
        return placebo_df, {}

    real_abs_t = real_for_outcome["t_stat"].abs()
    placebo_abs_t = placebo_df["abs_t"]

    summary = {
        "outcome": outcome, "n_iterations": n_iterations,
        "n_real_pairs": int(len(real_for_outcome)),
        "n_placebo_observations": int(len(placebo_df)),
        "real_mean_abs_t": float(real_abs_t.mean()),
        "real_median_abs_t": float(real_abs_t.median()),
        "placebo_mean_abs_t": float(placebo_abs_t.mean()),
        "placebo_median_abs_t": float(placebo_abs_t.median()),
        "real_p95_abs_t": float(real_abs_t.quantile(0.95)),
        "placebo_p95_abs_t": float(placebo_abs_t.quantile(0.95)),
        "real_sig_rate": float((real_for_outcome["p_value"] < 0.05).mean()),
        "placebo_sig_rate": float(placebo_df["significant"].mean()),
        "ratio_real_to_placebo_mean_abs_t": float(real_abs_t.mean() / max(1e-9, placebo_abs_t.mean())),
    }

    return placebo_df, summary


def write_report(out_dir, multi_full, multi_summary, iti_df, events_df, events_summary, placebo_df, placebo_summary, config):
    out_path = out_dir / "report_teammate_influence.md"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Teammate Influence Analysis Report\n\n")
        f.write(f"_Generated: {pd.Timestamp.utcnow():%Y-%m-%d %H:%M UTC}_\n\n")

        f.write("## Configuration\n\n")
        f.write("| Parameter | Value |\n|---|---|\n")
        for k, v in config.items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n")

        # section 1: multi-outcome uplift
        f.write("## 1. Multi-Outcome Uplift\n\n")
        if multi_full.empty:
            f.write("_No testable pairs._\n\n")
        else:
            per_outcome = multi_full.groupby("outcome").agg(
                n_tests=("p_value", "count"),
                n_sig=("significant", "sum"),
                n_bh_sig=("bh_significant", "sum"),
            ).reset_index()
            per_outcome["share_sig"] = (per_outcome["n_sig"] / per_outcome["n_tests"]).round(3)
            per_outcome["share_bh_sig"] = (per_outcome["n_bh_sig"] / per_outcome["n_tests"]).round(3)

            f.write("### Per-outcome breakdown\n\n")
            f.write("| Outcome | Tests | Sig (raw p<0.05) | Share | Sig (BH q=0.05) | Share |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for _, r in per_outcome.iterrows():
                f.write("| {} | {:,} | {:,} | {:.3f} | {:,} | {:.3f} |\n".format(
                    r["outcome"], int(r["n_tests"]), int(r["n_sig"]),
                    r["share_sig"], int(r["n_bh_sig"]), r["share_bh_sig"]
                ))

            sig_counts_per_pair = multi_full.groupby(["player_a_id", "player_b_id"])["significant"].sum()
            bh_counts_per_pair = multi_full.groupby(["player_a_id", "player_b_id"])["bh_significant"].sum()
            total_pairs = len(sig_counts_per_pair)

            f.write("\n### Pairs by number of significant outcomes\n\n")
            f.write("| Outcomes significant | Raw (p<0.05) | FDR adjusted |\n|---|---|---|\n")
            for threshold, label in [(1, "At least 1"), (2, "At least 2"), (3, "All 3")]:
                raw = (sig_counts_per_pair >= threshold).sum()
                bh = (bh_counts_per_pair >= threshold).sum()
                f.write("| {} | {:,} ({:.1f}%) | {:,} ({:.1f}%) |\n".format(
                    label, raw, 100 * raw / total_pairs, bh, 100 * bh / total_pairs
                ))
            f.write("\n")

            if not multi_summary.empty:
                top = multi_summary[multi_summary["n_sig_outcomes"] >= 2].head(25)
                if not top.empty:
                    delta_cols = [c for c in top.columns if c.startswith("delta_")]
                    outcome_labels = [c.replace("delta_", "") for c in delta_cols]
                    header_metrics = " | ".join(f"delta_{o}" for o in outcome_labels)
                    f.write(f"| A | B | Sig metrics | {header_metrics} | Max |t| |\n")
                    f.write("|---|---|---:|" + "|".join("---:" for _ in delta_cols) + "|---:|\n")
                    for _, r in top.iterrows():
                        delta_vals = " | ".join(f"{r[c]:+.4f}" if pd.notna(r.get(c)) else "-" for c in delta_cols)
                        f.write(f"| {r['player_a']} | {r['player_b']} | {int(r['n_sig_outcomes'])} | {delta_vals} | {abs(r['max_t']):.2f} |\n")
                    f.write("\n")

        # section 2: ITI
        f.write("## 2. Indirect Teammate Impact (ITI)\n\n")
        if iti_df.empty:
            f.write("_Not enough data._\n\n")
        else:
            iti_valid = iti_df[
                (iti_df["n_distinct_teammates"] >= 4) &
                iti_df["iti_t_stat"].notna() &
                iti_df["iti_p_value"].notna()
            ].copy()

            # I'm going to be honest, this is the most cursed horrific thing I've ever had to attempt to make work.
            f.write("### Top 25 by ITI\n\n")
            f.write("| Player | ITI | 95% CI | Cohen d | t | p | Own resid | Maps | Teammates |\n")
            f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|\n")
            for _, r in iti_valid.head(25).iterrows():
                ci_str = "[{:+.4f}, {:+.4f}]".format(r["iti_ci_lower"], r["iti_ci_upper"]) if pd.notna(r.get("iti_ci_lower")) else "-"
                d_str = "{:+.3f}".format(r["iti_cohens_d"]) if pd.notna(r.get("iti_cohens_d")) else "-"
                f.write("| {} | {:+.4f} | {} | {} | {:+.2f} | {:.4f} | {:+.4f} | {} | {} |\n".format(
                    r["player_a"], r["iti"], ci_str, d_str,
                    r["iti_t_stat"], r["iti_p_value"], r["own_residual"],
                    int(r["n_maps_with_a"]), int(r["n_distinct_teammates"])
                ))

            f.write("\n### Bottom 25 by ITI\n\n")
            f.write("| Player | ITI | 95% CI | Cohen d | t | p | Own resid | Maps | Teammates |\n")
            f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|\n")
            for _, r in iti_valid.tail(25).iloc[::-1].iterrows():
                ci_str = "[{:+.4f}, {:+.4f}]".format(r["iti_ci_lower"], r["iti_ci_upper"]) if pd.notna(r.get("iti_ci_lower")) else "-"
                d_str = "{:+.3f}".format(r["iti_cohens_d"]) if pd.notna(r.get("iti_cohens_d")) else "—"
                f.write("| {} | {:+.4f} | {} | {} | {:+.2f} | {:.4f} | {:+.4f} | {} | {} |\n".format(
                    r["player_a"], r["iti"], ci_str, d_str,
                    r["iti_t_stat"], r["iti_p_value"], r["own_residual"],
                    int(r["n_maps_with_a"]), int(r["n_distinct_teammates"])
                ))
            f.write("\n")

        # section 3: roster changes
        f.write("## 3. Roster-Change Event Study\n\n")
        if events_df.empty:
            f.write("_No qualifying events found._\n\n")
        else:
            f.write("### Top 25 join events by retained-teammate uplift\n\n")
            f.write("| Date | Team | Joining player | Retained | n_pre | n_post | mean_pre | mean_post | delta | p |\n")
            f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for _, r in events_df.head(25).iterrows():
                date_str = pd.to_datetime(r["event_date"]).strftime("%Y-%m-%d") if pd.notna(r["event_date"]) else "?"
                f.write("| {} | {} | {} | {} | {} | {} | {:+.4f} | {:+.4f} | {:+.4f} | {:.4f} |\n".format(
                    date_str, r["team_name"], r["player_a"],
                    int(r["n_retained"]), int(r["n_pre_obs"]), int(r["n_post_obs"]),
                    r["mean_pre"], r["mean_post"], r["delta"], r["p_value"]
                ))

            f.write("\n### Bottom 25 join events\n\n")
            f.write("| Date | Team | Joining player | Retained | n_pre | n_post | mean_pre | mean_post | delta | p |\n")
            f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for _, r in events_df.tail(25).iloc[::-1].iterrows():
                date_str = pd.to_datetime(r["event_date"]).strftime("%Y-%m-%d") if pd.notna(r["event_date"]) else "?"
                f.write("| {} | {} | {} | {} | {} | {} | {:+.4f} | {:+.4f} | {:+.4f} | {:.4f} |\n".format(
                    date_str, r["team_name"], r["player_a"],
                    int(r["n_retained"]), int(r["n_pre_obs"]), int(r["n_post_obs"]),
                    r["mean_pre"], r["mean_post"], r["delta"], r["p_value"]
                ))
            f.write("\n")

        # section 4: placebo
        f.write("## 4. Placebo Test\n\n")
        if not placebo_summary:
            f.write("_Not run._\n\n")
        else:
            s = placebo_summary
            f.write("### Real vs placebo\n\n")
            f.write("| Statistic | Real | Placebo | Ratio |\n|---|---:|---:|---:|\n")
            f.write("| Mean |t| | {:.3f} | {:.3f} | {:.2f}x |\n".format(
                s["real_mean_abs_t"], s["placebo_mean_abs_t"], s["ratio_real_to_placebo_mean_abs_t"]
            ))
            f.write("| Median |t| | {:.3f} | {:.3f} | - |\n".format(
                s["real_median_abs_t"], s["placebo_median_abs_t"]
            ))
            f.write("| Sig rate (p<0.05) | {:.3f} | {:.3f} | - |\n\n".format(
                s["real_sig_rate"], s["placebo_sig_rate"]
            ))

    log.info("Report written to " + str(out_path))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="hltv_dissertation.db")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--report-dir", default="outputs")
    p.add_argument("--min-maps", type=int, default=50)
    p.add_argument("--min-pair-maps", type=int, default=20)
    p.add_argument("--p-threshold", type=float, default=0.05)
    p.add_argument("--pre-window", type=int, default=10)
    p.add_argument("--post-window", type=int, default=10)
    p.add_argument("--min-retained", type=int, default=3)
    p.add_argument("--placebo-iters", type=int, default=100)
    p.add_argument("--placebo-outcome", default="rating")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-multi-outcome", action="store_true")
    p.add_argument("--skip-iti", action="store_true")
    p.add_argument("--skip-event-study", action="store_true")
    p.add_argument("--skip-placebo", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    setup_logging(args.verbose)

    db_path = Path(args.db)
    if not db_path.exists():
        log.error("Database not found: " + str(db_path))
        return 1

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    pm = load_player_map(db_path)
    pm = add_pre_map_features(pm)

    print("Computing residuals...")
    for outcome in ("rating", "adr", "kd_log"):
        pm = compute_residuals(pm, outcome)

    appearances = pm.groupby("hltv_player_id")["map_id"].nunique()
    active = set(appearances[appearances >= args.min_maps].index)
    log.info(f"Active players: {len(active)}")

    if len(active) < 10:
        log.error("Not enough active players")
        return 2

    config = {
        "db_path": str(db_path), "data_dir": str(data_dir), "report_dir": str(report_dir),
        "min_maps": args.min_maps, "min_pair_maps": args.min_pair_maps,
        "p_threshold": args.p_threshold, "pre_window": args.pre_window,
        "post_window": args.post_window, "min_retained": args.min_retained,
        "placebo_iters": args.placebo_iters, "n_active_players": len(active),
        "n_total_maps": pm["map_id"].nunique(),
    }

    if args.skip_multi_outcome:
        multi_full, multi_summary = pd.DataFrame(), pd.DataFrame()
    else:
        print("Running pairwise uplift...")
        multi_full, multi_summary = run_multi_outcome_uplift(
            pm, active, outcomes=["rating", "adr", "kd_log"],
            min_pair_maps=args.min_pair_maps, p_threshold=args.p_threshold
        )
        if not multi_full.empty:
            multi_full.to_csv(data_dir / "multi_outcome_uplift.csv", index=False)
        if not multi_summary.empty:
            multi_summary.to_csv(data_dir / "multi_outcome_summary.csv", index=False)

    if args.skip_iti:
        iti_df = pd.DataFrame()
    else:
        print("Computing ITI...")
        iti_df = run_indirect_teammate_impact(pm, active)
        if not iti_df.empty:
            iti_df.to_csv(data_dir / "indirect_teammate_impact.csv", index=False)

    if args.skip_event_study:
        events_df, events_summary = pd.DataFrame(), pd.DataFrame()
    else:
        print("Running roster-change event study...")
        events_df, events_summary = run_roster_change_event_study(
            pm, active, pre_window=args.pre_window,
            post_window=args.post_window, min_retained=args.min_retained
        )
        if not events_df.empty:
            events_df.to_csv(data_dir / "roster_change_events.csv", index=False)
        if not events_summary.empty:
            events_summary.to_csv(data_dir / "roster_change_summary.csv", index=False)

    if args.skip_placebo or multi_full.empty:
        placebo_df, placebo_summary = pd.DataFrame(), {}
    else:
        print("Running placebo test...")
        placebo_df, placebo_summary = run_placebo_test(
            pm, active, multi_full, outcome=args.placebo_outcome,
            min_pair_maps=args.min_pair_maps, n_iterations=args.placebo_iters,
            seed=args.seed, n_jobs=args.n_jobs
        )
        if not placebo_df.empty:
            placebo_df.to_csv(data_dir / "placebo_distribution.csv", index=False)
        if placebo_summary:
            pd.DataFrame([placebo_summary]).to_csv(data_dir / "placebo_summary.csv", index=False)

    write_report(report_dir, multi_full, multi_summary, iti_df, events_df, events_summary, placebo_df, placebo_summary, config)
    print("Done.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())