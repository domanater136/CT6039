import logging
import sqlite3

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# This is a nightmare. Do not look at the horrors
_RAW_QUERY = """
SELECT
    mp.id         AS row_id,
    mp.map_id,
    mp.match_id,
    ma.map_name,
    ma.map_order,
    ma.team1_rounds,
    ma.team2_rounds,
    ma.winner_team   AS map_winner_team,
    m.hltv_match_id,
    m.datetime_utc,
    m.event_name,
    m.stars,
    m.best_of,
    m.team1_name  AS match_team1,
    m.team2_name  AS match_team2,
    m.winner_team AS match_winner,
    mp.player_id,
    mp.hltv_player_id,
    p.hltv_slug,
    mp.player_name,
    mp.team_slot,
    mp.team_name,
    mp.opponent_team,
    mp.team_won,
    mp.kills,
    mp.deaths,
    mp.swing,
    mp.rating,
    mp.adr,
    mp.kd_ratio,
    mp.lineup_hash
FROM map_players mp
JOIN players  p  ON p.player_id  = mp.player_id
JOIN maps     ma ON ma.map_id    = mp.map_id
JOIN matches  m  ON m.match_id   = mp.match_id
ORDER BY m.datetime_utc, mp.map_id, mp.team_slot, mp.id
"""


def load_raw(conn):
    df = pd.read_sql_query(_RAW_QUERY, conn, parse_dates=["datetime_utc"])
    if df.empty:
        raise RuntimeError("No rows returned. need to scrape data first")

    n_matches = df["match_id"].nunique()
    n_maps = df["map_id"].nunique()
    n_players = df["hltv_player_id"].nunique()
    log.info(f"Loaded {len(df)} rows | {n_matches} matches | {n_maps} maps | {n_players} players")
    return df


def audit_raw(df):
    # check each map has exactly 10 players
    map_sizes = df.groupby("map_id").size()
    bad_not10 = map_sizes[map_sizes != 10].index.tolist()

    # check for missing values in the key stats columns
    key_cols = ["rating", "adr", "kills", "deaths", "team_won", "team1_rounds", "team2_rounds"]
    null_mask = df[key_cols].isna().any(axis=1)
    maps_with_nulls = df.loc[null_mask, "map_id"].unique().tolist()

    # a valid map needs at least 13 rounds played total
    round_check = df.groupby("map_id").agg(t1=("team1_rounds", "first"), t2=("team2_rounds", "first"))
    bad_rounds = round_check[(round_check["t1"] + round_check["t2"]) < 13].index.tolist()

    all_bad = set(bad_not10) | set(maps_with_nulls) | set(bad_rounds)

    audit = {
        "total_rows": len(df),
        "total_matches": df["match_id"].nunique(),
        "total_maps": df["map_id"].nunique(),
        "total_players": df["hltv_player_id"].nunique(),
        "maps_not_10_players": len(bad_not10),
        "maps_with_null_metrics": len(maps_with_nulls),
        "maps_bad_rounds": len(bad_rounds),
        "maps_excluded_total": len(all_bad),
        "maps_retained": df["map_id"].nunique() - len(all_bad),
        "bad_map_ids": sorted(list(all_bad)),
    }

    log.info(
        "Audit: {} maps total | excluded {} (not-10={}, nulls={}, bad-rounds={})".format(
            audit["total_maps"], audit["maps_excluded_total"],
            audit["maps_not_10_players"], audit["maps_with_null_metrics"], audit["maps_bad_rounds"]
        )
    )
    return audit


def build_player_map(df, bad_map_ids):
    clean = df[~df["map_id"].isin(bad_map_ids)].copy()

    # result_diff from each player's perspective (positive = their team won by that many)
    def get_result_diff(row):
        t1r = row["team1_rounds"]
        t2r = row["team2_rounds"]
        if pd.isna(t1r) or pd.isna(t2r):
            return np.nan
        t1r = int(t1r)
        t2r = int(t2r)
        if row["team_slot"] == "A":
            return t1r - t2r
        return t2r - t1r

    clean["result_diff"] = clean.apply(get_result_diff, axis=1)

    # log transform to reduce skew on K/D ratio
    clean["kd_log"] = np.log(
        (clean["kills"].clip(lower=0) + 1) / (clean["deaths"].clip(lower=1) + 1)
    )

    # label is slug if available, otherwise player_name, otherwise just the ID
    def get_label(row):
        for col in ["hltv_slug", "player_name"]:
            v = row.get(col)
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
        return str(int(row["hltv_player_id"]))

    clean["label"] = clean.apply(get_label, axis=1)

    if not pd.api.types.is_datetime64_any_dtype(clean["datetime_utc"]):
        clean["datetime_utc"] = pd.to_datetime(clean["datetime_utc"], errors="coerce")

    # need consistent ordering for the rolling features later
    clean = clean.sort_values(["datetime_utc", "map_id", "team_slot", "row_id"])

    log.info(f"player_map: {len(clean)} rows | {clean['map_id'].nunique()} maps | {clean['hltv_player_id'].nunique()} players")
    return clean


def build_team_map(player_map):
    team_agg = player_map.groupby(["map_id", "match_id", "team_name", "team_slot", "lineup_hash"]).agg(
        datetime_utc=("datetime_utc", "first"),
        map_name=("map_name", "first"),
        event_name=("event_name", "first"),
        opponent_team=("opponent_team", "first"),
        team_won=("team_won", "first"),
        result_diff=("result_diff", "first"),
        n_players=("hltv_player_id", "count"),
        avg_rating=("rating", "mean"),
        avg_adr=("adr", "mean"),
        avg_kd_log=("kd_log", "mean"),
    ).reset_index()

    # join opponent stats back in so we have context per team per map
    opp_stats = team_agg[["map_id", "team_name", "avg_rating", "avg_adr", "avg_kd_log"]].copy()
    opp_stats.columns = ["map_id", "opponent_team", "opp_avg_rating", "opp_avg_adr", "opp_avg_kd_log"]
    team_agg = team_agg.merge(opp_stats, on=["map_id", "opponent_team"], how="left")

    log.info(f"team_map: {len(team_agg)} rows | {team_agg['map_id'].nunique()} maps")
    return team_agg


def build_analysis_tables(conn):
    raw = load_raw(conn)
    audit = audit_raw(raw)
    player_map = build_player_map(raw, audit["bad_map_ids"])
    team_map = build_team_map(player_map)

    return {
        "raw": raw,
        "audit": audit,
        "player_map": player_map,
        "team_map": team_map,
    }