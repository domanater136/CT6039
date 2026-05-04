import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def rolling_player_avg(player_map, metric, new_col, min_periods=1):
    """
    For each player, compute their average of `metric` across all maps
    BEFORE the current one. This prevents any forward-looking bias.
    """
    tmp = player_map[["hltv_player_id", "datetime_utc", "map_id", metric]].copy()
    tmp = tmp.sort_values(["hltv_player_id", "datetime_utc", "map_id"])

    # shift(1) so the current map's value is excluded from its own average
    expanding = tmp.groupby("hltv_player_id")[metric].transform(
        lambda s: s.shift(1).expanding(min_periods=min_periods).mean()
    )

    tmp[new_col] = expanding.values
    result = player_map[["hltv_player_id", "datetime_utc", "map_id"]].copy()
    result[new_col] = tmp[new_col].values
    return result[new_col]


def teammate_avg_excl_self(player_map, pre_col, out_col):
    """
    Average pre-map metric for teammates on the same slot, excluding the player themselves.
    Had to do the sum-then-subtract approach because groupby doesn't let you exclude self easily.
    """
    tmp = player_map[["hltv_player_id", "map_id", "team_slot", pre_col]].copy()

    team_totals = tmp.groupby(["map_id", "team_slot"])[pre_col].agg(["sum", "count"])
    team_totals = team_totals.rename(columns={"sum": "team_sum", "count": "team_n"}).reset_index()

    merged = player_map[["hltv_player_id", "map_id", "team_slot"]].copy()
    merged = merged.merge(team_totals, on=["map_id", "team_slot"], how="left")
    merged = merged.merge(
        tmp.rename(columns={pre_col: "self_val"}),
        on=["hltv_player_id", "map_id"],
        how="left",
    )

    # subtract self from team total, then divide by remaining teammates (4)
    tm_sum = merged["team_sum"] - merged["self_val"].fillna(0)
    tm_n = (merged["team_n"] - 1).clip(lower=1)

    result = (tm_sum / tm_n).rename(out_col)
    result.index = player_map.index
    return result


def opponent_avg(player_map, pre_col, out_col):
    """Average pre-map metric of the opposing team."""
    tmp = player_map[["map_id", "team_slot", pre_col]].copy()

    # flip the slot so we can join on opponent's slot
    opp_slot_map = {"A": "B", "B": "A"}
    tmp["opp_slot"] = tmp["team_slot"].map(opp_slot_map)

    opp_agg = tmp.groupby(["map_id", "opp_slot"])[pre_col].mean().reset_index()
    opp_agg = opp_agg.rename(columns={"opp_slot": "team_slot", pre_col: out_col})

    merged = player_map[["map_id", "team_slot"]].copy()
    merged = merged.merge(opp_agg, on=["map_id", "team_slot"], how="left")

    result = merged[out_col]
    result.index = player_map.index
    return result


def compute_lineup_age(player_map):
    """How many maps this specific 5-player lineup has played together before this map."""
    lh = player_map.groupby(["lineup_hash", "map_id"])["datetime_utc"].first().reset_index()
    lh = lh.sort_values(["lineup_hash", "datetime_utc", "map_id"])
    lh["lineup_age"] = lh.groupby("lineup_hash").cumcount()

    merged = player_map[["lineup_hash", "map_id"]].copy()
    merged = merged.merge(lh[["lineup_hash", "map_id", "lineup_age"]], on=["lineup_hash", "map_id"], how="left")

    result = merged["lineup_age"]
    result.index = player_map.index
    return result


def build_features(player_map):
    pm = player_map.copy()

    log.info("Computing rolling pre-map averages")

    # own historical averages (excludes current map)
    pm["pre_rating"] = rolling_player_avg(pm, "rating", "pre_rating")
    pm["pre_adr"] = rolling_player_avg(pm, "adr", "pre_adr")
    pm["pre_kd_log"] = rolling_player_avg(pm, "kd_log", "pre_kd_log")

    # teammate and opponent context features
    pm["tm_pre_rating"] = teammate_avg_excl_self(pm, "pre_rating", "tm_pre_rating")
    pm["opp_pre_rating"] = opponent_avg(pm, "pre_rating", "opp_pre_rating")

    pm["tm_pre_adr"] = teammate_avg_excl_self(pm, "pre_adr", "tm_pre_adr")
    pm["opp_pre_adr"] = opponent_avg(pm, "pre_adr", "opp_pre_adr")

    pm["lineup_age"] = compute_lineup_age(pm)

    # quarter dummy so the model can pick up meta changes across time
    dt = pd.to_datetime(pm["datetime_utc"], errors="coerce")
    pm["time_quarter"] = dt.dt.to_period("Q").astype(str)

    non_null = pm["pre_rating"].notna().sum()
    log.info(f"Features done | pre_rating non-null: {non_null}/{len(pm)} | tm_pre_rating non-null: {pm['tm_pre_rating'].notna().sum()}")

    return pm