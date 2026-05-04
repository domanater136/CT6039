import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

log = logging.getLogger(__name__)

# tried with more alphas first (np.logspace(-3, 3, 50)) but it was overkill and barely changed results
# settled on these after a few runs
RAPM_ALPHAS = [0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 500, 700, 1000.0]


def build_rapm_matrix(player_map, min_maps=20, require_all_players_active=False, active_player_ids=None):
    # only use maps where all 10 players are present
    map_sizes = player_map.groupby("map_id").size()
    valid_map_ids = map_sizes[map_sizes == 10].index

    pm = player_map[player_map["map_id"].isin(valid_map_ids)].copy()

    if active_player_ids is None:
        appearances = pm.groupby("hltv_player_id")["map_id"].nunique()
        active_players = set(appearances[appearances >= min_maps].index)
    else:
        active_players = set(active_player_ids)

    log.info(f"RAPM: {len(active_players)} eligible players (min_maps={min_maps})")

    if len(active_players) < 10:
        raise RuntimeError("Not enough active players. try lowering --min-maps-rapm")

    rows = []
    for map_id, grp in pm.groupby("map_id"):
        team_a = grp[grp["team_slot"] == "A"]
        team_b = grp[grp["team_slot"] == "B"]

        # shouldn't happen after the size filter but just in case
        if len(team_a) != 5 or len(team_b) != 5:
            continue

        row = {
            "map_id": map_id,
            "match_id": int(grp["match_id"].iloc[0]),
            "datetime_utc": grp["datetime_utc"].iloc[0],
            "event_name": grp["event_name"].iloc[0],
            "map_name": grp["map_name"].iloc[0],
            "result_diff_a": int(team_a["result_diff"].iloc[0]),
            "team_a_won": int(team_a["team_won"].iloc[0]),
            "team_a_name": team_a["team_name"].iloc[0],
            "team_b_name": team_b["team_name"].iloc[0],
        }

        active_in_map = 0
        for _, prow in grp.iterrows():
            pid = int(prow["hltv_player_id"])
            if pid not in active_players:
                continue
            col = f"p_{pid}"
            row[col] = 1 if prow["team_slot"] == "A" else -1
            active_in_map += 1

        row["n_active_in_map"] = active_in_map

        if require_all_players_active and active_in_map != 10:
            continue

        rows.append(row)

    matrix = pd.DataFrame(rows)
    player_cols = sorted([c for c in matrix.columns if c.startswith("p_")])

    if not player_cols:
        raise RuntimeError("No player columns. something went wrong with the matrix build")

    matrix[player_cols] = matrix[player_cols].fillna(0).astype(np.int8)

    log.info(f"RAPM matrix: {len(matrix)} maps x {len(player_cols)} players")
    log.info("Active players per map. mean: {:.2f}".format(matrix["n_active_in_map"].mean()))

    player_ref = (
        player_map[["hltv_player_id", "hltv_slug", "player_name", "label"]]
        .drop_duplicates("hltv_player_id")
        .sort_values("hltv_player_id")
        .copy()
    )

    return matrix, player_cols, player_ref


def fit_rapm(matrix, player_cols, player_ref, alphas=None, cv=5):
    if alphas is None:
        alphas = RAPM_ALPHAS

    X = matrix[player_cols].to_numpy(dtype=float)
    y = matrix["result_diff_a"].to_numpy(dtype=float)

    # no intercept. The model is symmetric (team A vs team B sums to zero)
    model = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_squared_error", fit_intercept=False)
    model.fit(X, y)

    best_alpha = float(model.alpha_)
    log.info(f"Ridge CV selected alpha: {best_alpha}")

    scores = pd.DataFrame({
        "player_col": player_cols,
        "rapm_score": model.coef_,
    })

    scores["hltv_player_id"] = scores["player_col"].str.replace("p_", "", regex=False).astype(int)
    scores = scores.merge(
        player_ref[["hltv_player_id", "hltv_slug", "player_name", "label"]],
        on="hltv_player_id",
        how="left",
    )
    scores = scores.sort_values("rapm_score", ascending=False).reset_index(drop=True)

    scores.attrs["best_alpha"] = best_alpha
    scores.attrs["n_maps"] = len(matrix)

    return scores


def rank_shift_analysis(rapm_scores, player_summary):
    merged = rapm_scores.merge(
        player_summary[["hltv_player_id", "label", "maps", "avg_rating", "avg_adr", "avg_kd_log", "win_rate"]],
        on=["hltv_player_id", "label"],
        how="left",
    ).copy()

    merged["rapm_rank"] = merged["rapm_score"].rank(ascending=False, method="min")
    merged["rating_rank"] = merged["avg_rating"].rank(ascending=False, method="min")
    merged["adr_rank"] = merged["avg_adr"].rank(ascending=False, method="min")
    merged["kd_rank"] = merged["avg_kd_log"].rank(ascending=False, method="min")
    merged["winrate_rank"] = merged["win_rate"].rank(ascending=False, method="min")

    merged["delta_rating"] = merged["rating_rank"] - merged["rapm_rank"]
    merged["delta_adr"] = merged["adr_rank"] - merged["rapm_rank"]
    merged["delta_kd"] = merged["kd_rank"] - merged["rapm_rank"]

    # z-score each metric for the flag columns
    for col in ["rapm_score", "avg_rating", "avg_adr", "avg_kd_log"]:
        s = merged[col].std(ddof=0)
        if pd.isna(s) or s == 0:
            merged[col + "_z"] = 0.0
        else:
            merged[col + "_z"] = (merged[col] - merged[col].mean()) / s

    merged["flag_high_rating_low_rapm"] = (merged["avg_rating_z"] >= 1.0) & (merged["rapm_score_z"] <= -0.5)
    merged["flag_low_rating_high_rapm"] = (merged["avg_rating_z"] <= -0.5) & (merged["rapm_score_z"] >= 1.0)

    n_high = int(merged["flag_high_rating_low_rapm"].sum())
    n_low = int(merged["flag_low_rating_high_rapm"].sum())
    log.info(f"Rank shift. rating overestimates: {n_high} | underestimates: {n_low}")

    return merged.sort_values("rapm_rank")