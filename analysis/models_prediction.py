import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import RidgeCV

from analysis.models_rapm import build_rapm_matrix

log = logging.getLogger(__name__)

# same alphas as the main RAPM. could probably trim this down to a specific subset but just in case
PRED_ALPHAS = (0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def build_prediction_matrix(player_map, active_player_ids):
    """Build the design matrix for a given set of maps and active players."""
    rows = []
    for map_id, grp in player_map.groupby("map_id"):
        team_a = grp[grp["team_slot"] == "A"]
        team_b = grp[grp["team_slot"] == "B"]
        if len(team_a) != 5 or len(team_b) != 5:
            continue

        row = {
            "map_id": map_id,
            "datetime_utc": grp["datetime_utc"].iloc[0],
            "result_diff_a": int(team_a["result_diff"].iloc[0]),
            "team_a_won": int(team_a["team_won"].iloc[0]),
            "team_a_name": team_a["team_name"].iloc[0],
            "team_b_name": team_b["team_name"].iloc[0],
        }
        for _, prow in grp.iterrows():
            pid = int(prow["hltv_player_id"])
            if pid not in active_player_ids:
                continue
            row[f"p_{pid}"] = 1 if prow["team_slot"] == "A" else -1

        rows.append(row)

    matrix = pd.DataFrame(rows)
    player_cols = sorted([c for c in matrix.columns if c.startswith("p_")])
    matrix[player_cols] = matrix[player_cols].fillna(0).astype(np.int8)
    return matrix


def fit_individual_rapm(matrix, player_cols, alphas=None, cv=5):
    if alphas is None:
        alphas = PRED_ALPHAS
    X = matrix[player_cols].to_numpy(dtype=float)
    y = matrix["result_diff_a"].to_numpy(dtype=float)
    model = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_squared_error")
    model.fit(X, y)
    log.info(f"Individual RAPM (train): alpha={model.alpha_:.4f}")
    return model.coef_, float(model.alpha_)


def compute_rapm_residuals(player_map, player_cols, individual_coefs):
    coef_map = dict(zip(player_cols, individual_coefs))
    map_pids = player_map.groupby("map_id")["hltv_player_id"].apply(list)
    map_slots = player_map.groupby("map_id")["team_slot"].apply(list)

    rapm_pred = {}
    for map_id in map_pids.index:
        pred = 0
        for pid, slot in zip(map_pids[map_id], map_slots[map_id]):
            if slot == "A":
                pred += coef_map.get(f"p_{pid}", 0.0)
            else:
                pred -= coef_map.get(f"p_{pid}", 0.0)
        rapm_pred[map_id] = pred

    pm = player_map.copy()
    pm["rapm_pred"] = pm["map_id"].map(rapm_pred)
    pm["rapm_resid"] = pm["result_diff"] - pm["rapm_pred"]
    return pm


def uplift_on_rapm_residuals(player_map_train, active_player_ids, min_pair_maps=5):
    """Run pairwise t-tests on RAPM residuals to find pairs with systematic effects."""
    from scipy import stats as sp_stats

    resid_col = "rapm_resid"
    eligible = sorted(list(active_player_ids))

    # build slot and residual lookups for fast iteration
    slot_lookup = {}
    for _, row in player_map_train[player_map_train["hltv_player_id"].isin(active_player_ids)].iterrows():
        mid = int(row["map_id"])
        if mid not in slot_lookup:
            slot_lookup[mid] = {}
        slot_lookup[mid][int(row["hltv_player_id"])] = row["team_slot"]

    resid_lookup = {}
    valid_rows = player_map_train[
        player_map_train["hltv_player_id"].isin(active_player_ids) & player_map_train[resid_col].notna()
    ]
    for _, row in valid_rows.iterrows():
        pid = int(row["hltv_player_id"])
        if pid not in resid_lookup:
            resid_lookup[pid] = {}
        resid_lookup[pid][int(row["map_id"])] = float(row[resid_col])

    log.info(f"RAPM-residual uplift: {len(eligible)} candidates")

    def test_pair(pid_a, pid_b):
        b_maps = resid_lookup.get(pid_b, {})
        if not b_maps:
            return None

        with_a = []
        without_a = []
        for mid, resid in b_maps.items():
            slots = slot_lookup.get(mid, {})
            sa = slots.get(pid_a)
            sb = slots.get(pid_b)
            if sb is None:
                continue
            if sa is not None and sa == sb:
                with_a.append(resid)
            else:
                without_a.append(resid)

        if len(with_a) < min_pair_maps or len(without_a) < min_pair_maps:
            return None

        t, p = sp_stats.ttest_ind(with_a, without_a, equal_var=False)
        return {
            "player_a_id": pid_a, "player_b_id": pid_b,
            "delta": float(np.mean(with_a) - np.mean(without_a)),
            "t_stat": float(t), "p_value": float(p),
            "n_with": len(with_a), "n_without": len(without_a),
        }

    records = []
    for pid_a, pid_b in combinations(eligible, 2):
        for a, b in [(pid_a, pid_b), (pid_b, pid_a)]:
            r = test_pair(a, b)
            if r:
                records.append(r)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values("t_stat", ascending=False).reset_index(drop=True)


def select_joint_pairs(uplift_pairs, min_n_with=15, min_mean_t=1.0, top_k=500):
    if uplift_pairs.empty:
        return []

    df = uplift_pairs.copy()
    df["uid_a"] = df[["player_a_id", "player_b_id"]].min(axis=1).astype(int)
    df["uid_b"] = df[["player_a_id", "player_b_id"]].max(axis=1).astype(int)
    df["abs_t"] = df["t_stat"].abs()

    agg = df.groupby(["uid_a", "uid_b"]).agg(
        mean_abs_t=("abs_t", "mean"),
        n_directions=("abs_t", "count"),
        mean_n_with=("n_with", "mean"),
    ).reset_index()

    agg = agg[
        (agg["n_directions"] == 2) &
        (agg["mean_n_with"] >= min_n_with) &
        (agg["mean_abs_t"] >= min_mean_t)
    ].copy()

    agg["score"] = agg["mean_n_with"] * agg["mean_abs_t"]
    agg = agg.sort_values("score", ascending=False).head(top_k)

    selected = list(zip(agg["uid_a"].astype(int), agg["uid_b"].astype(int)))
    log.info(f"Selected {len(selected)} pairs for joint model")
    return selected


def build_pair_columns(player_map, selected_pairs, map_ids_order):
    if not selected_pairs:
        return pd.DataFrame(index=range(len(map_ids_order)))

    pair_pids = set()
    for a, b in selected_pairs:
        pair_pids.add(a)
        pair_pids.add(b)

    pid_map_slot = {}
    valid_rows = player_map[player_map["hltv_player_id"].isin(pair_pids)]
    for _, row in valid_rows.iterrows():
        pid = int(row["hltv_player_id"])
        mid = int(row["map_id"])
        if pid not in pid_map_slot:
            pid_map_slot[pid] = {}
        pid_map_slot[pid][mid] = row["team_slot"]

    map_id_to_idx = {mid: i for i, mid in enumerate(map_ids_order)}
    n_maps = len(map_ids_order)

    pair_arrays = {}
    for pid_a, pid_b in selected_pairs:
        col_name = f"pr_{pid_a}_{pid_b}"
        col = np.zeros(n_maps, dtype=np.int8)
        maps_a = pid_map_slot.get(pid_a, {})
        maps_b = pid_map_slot.get(pid_b, {})

        for mid in set(maps_a.keys()) & set(maps_b.keys()):
            idx = map_id_to_idx.get(mid)
            if idx is None:
                continue
            slot_a = maps_a[mid]
            slot_b = maps_b[mid]
            if slot_a == slot_b:
                col[idx] = 1 if slot_a == "A" else -1

        pair_arrays[col_name] = col

    return pd.DataFrame(pair_arrays)


def fit_joint_rapm(train_matrix, individual_cols, pair_col_df, cv=5):
    X_ind = train_matrix[individual_cols].to_numpy(dtype=float)
    X_pair = pair_col_df.to_numpy(dtype=float)
    X = np.hstack([X_ind, X_pair])
    y = train_matrix["result_diff_a"].to_numpy(dtype=float)

    alphas = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 500.0, 1000.0)
    model = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_squared_error")
    model.fit(X, y)

    n_ind = len(individual_cols)
    ind_coefs = model.coef_[:n_ind]
    pair_coefs = model.coef_[n_ind:]

    # log how many pairs actually have non-trivial coefficients
    n_nonzero = int((np.abs(pair_coefs) >= 0.01).sum())
    log.info(f"Joint RAPM alpha={model.alpha_:.1f} | non-trivial pair coefs: {n_nonzero}/{len(pair_coefs)}")

    return ind_coefs, pair_coefs, float(model.alpha_)


def predict_from_matrix(matrix, player_cols, coefs):
    return matrix[player_cols].to_numpy(dtype=float) @ coefs


def predict_joint(matrix, individual_cols, pair_col_df, individual_coefs, pair_coefs):
    X_ind = matrix[individual_cols].to_numpy(dtype=float)
    X_pair = pair_col_df.to_numpy(dtype=float)
    return np.hstack([X_ind, X_pair]) @ np.concatenate([individual_coefs, pair_coefs])


def compute_metrics(actuals, preds, actual_wins):
    mae = float(np.mean(np.abs(actuals - preds)))
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    acc = float(np.mean((preds > 0).astype(int) == actual_wins))
    return {"mae": mae, "rmse": rmse, "win_loss_accuracy": acc}


def mcnemar_test(actual_wins, pred_a, pred_b):
    correct_a = (pred_a > 0).astype(int) == actual_wins
    correct_b = (pred_b > 0).astype(int) == actual_wins
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c

    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "chi2_stat": 0.0, "p_value_asymp": 1.0, "p_value_midp": 1.0, "significant": False, "interpretation": "No discordant cases."}

    chi2_stat = (abs(b - c) - 1) ** 2 / n
    p_asymp = float(1 - chi2.cdf(chi2_stat, df=1))

    from scipy.stats import binom
    p_midp = float(2 * binom.cdf(min(b, c), n, 0.5) - binom.pmf(min(b, c), n, 0.5))
    sig = p_midp < 0.05
    winner = "Model 2" if c > b else "Model 1"

    return {
        "b": b, "c": c, "n_discordant": n,
        "chi2_stat": round(chi2_stat, 4),
        "p_value_asymp": round(p_asymp, 4),
        "p_value_midp": round(p_midp, 4),
        "significant": sig,
        "interpretation": f"{winner} wins more cases. Mid-p={p_midp:.4f}",
    }


def bootstrap_improvement_ci(actuals, actual_wins, pred_base, pred_comp, n_bootstrap=1000, seed=42, ci=0.95):
    rng = np.random.default_rng(seed)
    n = len(actuals)
    mae_d, rmse_d, acc_d = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a, aw, pb, pc = actuals[idx], actual_wins[idx], pred_base[idx], pred_comp[idx]
        mae_d.append(np.mean(np.abs(a - pb)) - np.mean(np.abs(a - pc)))
        rmse_d.append(np.sqrt(np.mean((a - pb) ** 2)) - np.sqrt(np.mean((a - pc) ** 2)))
        acc_d.append(np.mean((pc > 0).astype(int) == aw) - np.mean((pb > 0).astype(int) == aw))

    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2

    def ci_for(arr):
        obs = float(np.mean(arr))
        l_, h_ = float(np.quantile(arr, lo)), float(np.quantile(arr, hi))
        excludes_zero = not (l_ <= 0 <= h_)
        return {"observed": round(obs, 4), "ci_lower": round(l_, 4), "ci_upper": round(h_, 4), "excludes_zero": excludes_zero, "ci_level": ci}

    return {
        "mae": ci_for(mae_d), "rmse": ci_for(rmse_d), "accuracy": ci_for(acc_d),
        "raw": {"mae": mae_d, "rmse": rmse_d, "accuracy": acc_d}
    }


def compute_pair_survival(train_pair_col_df, test_pair_col_df, pair_coef_table):
    records = []
    for col in train_pair_col_df.columns:
        train_maps = int((train_pair_col_df[col] != 0).sum())
        test_maps = int((test_pair_col_df[col] != 0).sum()) if col in test_pair_col_df.columns else 0
        records.append({
            "pair_col": col,
            "train_maps": train_maps,
            "test_maps": test_maps,
            "appeared_in_test": test_maps > 0,
        })

    survival_df = pd.DataFrame(records)
    if not pair_coef_table.empty:
        survival_df = survival_df.merge(
            pair_coef_table[["pair_col", "pair_coef", "abs_pair_coef", "player_a_label", "player_b_label"]],
            on="pair_col", how="left"
        )
    return survival_df


def run_prediction_evaluation(player_map, player_ref, cutoff_date="2025-07-01", min_maps_rapm=50,
                               alphas=None, cv=5):
    if alphas is None:
        alphas = PRED_ALPHAS

    cutoff = pd.Timestamp(cutoff_date, tz="UTC")
    pm = player_map.copy()
    pm["datetime_utc"] = pd.to_datetime(pm["datetime_utc"], utc=True, errors="coerce")

    map_dates = pm.groupby("map_id")["datetime_utc"].first().reset_index().rename(columns={"datetime_utc": "map_date"})
    train_ids = set(map_dates[map_dates["map_date"] < cutoff]["map_id"])
    test_ids = set(map_dates[map_dates["map_date"] >= cutoff]["map_id"])

    pm_train = pm[pm["map_id"].isin(train_ids)].copy()
    pm_test = pm[pm["map_id"].isin(test_ids)].copy()

    log.info(f"Split at {cutoff_date} | train={pm_train['map_id'].nunique()} test={pm_test['map_id'].nunique()}")

    if pm_train["map_id"].nunique() < 50:
        raise RuntimeError("Too few training maps")
    if pm_test["map_id"].nunique() < 20:
        raise RuntimeError("Too few test maps")

    # only include players with enough training maps
    train_app = pm_train.groupby("hltv_player_id")["map_id"].nunique()
    active = set(train_app[train_app >= min_maps_rapm].index)
    if len(active) < 10:
        raise RuntimeError("Not enough active players")

    train_matrix, player_cols, _ = build_rapm_matrix(
        pm_train, min_maps=min_maps_rapm, require_all_players_active=False, active_player_ids=active
    )

    ind_coefs, alpha_m1 = fit_individual_rapm(train_matrix, player_cols, alphas, cv)

    pm_train_resid = compute_rapm_residuals(pm_train, player_cols, ind_coefs)
    uplift_pairs = uplift_on_rapm_residuals(pm_train_resid, active, min_pair_maps=8)

    selected_pairs = select_joint_pairs(uplift_pairs, min_n_with=15, min_mean_t=1.0, top_k=500)
    train_pair_col_df = build_pair_columns(pm_train, selected_pairs, list(train_matrix["map_id"]))

    joint_ind_coefs, joint_pair_coefs, alpha_m2 = fit_joint_rapm(train_matrix, player_cols, train_pair_col_df, cv=cv)

    test_matrix, _, _ = build_rapm_matrix(
        pm_test, min_maps=min_maps_rapm, require_all_players_active=False, active_player_ids=active
    )

    # add any missing player columns to the test matrix
    missing_p = {col: 0 for col in player_cols if col not in test_matrix.columns}
    if missing_p:
        test_matrix = pd.concat([test_matrix, pd.DataFrame(missing_p, index=test_matrix.index)], axis=1)

    base_cols = ["map_id", "datetime_utc", "result_diff_a", "team_a_won", "team_a_name", "team_b_name"]
    test_matrix = test_matrix[base_cols + player_cols]

    test_pair_col_df = build_pair_columns(pm_test, selected_pairs, list(test_matrix["map_id"]))

    missing_pairs = {pcol: 0 for pcol in train_pair_col_df.columns if pcol not in test_pair_col_df.columns}
    if missing_pairs:
        test_pair_col_df = pd.concat([test_pair_col_df, pd.DataFrame(missing_pairs, index=test_pair_col_df.index)], axis=1)

    if not train_pair_col_df.empty:
        test_pair_col_df = test_pair_col_df[train_pair_col_df.columns]

    actuals = test_matrix["result_diff_a"].to_numpy(dtype=float)
    actual_wins = test_matrix["team_a_won"].to_numpy(dtype=int)

    pred_bl = np.zeros(len(test_matrix))
    pred_m1 = predict_from_matrix(test_matrix, player_cols, ind_coefs)
    pred_m2 = predict_joint(test_matrix, player_cols, test_pair_col_df, joint_ind_coefs, joint_pair_coefs)

    results = {
        "Zero Baseline": compute_metrics(actuals, pred_bl, actual_wins),
        "Model 1: Individual RAPM": compute_metrics(actuals, pred_m1, actual_wins),
        "Model 2: Joint RAPM + Pair Interactions": compute_metrics(actuals, pred_m2, actual_wins),
    }

    metrics_table = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    for col in ["mae", "rmse", "win_loss_accuracy"]:
        metrics_table[col] = metrics_table[col].round(4)

    stat_tests = {
        "mcnemar_m1_vs_m2": mcnemar_test(actual_wins, pred_m1, pred_m2),
        "bootstrap_m1_vs_m2": bootstrap_improvement_ci(actuals, actual_wins, pred_m1, pred_m2),
    }

    label_lookup = player_ref.set_index("hltv_player_id")["label"].to_dict()
    coef_table = pd.DataFrame({"player_col": player_cols, "rapm_train_score": ind_coefs})
    coef_table["hltv_player_id"] = coef_table["player_col"].str.replace("p_", "", regex=False).astype(int)
    coef_table["label"] = coef_table["hltv_player_id"].map(label_lookup)
    coef_table = coef_table.sort_values("rapm_train_score", ascending=False).reset_index(drop=True)

    test_true_pm = pm_test[pm_test["hltv_player_id"].isin(active)].groupby("hltv_player_id").agg(
        test_true_plus_minus=("result_diff", "mean"),
        test_maps=("map_id", "nunique"),
        test_win_rate=("team_won", "mean"),
        test_avg_rating=("rating", "mean"),
    ).reset_index()

    player_level_eval = coef_table.merge(test_true_pm, on="hltv_player_id", how="inner")
    player_level_eval = player_level_eval[player_level_eval["test_maps"] >= 10].copy()

    corr = np.nan
    if len(player_level_eval) >= 3:
        corr = float(np.corrcoef(player_level_eval["rapm_train_score"], player_level_eval["test_true_plus_minus"])[0, 1])

    pair_coef_table = pd.DataFrame({"pair_col": train_pair_col_df.columns, "pair_coef": joint_pair_coefs})

    if not pair_coef_table.empty:
        pair_ids = pair_coef_table["pair_col"].str.extract(r"pr_(\d+)_(\d+)")
        pair_coef_table["player_a_id"] = pair_ids[0].astype(int)
        pair_coef_table["player_b_id"] = pair_ids[1].astype(int)
        pair_coef_table["player_a_label"] = pair_coef_table["player_a_id"].map(label_lookup)
        pair_coef_table["player_b_label"] = pair_coef_table["player_b_id"].map(label_lookup)
        pair_coef_table["abs_pair_coef"] = pair_coef_table["pair_coef"].abs()
        pair_coef_table = pair_coef_table.sort_values("abs_pair_coef", ascending=False).reset_index(drop=True)

    pair_survival = compute_pair_survival(train_pair_col_df, test_pair_col_df, pair_coef_table)

    return {
        "metrics_table": metrics_table,
        "statistical_tests": stat_tests,
        "uplift_pairs": uplift_pairs,
        "player_coef_table": coef_table,
        "pair_coef_table": pair_coef_table,
        "train_maps": len(train_matrix),
        "test_maps": len(test_matrix),
        "n_players_train": len(player_cols),
        "n_uplift_pairs": len(uplift_pairs),
        "cutoff_date": cutoff_date,
        "alpha_model1": alpha_m1,
        "alpha_model2": alpha_m2,
        "player_level_eval": player_level_eval,
        "player_level_corr_train_rapm_vs_test_pm": corr,
        "n_joint_pairs_selected": len(selected_pairs),
        "pair_survival": pair_survival,
    }