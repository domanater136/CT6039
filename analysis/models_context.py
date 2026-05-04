import logging
import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    _HAS_SM = True
except ImportError:
    _HAS_SM = False
    logging.getLogger(__name__).warning("statsmodels not found.")

log = logging.getLogger(__name__)


def run_one_regression(df, outcome, pre_own, pre_tm, pre_opp, cluster_col="match_id"):
    if not _HAS_SM:
        raise ImportError("statsmodels required. pip install statsmodels")

    needed = [outcome, pre_own, pre_tm, pre_opp, "hltv_player_id", "time_quarter", cluster_col]
    sub = df[needed].dropna().copy()

    if len(sub) < 50:
        log.warning(f"Skipping {outcome} regression. only {len(sub)} rows after dropping NaN")
        return {}

    # standardise all columns so the betas are directly comparable across metrics
    for col in [outcome, pre_own, pre_tm, pre_opp]:
        std = sub[col].std()
        if std > 0:
            sub[col] = sub[col] / std

    # within-player demean to control for player fixed effects
    for col in [outcome, pre_own, pre_tm, pre_opp]:
        player_mean = sub.groupby("hltv_player_id")[col].transform("mean")
        sub["dm_" + col] = sub[col] - player_mean

    formula = (
        "dm_" + outcome
        + " ~ dm_" + pre_own
        + " + dm_" + pre_tm
        + " + dm_" + pre_opp
        + " + C(time_quarter) - 1"
    )

    try:
        fit = smf.ols(formula=formula, data=sub).fit(
            cov_type="cluster",
            cov_kwds={"groups": sub[cluster_col]},
        )
    except Exception as exc:
        log.error(f"OLS failed for {outcome}: {exc}")
        return {}

    name_map = {
        "dm_" + pre_own: "coef_own_pre",
        "dm_" + pre_tm: "coef_tm_pre",
        "dm_" + pre_opp: "coef_opp_pre",
    }

    out = {
        "outcome": outcome,
        "n_obs": int(fit.nobs),
        "r2": float(fit.rsquared),
        "r2_adj": float(fit.rsquared_adj),
        "model": fit,
    }

    for sm_col, label in name_map.items():
        if sm_col in fit.params.index:
            out[label] = float(fit.params[sm_col])
            out[label + "_se"] = float(fit.bse[sm_col])
            out[label + "_pval"] = float(fit.pvalues[sm_col])
        else:
            out[label] = np.nan
            out[label + "_se"] = np.nan
            out[label + "_pval"] = np.nan

    log.info("OLS {} | n={} | R2={:.4f} | tm_coef={:.4f}".format(
        outcome, out["n_obs"], out["r2"], out.get("coef_tm_pre", np.nan)
    ))
    return out


def run_context_regressions(player_map, min_maps_context=20):
    # which metrics to run and what their pre-map columns are called
    metrics_config = {
        "rating": ("pre_rating", "tm_pre_rating", "opp_pre_rating"),
        "adr": ("pre_adr", "tm_pre_adr", "opp_pre_adr"),
        "kd_log": ("pre_kd_log", "tm_pre_rating", "opp_pre_rating"),
    }

    # filter to players with enough maps
    appearances = player_map.groupby("hltv_player_id")["map_id"].nunique()
    eligible = appearances[appearances >= min_maps_context].index
    sub = player_map[player_map["hltv_player_id"].isin(eligible)].copy()

    log.info(f"Context regressions: {len(eligible)} eligible players, {len(sub)} rows")

    results = {}
    for outcome, (pre_own, pre_tm, pre_opp) in metrics_config.items():
        missing_cols = [c for c in [outcome, pre_own, pre_tm, pre_opp] if c not in sub.columns]
        if missing_cols:
            log.warning(f"Skipping {outcome} |  missing columns: {missing_cols}")
            continue

        results[outcome] = run_one_regression(sub, outcome=outcome, pre_own=pre_own, pre_tm=pre_tm, pre_opp=pre_opp)

    return results


def context_coeff_table(results):
    rows = []
    for outcome, res in results.items():
        if not res:
            continue
        rows.append({
            "metric": outcome,
            "n_obs": res.get("n_obs"),
            "r2_adj": res.get("r2_adj"),
            "coef_own_pre": res.get("coef_own_pre"),
            "coef_own_pre_pval": res.get("coef_own_pre_pval"),
            "coef_tm_pre": res.get("coef_tm_pre"),
            "coef_tm_pre_pval": res.get("coef_tm_pre_pval"),
            "coef_opp_pre": res.get("coef_opp_pre"),
            "coef_opp_pre_pval": res.get("coef_opp_pre_pval"),
        })
    return pd.DataFrame(rows)