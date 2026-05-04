CS2 Dissertation for Detecting Teammate Influence in Professional CS2

Code for my BSc dissertation. Collects professional CS2 match data from HLTV via Liquipedia tournament discovery, then runs a statistical analysis pipeline to detect and measure teammate influence on individual player performance.

Data covers S, A and B tier tournaments from January 2024 to December 2025.

---

## Requirements

You need a Liquipedia API key to collect data. Set it as an environment variable or pass it with `--liquipedia-key`. You also need the HLTV API running locally (see `hltv-api/`).

Install dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt covers:**
- `pandas`, `numpy`, `scikit-learn` - data handling and modelling
- `scipy`, `statsmodels` - statistical tests and regressions
- `matplotlib` - plots
- `requests` - API calls
---

## Usage

### Step 1: Collect data

```bash
python collecter.py
```

This pulls tournaments from Liquipedia, fetches match data from the HLTV API, and stores everything in a local SQLite database. Role data is fetched automatically at the end.

This requires alot of set-up, an API key you have to apply for, and a locally run node server. I'd personally suggest just using the database provided and skip this step as collection takes a long amount of time as well due to the rate limits of both Liquipedia and HLTV.

Key flags:
```
--db                database file (default: hltv_dissertation.db)
--start             start date (default: 2024-01-01)
--end               end date (default: 2025-12-31)
--tiers             which tiers to include, e.g. s a b (default: s a b)
--limit             max maps to collect (default: 10000)
--hltv-url          URL for local HLTV API (default: http://localhost:3000)
--liquipedia-key    API key (or set LIQUIPEDIA_API_KEY env variable)
--roles-only        skip match collection, just fetch roles
--overwrite-roles   re-fetch roles even if already set
--skip-roles        skip role collection after collecting matches
--summary           print role coverage summary and exit
```

### Step 2: Run analysis

```bash
python analyse.py
```

Runs the full pipeline: feature engineering, context regressions, RAPM, teammate influence, predictive validation, plots, and reports. 

Key flags:
```
--db              database file (default: hltv_dissertation.db)
--out-dir         output directory (default: outputs)
--min-maps        minimum maps for player to be included in summaries (default: 20)
--min-maps-rapm   minimum maps for RAPM eligibility (default: 50)
--cutoff          train/test split date (default: 2025-07-01)
--min-pair-maps   minimum shared maps for a pair to be tested (default: 20)
--placebo-iters   number of placebo shuffle iterations (default: 100)
--n-jobs          parallel workers, -1 uses all cores (default: -1)
--pre-window      maps before a roster change to include (default: 10)
--post-window     maps after a roster change to include (default: 10)
--skip-context    skip context sensitivity regressions
--skip-rapm       skip RAPM
--skip-prediction skip predictive validation
--skip-influence  skip the teammate influence framework
```

---

## Output

The main outputs are:
- `rapm_scores.csv` - RAPM scores for all eligible players
- `indirect_teammate_impact.csv` - ITI scores
- `multi_outcome_uplift.csv` - pairwise uplift results
- `roster_change_events.csv` - roster change event study
- `prediction_metrics.csv` - model comparison metrics
- `report_summary.md` - summary report
