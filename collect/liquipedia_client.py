import requests
import time
import os
import logging

log = logging.getLogger(__name__)

API_BASE = "https://api.liquipedia.net/api/v3"

TIER_NAMES = {
    "s": "S-Tier",
    "a": "A-Tier",
    "b": "B-Tier",
    "c": "C-Tier",
}

VALID_ROLES = ["Rifler", "AWPer", "IGL", "Entry", "Lurker", "Support"]
ROLE_MAP = {
    "In-game leader": "IGL",
    "In-Game Leader": "IGL",
}


class LiquipediaClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        if self.api_key is None:
            self.api_key = os.environ.get("LIQUIPEDIA_API_KEY", "")
        self.last_request = 0

    def _get(self, endpoint, params):
        # rate limit as per https://liquipedia.net/api-terms-of-use
        diff = time.time() - self.last_request
        if diff < 1.2:
            time.sleep(1.2 - diff)

        headers = {
            "Authorization": "Apikey " + self.api_key,
            "Accept": "application/json",
        }

        url = API_BASE + "/" + endpoint
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        self.last_request = time.time()

        if resp.status_code == 429:
            log.warning("rate limited, sleeping 60s")
            time.sleep(60)
            resp = requests.get(url, headers=headers, params=params, timeout=30)

        resp.raise_for_status()
        return resp.json()

    def get_tournaments(self, tiers, start_date, end_date):
        tier_labels = []
        for t in tiers:
            if t in TIER_NAMES:
                tier_labels.append(TIER_NAMES[t])

        conditions = "[[liquipediatier::" + " OR ".join(tier_labels) + "]]"
        conditions += " AND [[date_start::>" + start_date + "]]"
        conditions += " AND [[date_end::<" + end_date + "]]"
        conditions += " AND [[status::Finished]]"

        params = {
            "wiki": "counterstrike",
            "conditions": conditions,
            "fields": "name, pagename, liquipediatier, date_start, date_end",
            "limit": 100,
        }

        data = self._get("tournament", params)
        results = []

        for t in data.get("result", []):
            row = {
                "id": t.get("pagename", ""),
                "name": t.get("name", ""),
                "tier": t.get("liquipediatier", ""),
                "start_date": t.get("date_start", ""),
                "end_date": t.get("date_end", ""),
            }
            results.append(row)

        log.info("got " + str(len(results)) + " tournaments")
        return results

    def get_matches(self, tournament_id):
        params = {
            "wiki": "counterstrike",
            "conditions": "[[tournament::" + tournament_id + "]] AND [[finished::1]]",
            "fields": "match2id, date, team1, team2, winner, score1, score2, extradata",
            "limit": 500,
        }

        data = self._get("match", params)
        matches = data.get("result", [])

        results = []
        for m in matches:
            extradata = m.get("extradata", {})
            if extradata is None:
                extradata = {}

            hltv_id = extradata.get("hltv_match_id", "")
            if hltv_id == "":
                hltv_id = extradata.get("hltvid", "")

            row = {
                "match2id": m.get("match2id", ""),
                "date": m.get("date", ""),
                "team1": m.get("team1", ""),
                "team2": m.get("team2", ""),
                "hltv_match_id": hltv_id,
            }
            results.append(row)

        return results

    def search_player(self, slug):
        params = {
            "wiki": "counterstrike",
            "conditions": "[[id::" + slug + "]]",
            "fields": "id, name, role, team",
            "limit": 1,
        }

        data = self._get("player", params)
        result = data.get("result", [])

        if len(result) == 0:
            return None

        return result[0]

    def normalise_role(self, raw_role):
        if raw_role is None:
            return None
        if raw_role == "":
            return None

        parts = raw_role.split(",")
        cleaned = []
        for p in parts:
            p = p.strip()
            if p in ROLE_MAP:
                p = ROLE_MAP[p]
            if p in VALID_ROLES:
                cleaned.append(p)

        if len(cleaned) == 0:
            return None

        return ", ".join(cleaned)