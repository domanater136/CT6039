import requests
import time
import logging

log = logging.getLogger(__name__)


class HLTVClient:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.last_request = 0

    def _get(self, path, params=None):
        # dont hammer the api
        diff = time.time() - self.last_request
        if diff < 1.0:
            time.sleep(1.0 - diff)

        url = self.base_url + path
        log.debug("fetching " + url)

        tries = 0
        while tries < 3:
            try:
                resp = requests.get(url, params=params, timeout=30)
                self.last_request = time.time()

                if resp.status_code == 404:
                    return None

                if resp.status_code == 429:
                    log.warning("rate limited, waiting...")
                    time.sleep(30)
                    tries += 1
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.ConnectionError:
                log.error("cant connect to hltv-api, is it running? (cd hltv-api && npm start)")
                time.sleep(5)
                tries += 1

        return None

    def get_match(self, match_id):
        return self._get("/match/" + str(match_id))

    def get_results(self, stars=1, offset=0, start_date=None, end_date=None):
        params = {
            "stars": stars,
            "offset": offset,
        }
        if start_date is not None:
            params["startDate"] = start_date
        if end_date is not None:
            params["endDate"] = end_date

        data = self._get("/results", params)
        if data is None:
            return []

        if isinstance(data, list):
            return data

        return data.get("results", [])

    def get_player(self, player_id):
        return self._get("/player/" + str(player_id))

    def get_ranking(self):
        data = self._get("/ranking")
        if data is None:
            return []
        if isinstance(data, list):
            return data
        return data.get("ranking", [])