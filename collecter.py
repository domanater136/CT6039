import argparse
import hashlib
import logging
import os
import sqlite3
import sys
from pathlib import Path

from collect.hltv_client import HLTVClient
from collect.liquipedia_client import LiquipediaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS players (
            hltv_player_id  INTEGER PRIMARY KEY,
            hltv_slug       TEXT,
            player_name     TEXT,
            role            TEXT
        );

        CREATE TABLE IF NOT EXISTS matches (
            match_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            hltv_match_id   INTEGER UNIQUE,
            datetime_utc    TEXT,
            event_name      TEXT,
            team1           TEXT,
            team2           TEXT,
            stars           INTEGER
        );

        CREATE TABLE IF NOT EXISTS maps (
            map_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id        INTEGER,
            map_name        TEXT,
            team1_rounds    INTEGER,
            team2_rounds    INTEGER
        );

        CREATE TABLE IF NOT EXISTS map_players (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            map_id          INTEGER,
            match_id        INTEGER,
            hltv_player_id  INTEGER,
            player_name     TEXT,
            team_slot       TEXT,
            team_name       TEXT,
            opponent_team   TEXT,
            team_won        INTEGER,
            kills           INTEGER,
            deaths          INTEGER,
            rating          REAL,
            adr             REAL,
            kd_ratio        REAL,
            lineup_hash     TEXT
        );

        CREATE TABLE IF NOT EXISTS scrape_log (
            hltv_match_id   INTEGER PRIMARY KEY,
            status          TEXT,
            error           TEXT,
            fetched_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS discovery_tournaments (
            tournament_id   TEXT PRIMARY KEY,
            name            TEXT,
            tier            TEXT,
            start_date      TEXT,
            end_date        TEXT
        );
    """)
    conn.commit()


def add_role_column_if_missing(conn):
    # check if role column exists and add it if not
    cols = [row[1] for row in conn.execute("PRAGMA table_info(players)").fetchall()]
    if "role" not in cols:
        conn.execute("ALTER TABLE players ADD COLUMN role TEXT")
        conn.commit()
        log.info("added role column to players table")


def already_done(conn, hltv_match_id):
    row = conn.execute(
        "SELECT status FROM scrape_log WHERE hltv_match_id = ?", (hltv_match_id,)
    ).fetchone()
    if row is None:
        return False
    return row[0] == "ok"


def save_player(conn, player_id, slug, name):
    conn.execute("""
        INSERT INTO players (hltv_player_id, hltv_slug, player_name)
        VALUES (?, ?, ?)
        ON CONFLICT(hltv_player_id) DO UPDATE SET
            hltv_slug = excluded.hltv_slug,
            player_name = excluded.player_name
    """, (player_id, slug, name))


def get_match_id(conn, hltv_match_id):
    row = conn.execute(
        "SELECT match_id FROM matches WHERE hltv_match_id = ?", (hltv_match_id,)
    ).fetchone()
    if row is None:
        return None
    return row[0]


def store_match(conn, hltv_match_id, match_data, stub):
    event_name = ""
    event = match_data.get("event", {})
    if isinstance(event, dict):
        event_name = event.get("name", "")

    team1_name = stub.get("team1", "")
    team2_name = stub.get("team2", "")

    t1 = match_data.get("team1", {})
    if isinstance(t1, dict) and t1.get("name"):
        team1_name = t1["name"]

    t2 = match_data.get("team2", {})
    if isinstance(t2, dict) and t2.get("name"):
        team2_name = t2["name"]

    conn.execute("""
        INSERT OR IGNORE INTO matches
        (hltv_match_id, datetime_utc, event_name, team1, team2, stars)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        hltv_match_id,
        match_data.get("date", stub.get("date", "")),
        event_name,
        team1_name,
        team2_name,
        match_data.get("stars", 0),
    ))

    match_id = get_match_id(conn, hltv_match_id)
    if match_id is None:
        return 0

    maps = match_data.get("maps", [])
    if maps is None:
        maps = []

    maps_saved = 0

    for map_data in maps:
        map_name = map_data.get("name", "unknown")
        result = map_data.get("result", {})
        if result is None:
            result = {}

        t1_rounds = result.get("team1", 0)
        t2_rounds = result.get("team2", 0)

        # skip maps with no score
        if t1_rounds == 0 and t2_rounds == 0:
            continue

        stats = map_data.get("stats", {})
        if stats is None:
            stats = {}

        team1_players = stats.get("team1", [])
        team2_players = stats.get("team2", [])

        if team1_players is None:
            team1_players = []
        if team2_players is None:
            team2_players = []

        all_ids = []
        for p in team1_players:
            if p.get("id"):
                all_ids.append(p["id"])
        for p in team2_players:
            if p.get("id"):
                all_ids.append(p["id"])

        # need exactly 10 players or skip
        if len(all_ids) != 10:
            continue

        cur = conn.execute(
            "INSERT INTO maps (match_id, map_name, team1_rounds, team2_rounds) VALUES (?, ?, ?, ?)",
            (match_id, map_name, t1_rounds, t2_rounds)
        )
        map_id = cur.lastrowid

        t1_ids = sorted([p["id"] for p in team1_players if p.get("id")])
        t2_ids = sorted([p["id"] for p in team2_players if p.get("id")])

        t1_hash = hashlib.md5(",".join(str(i) for i in t1_ids).encode()).hexdigest()[:12]
        t2_hash = hashlib.md5(",".join(str(i) for i in t2_ids).encode()).hexdigest()[:12]

        team1_won = 1
        if t1_rounds < t2_rounds:
            team1_won = 0

        for slot, players, lineup_hash in [("A", team1_players, t1_hash), ("B", team2_players, t2_hash)]:
            if slot == "A":
                team_name = team1_name
                opp_name = team2_name
                won = team1_won
            else:
                team_name = team2_name
                opp_name = team1_name
                won = 1 - team1_won

            for p in players:
                pid = p.get("id")
                if pid is None:
                    continue

                pname = p.get("name", "")
                slug = p.get("slug", "")
                if slug == "":
                    slug = pname.lower().replace(" ", "")

                save_player(conn, pid, slug, pname)

                kills = p.get("kills", 0)
                deaths = p.get("deaths", 1)
                if deaths == 0:
                    deaths = 1
                kd = kills / deaths

                conn.execute("""
                    INSERT INTO map_players
                    (map_id, match_id, hltv_player_id, player_name, team_slot,
                     team_name, opponent_team, team_won, kills, deaths,
                     rating, adr, kd_ratio, lineup_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    map_id, match_id, pid, pname, slot,
                    team_name, opp_name, won,
                    kills, deaths,
                    p.get("rating", None),
                    p.get("adr", None),
                    kd,
                    lineup_hash,
                ))

        maps_saved += 1

    return maps_saved


def collect_roles(liquipedia, conn, overwrite=False):
    # get players that still need roles fetched
    if overwrite:
        rows = conn.execute(
            "SELECT hltv_player_id, hltv_slug FROM players WHERE hltv_slug IS NOT NULL"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT hltv_player_id, hltv_slug FROM players WHERE hltv_slug IS NOT NULL AND role IS NULL"
        ).fetchall()

    total = len(rows)
    log.info("fetching roles for " + str(total) + " players")

    found = 0
    not_found = 0

    for i, row in enumerate(rows):
        pid = row[0]
        slug = row[1]

        log.info("(" + str(i + 1) + "/" + str(total) + ") " + str(slug))

        try:
            player_data = liquipedia.search_player(slug)

            if player_data is None:
                log.info("  not found on liquipedia")
                conn.execute(
                    "UPDATE players SET role = 'Unknown' WHERE hltv_player_id = ?", (pid,)
                )
                conn.commit()
                not_found += 1
                continue

            raw_role = player_data.get("role", "")
            role = liquipedia.normalise_role(raw_role)

            if role is None:
                log.info("  no valid role found")
                conn.execute(
                    "UPDATE players SET role = 'Unknown' WHERE hltv_player_id = ?", (pid,)
                )
                not_found += 1
            else:
                log.info("  role: " + str(role))
                conn.execute(
                    "UPDATE players SET role = ? WHERE hltv_player_id = ?", (role, pid)
                )
                found += 1

            conn.commit()

        except Exception as e:
            log.warning("failed for " + str(slug) + ": " + str(e))
            not_found += 1

    log.info("roles done. found=" + str(found) + " not_found=" + str(not_found))


def print_role_summary(conn):
    total = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
    with_role = conn.execute(
        "SELECT COUNT(*) FROM players WHERE role IS NOT NULL AND role != 'Unknown'"
    ).fetchone()[0]
    unknown = conn.execute(
        "SELECT COUNT(*) FROM players WHERE role = 'Unknown'"
    ).fetchone()[0]

    print("\n--- Role Summary ---")
    print("Total players:      " + str(total))
    print("Players with role:  " + str(with_role))
    print("Unknown/not found:  " + str(unknown))
    print("No role yet:        " + str(total - with_role - unknown))

    roles = conn.execute(
        "SELECT role, COUNT(*) as n FROM players WHERE role IS NOT NULL GROUP BY role ORDER BY n DESC"
    ).fetchall()

    print("\nRole breakdown:")
    for row in roles:
        print("  " + str(row[0]).ljust(30) + str(row[1]))
    print("--------------------\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="hltv_dissertation.db")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--tiers", nargs="+", default=["s", "a", "b"])
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--hltv-url", default="http://localhost:3000")
    p.add_argument("--liquipedia-key", default=None)
    p.add_argument("--roles-only", action="store_true", help="only fetch roles, skip match collection")
    p.add_argument("--overwrite-roles", action="store_true", help="re-fetch roles even if already set")
    p.add_argument("--summary", action="store_true", help="print role summary and exit")
    p.add_argument("--skip-roles", action="store_true", help="skip role collection after match collection")
    args = p.parse_args()

    api_key = args.liquipedia_key
    if api_key is None:
        api_key = os.environ.get("LIQUIPEDIA_API_KEY", "")

    if api_key == "":
        log.warning("no liquipedia api key set -- role collection will fail")

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    init_db(conn)
    add_role_column_if_missing(conn)

    liquipedia = LiquipediaClient(api_key=api_key)
    hltv = HLTVClient(base_url=args.hltv_url)

    # just print summary and exit
    if args.summary:
        print_role_summary(conn)
        conn.close()
        return 0

    # just do roles, skip match collection
    if args.roles_only:
        collect_roles(liquipedia, conn, overwrite=args.overwrite_roles)
        print_role_summary(conn)
        conn.close()
        return 0

    # normal flow: collect matches then roles
    log.info("fetching tournaments from liquipedia...")
    tournaments = liquipedia.get_tournaments(
        tiers=args.tiers,
        start_date=args.start,
        end_date=args.end,
    )

    for t in tournaments:
        conn.execute("""
            INSERT OR IGNORE INTO discovery_tournaments
            (tournament_id, name, tier, start_date, end_date)
            VALUES (?, ?, ?, ?, ?)
        """, (t["id"], t["name"], t["tier"], t["start_date"], t["end_date"]))
    conn.commit()

    log.info("found " + str(len(tournaments)) + " tournaments")

    total_maps = 0

    for tournament in tournaments:
        t_id = tournament["id"]
        t_name = tournament["name"]
        log.info("collecting: " + t_name)

        matches = liquipedia.get_matches(t_id)
        if len(matches) == 0:
            log.info("  no matches found")
            continue

        for stub in matches:
            hltv_id = stub.get("hltv_match_id", "")
            if hltv_id == "" or hltv_id is None:
                continue

            hltv_id = int(hltv_id)

            if already_done(conn, hltv_id):
                continue

            if total_maps >= args.limit:
                log.info("hit map limit")
                break

            try:
                match_data = hltv.get_match(hltv_id)

                if match_data is None:
                    conn.execute(
                        "INSERT OR REPLACE INTO scrape_log (hltv_match_id, status) VALUES (?, ?)",
                        (hltv_id, "not_found")
                    )
                    conn.commit()
                    continue

                n = store_match(conn, hltv_id, match_data, stub)
                total_maps += n

                conn.execute(
                    "INSERT OR REPLACE INTO scrape_log (hltv_match_id, status) VALUES (?, ?)",
                    (hltv_id, "ok")
                )
                conn.commit()

                log.info("  match " + str(hltv_id) + " -> " + str(n) + " maps (total " + str(total_maps) + ")")

            except Exception as e:
                log.warning("  error on match " + str(hltv_id) + ": " + str(e))
                conn.execute(
                    "INSERT OR REPLACE INTO scrape_log (hltv_match_id, status, error) VALUES (?, ?, ?)",
                    (hltv_id, "error", str(e))
                )
                conn.commit()

        if total_maps >= args.limit:
            break

    log.info("match collection done. total maps: " + str(total_maps))

    # fetch roles for all players that dont have one yet
    if not args.skip_roles:
        log.info("starting role collection for new players...")
        collect_roles(liquipedia, conn, overwrite=args.overwrite_roles)
        print_role_summary(conn)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())