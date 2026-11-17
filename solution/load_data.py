"""Fast loader for the tactical ATD tracking data, with numpy caching.

The raw ATD line is:  <frame_id>:<p1>;<p2>;...;<p46>:<ball>
where each group is "x,y,speed" in normalised 0-1 pitch coordinates (NaN when missing).

Parsing 199,319 lines x 142 float fields with pandas' C engine takes ~15 s; the result is
cached as .npy files so every later experiment starts in well under a second.
"""

import os
import sys
import io
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

CACHE_DIR = os.environ.get("MTAH_CACHE_DIR", "/tmp/mtah_cache")
N_FRAMES = source_frames("tactical", GAME)
PITCH = GAMES[GAME]["pitch"]  # (105.3, 68.0) metres


def team_of_each_track(meta_path):
    """(team_of_slot int8 array, home_name, away_name). 0 = home, 1 = away."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    home_id = teams[0].get("id")
    home_name = teams[0].findtext("Name")
    away_name = teams[1].findtext("Name")
    slots = [p.get("teamId") for p in root.iter("Player")]
    return (np.array([0 if t == home_id else 1 for t in slots], dtype=np.int8),
            home_name, away_name)


def parse_atd(raw_path):
    """Parse the raw ATD into (frame_ids, players, ball).

    players: (n_frames, n_slots, 3) float32  -- x, y, speed (normalised coords, m/s speed)
    ball:    (n_frames, 3) float32           -- x, y, speed
    """
    with open(raw_path, "r") as handle:
        text = handle.read()
    # Normalise the two structural separators to commas: every line then has exactly
    # 1 + 46*3 + 3 = 142 comma-separated float fields.
    text = text.replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32).values
    frame_ids = table[:, 0].astype(np.int64)
    players = table[:, 1:1 + 46 * 3].reshape(len(table), 46, 3)
    ball = table[:, 1 + 46 * 3:1 + 46 * 3 + 3]
    return frame_ids, players, ball


def load_all(game=GAME, data_dir=None, use_cache=True):
    """Return a dict of arrays covering the whole tactical video. Cached as .npy."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "atd_tactical.npz")

    if use_cache and os.path.isfile(cache_path):
        z = np.load(cache_path)
        return {k: z[k] for k in z.files}

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team_of_slot, home_name, away_name = team_of_each_track(meta_path)

    t0 = time.time()
    frame_ids, players, ball = parse_atd(raw_path)
    print("parsed ATD in {:.1f}s".format(time.time() - t0))
    assert len(frame_ids) == N_FRAMES, (len(frame_ids), N_FRAMES)
    assert frame_ids[0] == 1 and frame_ids[-1] == N_FRAMES

    out = {
        "frame_ids": frame_ids,
        "players": players,          # (n, 46, 3) x, y, speed
        "ball": ball,                # (n, 3) x, y, speed
        "team_of_slot": team_of_slot,  # (46,) 0 home / 1 away
        "home_name": np.array(home_name),
        "away_name": np.array(away_name),
    }
    np.savez_compressed(cache_path, **out)
    print("cached to", cache_path)
    return out


def load_smart_tagging(game=GAME, data_dir=None):
    """Smart tagging instances as a list of dicts with video-frame intervals."""
    path = require(["smart_tagging"], game, data_dir)[0]
    fps = GAMES[game]["fps"]
    events = []
    for instance in ET.parse(path).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        events.append({
            "code": (instance.findtext("code") or "").strip(),
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "tags": {label.findtext("group"): (label.findtext("text") or "").strip()
                     for label in instance.findall("label")},
        })
    return events


def load_gt_public():
    path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    table = pd.read_csv(path)
    return table["label"].values


# Graded video-frame blocks: (half, first, last) inclusive.
HALF_BLOCKS = [(1, 11185, 90651), (2, 110808, 190197)]


def graded_mask(n=N_FRAMES):
    mask = np.zeros(n, dtype=bool)
    for _, first, last in HALF_BLOCKS:
        mask[first:last + 1] = True
    return mask


if __name__ == "__main__":
    data = load_all()
    players, ball = data["players"], data["ball"]
    print("players", players.shape, "ball", ball.shape)
    print("ball visible on {} of {} frames ({:.1f}%)".format(
        int(np.sum(~np.isnan(ball[:, 0]))), N_FRAMES,
        100 * np.mean(~np.isnan(ball[:, 0]))))
    print("player slots present per frame: mean {:.1f}".format(
        np.mean(np.sum(~np.isnan(players[:, :, 0]), axis=1))))
    print("team_of_slot:", data["team_of_slot"])
    print("home:", data["home_name"], " away:", data["away_name"])
