"""Metrica sample games 1 and 2 as labelled matches, in the same shape as the DFL bench.

Why: Metrica is the provider of the challenge's tracking, and its sample tracking is equally
raw -- the ball is visible on ~60% of frames (the challenge: 59%) -- so it needs no artificial
degradation.  That makes it the closest available stand-in for the challenge match.

Labels are built from the Metrica events (they are not shipped with the data):
  D  from the end of a BALL OUT, or from a FAULT RECEIVED, until the next SET PIECE starts;
     also before the first event of each half.
  H/A  the team of the latest on-ball event (PASS, RECOVERY, SET PIECE, SHOT, BALL LOST),
     carried forward.  A lost ball stays with the team that lost it until the other team
     recovers it -- the challenge's rule that a ball in flight stays with the kicker.
These are an approximation of a possession label, good enough to compare models, not to
read as absolute scores.

Data: other-games/metrica-sample-data/ (git clone https://github.com/metrica-sports/sample-data).
"Please acknowledge the source": Metrica Sports.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "other-games", "metrica-sample-data", "data")
CACHE = os.path.join(ROOT, "other-games", "metrica-sample-data", "cache")
GAMES = ("1", "2")
FPS = 25.0
L, W = 105.0, 68.0
HOME, AWAY = 0, 1

ON_BALL = ("PASS", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST")


def _tracking(game, side):
    path = os.path.join(DATA, "Sample_Game_%s" % game,
                        "Sample_Game_%s_RawTrackingData_%s_Team.csv" % (game, side))
    t = pd.read_csv(path, skiprows=2)
    cols = list(t.columns)
    players = []
    for k, c in enumerate(cols):
        if c.startswith("Player"):
            players.append((t[c].values.astype(np.float64), t[cols[k + 1]].values.astype(np.float64)))
    k = cols.index("Ball")
    ball = (t["Ball"].values.astype(np.float64), t[cols[k + 1]].values.astype(np.float64))
    return t["Period"].values, t["Frame"].values, players, ball


def _speed(x, y):
    """Speed in m/s from normalised positions, smoothed over 0.2 s (the ATD has a speed field;
    Metrica's sample CSVs do not)."""
    vx = np.gradient(x * L) * FPS
    vy = np.gradient(y * W) * FPS
    v = np.hypot(vx, vy)
    k = 5
    pad = np.pad(np.nan_to_num(v, nan=0.0), (k // 2, k // 2), mode="edge")
    sm = np.convolve(pad, np.ones(k) / k, mode="valid")
    return np.where(np.isfinite(v), sm, np.nan)


def _labels(game, n, period):
    ev = pd.read_csv(os.path.join(DATA, "Sample_Game_%s" % game,
                                  "Sample_Game_%s_RawEventsData.csv" % game))
    ev = ev.sort_values(["Start Frame", "End Frame"], kind="stable")
    gt = np.full(n, "X", dtype="<U1")
    # frame index = Frame - 1
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        gt[idx] = "D"                      # until the first event of the half
    owner = None
    dead = True
    rows = ev.to_dict("records")
    for k, e in enumerate(rows):
        a = int(e["Start Frame"]) - 1
        b = int(rows[k + 1]["Start Frame"]) - 1 if k + 1 < len(rows) else n
        typ, team = e["Type"], (HOME if e["Team"] == "Home" else AWAY)
        if typ == "SET PIECE":
            dead = False
        if typ in ON_BALL:
            owner = team
        if typ == "BALL OUT":
            # alive until the ball leaves (the event's end), dead after
            end = max(a, int(e["End Frame"]) - 1)
            if not dead and owner is not None:
                gt[a:end] = "H" if owner == HOME else "A"
            gt[end:b] = "D"
            dead = True
            continue
        if typ == "FAULT RECEIVED":
            dead = True
        if dead or owner is None:
            gt[a:b] = "D"
        else:
            gt[a:b] = "H" if owner == HOME else "A"
    gt[period == 0] = "X"
    return gt


def parse_game(game):
    """{players, ball, team_of_slot, gt, half_blocks} like dfl_bench.parse_match.  Cached."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "game%s.npz" % game)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    period, frame, home, ball = _tracking(game, "Home")
    _, _, away, _ = _tracking(game, "Away")
    n = len(frame)
    slots = home + away
    players = np.full((n, len(slots), 3), np.nan, dtype=np.float32)
    for j, (x, y) in enumerate(slots):
        players[:, j, 0], players[:, j, 1] = x, y
        players[:, j, 2] = _speed(x, y)
    b = np.full((n, 3), np.nan, dtype=np.float32)
    b[:, 0], b[:, 1] = ball
    team_of_slot = np.array([HOME] * len(home) + [AWAY] * len(away), dtype=np.int8)
    gt = _labels(game, n, period)
    blocks = []
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        blocks.append((p, int(idx[0]), int(idx[-1])))
    out = {"players": players, "ball": b, "team_of_slot": team_of_slot, "gt": gt,
           "half_blocks": np.array(blocks)}
    np.savez_compressed(cache, **out)
    return out
