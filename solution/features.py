"""Per-frame feature computation: ball recovery (Kalman), player geometry, activity,
smart-tagging priors.

Everything is vectorised numpy over the 199,319 tactical frames. The ball is missing on
~41% of graded frames, so the first job is a constant-velocity Kalman filter that carries
the ball across gaps; downstream features use the recovered track.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load_data import load_all, load_smart_tagging, PITCH, N_FRAMES

L, W = PITCH          # 105.3 x 68.0 metres
FPS = 25.0

HOME, AWAY = 0, 1


# ----------------------------------------------------------------------------- ball recovery

def kalman_ball(bx, by, bvis, fps=FPS,
                obs_noise=0.35,      # metres, trust in a seen ball position
                accel_noise=6.0,     # m/s^2 process noise (ball can curve/bounce)
                jump_dist=12.0):     # metres; larger re-initialises the filter
    """Constant-velocity Kalman filter over the ball track.

    Returns (rx, ry, rspeed, runcertain) -- recovered position, recovered speed, and the
    positional standard deviation (a confidence measure). Missing observations are predicted
    through; a large innovation re-initialises the filter (the tracker switched to a
    different object or the ball teleported after being out of play).
    """
    n = len(bx)
    rx = np.zeros(n)
    ry = np.zeros(n)
    rspeed = np.zeros(n)
    runc = np.zeros(n)

    # state: x, y, vx, vy  (metres, m/s)
    F = np.array([[1, 0, 1 / fps, 0],
                  [0, 1, 0, 1 / fps],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float64)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=np.float64)
    R = np.eye(2) * obs_noise ** 2
    Q = np.eye(4) * 0.0
    dt = 1.0 / fps
    # process noise: acceleration uncertainty grows with the square of the step
    Q[0, 0] = Q[1, 1] = (accel_noise ** 2) * dt ** 4 / 4.0
    Q[0, 2] = Q[2, 0] = (accel_noise ** 2) * dt ** 3 / 2.0
    Q[1, 3] = Q[3, 1] = (accel_noise ** 2) * dt ** 3 / 2.0
    Q[2, 2] = Q[3, 3] = (accel_noise ** 2) * dt ** 2

    x = np.zeros(4)
    P = np.eye(4) * 100.0
    init = False

    for i in range(n):
        if bvis[i]:
            z = np.array([bx[i], by[i]])
            if not init:
                x = np.array([bx[i], by[i], 0.0, 0.0])
                P = np.eye(4) * 1.0
                init = True
            else:
                # innovation gate: reject wild jumps (tracker identity switch)
                pred = F @ x
                innov = z - H @ pred
                S = H @ P @ H.T + R
                if np.sqrt(innov @ np.linalg.solve(S, innov)) > 6.0:
                    # re-initialise at the observation with a fresh velocity
                    x = np.array([bx[i], by[i], x[2], x[3]])
                    P = np.eye(4) * 1.0
                else:
                    K = P @ H.T @ np.linalg.inv(S)
                    x = pred + K @ innov
                    P = (np.eye(4) - K @ H) @ P
        else:
            if init:
                x = F @ x
                P = F @ P @ F.T + Q
            else:
                rx[i] = np.nan
                ry[i] = np.nan
                rspeed[i] = np.nan
                runc[i] = np.inf
                continue
        rx[i] = x[0]
        ry[i] = x[1]
        rspeed[i] = np.hypot(x[2], x[3])
        runc[i] = np.sqrt(P[0, 0] + P[1, 1])
    return rx, ry, rspeed, runc


# ----------------------------------------------------------------------------- features

def compute_features():
    data = load_all()
    players = data["players"]            # (n, 46, 3)
    ball = data["ball"]                  # (n, 3)
    team_of_slot = data["team_of_slot"]  # (46,)

    px = players[:, :, 0] * L
    py = players[:, :, 1] * W
    pspeed = players[:, :, 2]
    bx = ball[:, 0] * L
    by = ball[:, 1] * W
    bvis = ~np.isnan(ball[:, 0])

    # -- ball recovery ---------------------------------------------------------
    rx, ry, rspeed, runc = kalman_ball(bx, by, bvis)

    # -- distances from (recovered) ball to every player ------------------------
    # use the recovered position but only trust it where the filter is not lost
    trust = np.isfinite(rx) & (runc < 8.0)   # within 8 m positional sigma
    ex = np.where(trust, rx, np.nan)
    ey = np.where(trust, ry, np.nan)
    # when the ball is visible, snap to the observation (the filter lags slightly)
    ex = np.where(bvis, bx, ex)
    ey = np.where(bvis, by, ey)

    dx = px - ex[:, None]
    dy = py - ey[:, None]
    dist = np.hypot(dx, dy)              # (n, 46)
    dist = np.where(np.isnan(px), np.inf, dist)

    home_slots = team_of_slot == HOME
    away_slots = team_of_slot == AWAY
    dH = dist[:, home_slots].min(axis=1)
    dA = dist[:, away_slots].min(axis=1)
    dall = np.minimum(dH, dA)

    # nearest player of each team: index and speed
    def nearest(dist_team):
        idx = np.argmin(dist_team, axis=1)
        rows = np.arange(len(dist_team))
        d = dist_team[rows, idx]
        return idx, d

    hidx, _ = nearest(dist[:, home_slots])
    aidx, _ = nearest(dist[:, away_slots])
    home_slot_ids = np.flatnonzero(home_slots)
    away_slot_ids = np.flatnonzero(away_slots)
    sp_near_H = pspeed[np.arange(N_FRAMES), home_slot_ids[hidx]]
    sp_near_A = pspeed[np.arange(N_FRAMES), away_slot_ids[aidx]]

    # -- ball geometry ----------------------------------------------------------
    bspeed_raw = np.full(N_FRAMES, np.nan)
    bspeed_raw[1:] = np.hypot(np.diff(bx), np.diff(by)) * FPS
    bspeed = np.where(bvis, bspeed_raw, rspeed)   # recovered speed when missing
    boob = bvis & ((bx < -0.5) | (bx > L + 0.5) | (by < -0.5) | (by > W + 0.5))

    # -- player activity --------------------------------------------------------
    present = ~np.isnan(pspeed)
    n_present = present.sum(axis=1)
    ps = np.where(present, pspeed, np.nan)
    act3 = np.nanmean(np.where(present, pspeed > 3.0, np.nan), axis=1)
    act4 = np.nanmean(np.where(present, pspeed > 4.0, np.nan), axis=1)
    with np.errstate(all="ignore"):
        p90 = np.nanpercentile(ps, 90, axis=1)
        pmax = np.nanmax(ps, axis=1)
        pmed = np.nanmedian(ps, axis=1)
    # per-team activity
    act3_H = np.nanmean(np.where(present & home_slots[None, :], pspeed > 3.0, np.nan), axis=1)
    act3_A = np.nanmean(np.where(present & away_slots[None, :], pspeed > 3.0, np.nan), axis=1)

    # -- smart tagging priors ---------------------------------------------------
    events = load_smart_tagging()
    phase_team = np.full(N_FRAMES, -1, dtype=np.int8)   # team the phase says has the ball
    setpiece_start = np.zeros(N_FRAMES, dtype=bool)
    setpiece_team = np.full(N_FRAMES, -1, dtype=np.int8)
    setpiece_type = np.full(N_FRAMES, "", dtype=object)

    OFFENSIVE = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                 "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL",
                 "GOALS"}
    DEFENSIVE = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                 "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}

    def team_code(name):
        if name == "Barcelona":
            return HOME
        if name in ("Atletic Club", "Athletic Club"):
            return AWAY
        return -1

    for ev in events:
        s, e = max(0, ev["start_frame"]), min(N_FRAMES, ev["end_frame"])
        if e <= s:
            continue
        tc = team_code(ev["tags"].get("Team", "N/A"))
        if ev["code"] == "SET PIECES":
            setpiece_start[s:e] = True
            if tc >= 0:
                setpiece_team[s:e] = tc
            setpiece_type[s:e] = ev["tags"].get("Type", "")
        elif ev["code"] in OFFENSIVE and tc >= 0:
            phase_team[s:e] = tc
        elif ev["code"] in DEFENSIVE and tc >= 0:
            phase_team[s:e] = 1 - tc   # the defending team's opponent has the ball

    feats = {
        "bx": bx, "by": by, "bvis": bvis,
        "rx": ex, "ry": ey, "rspeed": rspeed, "runc": runc, "trust": trust,
        "bspeed": bspeed, "bspeed_raw": bspeed_raw, "boob": boob,
        "dH": dH, "dA": dA, "dall": dall,
        "sp_near_H": sp_near_H, "sp_near_A": sp_near_A,
        "act3": act3, "act4": act4, "p90": p90, "pmax": pmax, "pmed": pmed,
        "act3_H": act3_H, "act3_A": act3_A, "n_present": n_present,
        "phase_team": phase_team, "setpiece_start": setpiece_start,
        "setpiece_team": setpiece_team,
    }
    return feats


def rolling_mean(x, k):
    """Centred rolling mean, edge-padded. k must be odd."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(np.nan_to_num(x, nan=0.0), (k // 2, k // 2), mode="edge")
    out = sliding_window_view(pad, k).mean(axis=1)
    # correct for NaN padding: count valid via a second pass on a mask
    mask = ~np.isnan(x)
    padm = np.pad(mask.astype(float), (k // 2, k // 2), mode="edge")
    cnt = sliding_window_view(padm, k).sum(axis=1)
    return out, cnt


if __name__ == "__main__":
    f = compute_features()
    print("features computed")
    for k in ("dH", "dA", "dall", "act3", "pmax", "bspeed", "rspeed", "runc"):
        v = f[k]
        print("  {:>10}: med {:.3f}  p10 {:.3f}  p90 {:.3f}".format(
            k, np.nanmedian(v), np.nanpercentile(v, 10), np.nanpercentile(v, 90)))
    print("  trust: {:.3f}  boob: {}".format(np.mean(f["trust"]), int(f["boob"].sum())))
    print("  phase_team set: H {} A {} none {}".format(
        int((f["phase_team"] == 0).sum()), int((f["phase_team"] == 1).sum()),
        int((f["phase_team"] < 0).sum())))
