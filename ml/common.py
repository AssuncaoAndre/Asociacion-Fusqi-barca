"""Shared pieces of the hybrid possession model: per-frame features and the decoder.

The learned model sits on top of the rules model (solution/model.py).  It reuses the rules
model's per-frame signals (distances, ball speed and its changes, visibility, activity,
heading) and the rules model's own answers as features, adds context from up to 5 s either
side of each frame, and learns P(H), P(A), P(D) per frame with gradient boosting.  The hybrid
then fixes in play vs dead to the rules model's answer and decodes the team with Viterbi: the
H/A sequence that best fits the probabilities, minus a cost for every change of team.

Training data is other matches only (the DFL open data, degraded to the challenge's noise
profile, and the Metrica sample games -- see validation/).  Nothing about the challenge match
is learned.
"""

import contextlib
import io
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "solution"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "validation"))

import model as M  # noqa: E402
import possession_metrics as pm  # noqa: E402

CLASSES = ("H", "A", "D")          # class index 0, 1, 2
STEP = 3                           # train on every 3rd frame (neighbours are near-copies)
SEED = 0

# ---------------------------------------------------------------- helpers

def centred_mean(x, k):
    """Mean over a centred window of k frames (edges use what is available)."""
    x = np.asarray(x, dtype=np.float64)
    c = np.concatenate(([0.0], np.cumsum(x)))
    n = len(x)
    i = np.arange(n)
    lo = np.clip(i - k // 2, 0, n)
    hi = np.clip(i + k // 2 + 1, 0, n)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def shifted(x, k):
    """x[i - k] (k > 0: the past; k < 0: the future), edge-padded."""
    x = np.asarray(x)
    if k == 0:
        return x.copy()
    out = np.empty_like(x)
    if k > 0:
        out[k:] = x[:-k]
        out[:k] = x[0]
    else:
        out[:k] = x[-k:]
        out[k:] = x[-1]
    return out


def frames_since(mask, cap):
    out = M.frames_since(np.asarray(mask, dtype=bool))
    return np.minimum(out, cap)


def frames_until(mask, cap):
    return frames_since(np.asarray(mask, dtype=bool)[::-1], cap)[::-1]


# ---------------------------------------------------------------- features

def rules_outputs(flags=None):
    """Run the rules model on whatever match the model module currently holds.  Returns the
    signals, the rules model's dead mask and team array."""
    flags = flags or {}
    old = {f: getattr(M, f) for f in flags}
    for f, v in flags.items():
        setattr(M, f, v)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            s = M.build_signals()
            dead, sig = M.detect_dead(s)
            team = M.detect_possession(s, dead, sig)
    finally:
        for f, v in old.items():
            setattr(M, f, v)
    return s, dead, team


def base_columns(s, dead, team, swap=False):
    """Frame-level columns.  swap=True mirrors the two teams (home <-> away), which doubles the
    training data and makes the model treat both teams alike."""
    L, W = M.L, M.W
    n = len(s["dH"])

    def clip(x, lo, hi, fill):
        x = np.asarray(x, dtype=np.float64)
        return np.clip(np.where(np.isfinite(x), x, fill), lo, hi)

    dH, dA = clip(s["dH"], 0, 30, 30), clip(s["dA"], 0, 30, 30)
    spH, spA = clip(s["sp_H"], 0, 12, 0), clip(s["sp_A"], 0, 12, 0)
    relH, relA = clip(s["rel_H"], 0, 30, 30), clip(s["rel_A"], 0, 30, 30)
    heading = s["heading"].astype(np.float64)
    rule_team = np.where(team == M.HOME, 1.0, np.where(team == M.AWAY, -1.0, 0.0))
    rule_team[dead] = 0.0

    # ball position from the home team's point of view: 0 = home's own goal line
    ex = np.asarray(s["ex"], dtype=np.float64)
    x_home = np.full(n, np.nan)
    for h, (lo, hi) in enumerate(M.half_ranges(n)):
        own = s["sides"].get((h, M.HOME), (0.0, 0))[0]
        x_home[lo:hi] = ex[lo:hi] if own == 0.0 else L - ex[lo:hi]
    x_home = clip(x_home, -5, L + 5, -10)

    if swap:
        dH, dA = dA, dH
        spH, spA = spA, spH
        relH, relA = relA, relH
        heading, rule_team = -heading, -rule_team
        x_home = np.where(x_home >= -5, L - x_home, x_home)

    ey = np.asarray(s["ey"], dtype=np.float64)
    cols = {
        "dH": dH, "dA": dA, "dHA": dH - dA, "dmin": np.minimum(dH, dA),
        "spH": spH, "spA": spA, "relH": relH, "relA": relA,
        "heading": heading, "rule_team": rule_team, "rule_dead": dead.astype(np.float64),
        "x_home": x_home, "y_centre": clip(np.abs(ey - W / 2), 0, W / 2 + 5, -1),
        "bseen": s["bseen"].astype(np.float64), "bvis": s["bvis_raw"].astype(np.float64),
        "bvalid": s["bvalid"].astype(np.float64),
        "oob": s["oob"].astype(np.float64), "crossed": s["crossed"].astype(np.float64),
        "bspd": clip(s["bspd"], 0, 40, -1), "bspd_sm": clip(s["bspd_sm"], 0, 40, -1),
        "bdec": clip(s["bspd_dec5"], -30, 30, 0), "bacc": clip(s["bspd_acc4"], -30, 30, 0),
        "bprev": clip(s["bspd_prev5"], 0, 40, -1),
        "act3": clip(s["act_sm"], 0, 1, 0), "act15": clip(s["act15_sm"], 0, 1, 0),
        "pmax": clip(s["pmax_sm"], 0, 12, 0),
    }
    # how long since / until things happened (in frames, capped)
    cols["since_seen"] = frames_since(s["bvis_raw"], 500)
    cols["until_seen"] = frames_until(s["bvis_raw"], 500)
    cols["since_oob"] = frames_since(s["oob"], 500)
    change = np.concatenate(([False], np.diff(rule_team) != 0))
    cols["since_rule_change"] = frames_since(change, 500)
    cols["until_rule_change"] = frames_until(change, 500)
    return cols


CONTEXT = ["dHA", "dmin", "bseen", "bvis", "bspd_sm", "act3", "act15", "heading",
           "rule_team", "rule_dead", "oob", "x_home"]
WINDOWS = (13, 51, 125)            # ~0.5 s, 2 s, 5 s
SHIFTS = (-25, -12, 12, 25)        # +-0.5 s and 1 s
SHIFT_COLS = ["dHA", "bspd_sm", "bseen", "rule_team"]


def feature_matrix(cols):
    """Frame columns + centred window means + past/future values."""
    feats = dict(cols)
    for k in CONTEXT:
        for w in WINDOWS:
            feats["%s_m%d" % (k, w)] = centred_mean(cols[k], w)
    for k in SHIFT_COLS:
        for sh in SHIFTS:
            feats["%s_s%+d" % (k, sh)] = shifted(cols[k], sh)
    names = sorted(feats)
    X = np.column_stack([feats[k] for k in names]).astype(np.float32)
    return X, names


def label_index(labels, swap=False):
    """'H'/'A'/'D' -> 0/1/2 (-1 for X).  swap mirrors H and A."""
    y = np.full(len(labels), -1, dtype=np.int8)
    y[labels == "H"] = 1 if swap else 0
    y[labels == "A"] = 0 if swap else 1
    y[labels == "D"] = 2
    return y


# ---------------------------------------------------------------- decoding

def viterbi(logp, graded_runs, cost_team, cost_dead):
    """Best H/A/D path per graded run: sum of log P(state) minus a cost per change
    (cost_team for H<->A, cost_dead for into/out of D)."""
    n = len(logp)
    out = np.full(n, "X", dtype="<U1")
    trans = ((0.0, cost_team, cost_dead),
             (cost_team, 0.0, cost_dead),
             (cost_dead, cost_dead, 0.0))
    for a, b in graded_runs:
        lp = logp[a:b].tolist()
        m = b - a
        back = [None] * m
        v = list(lp[0])
        for t in range(1, m):
            row = lp[t]
            nv = [0.0, 0.0, 0.0]
            bt = [0, 0, 0]
            for j in range(3):
                best_i, best = 0, v[0] - trans[0][j]
                for i in (1, 2):
                    c = v[i] - trans[i][j]
                    if c > best:
                        best_i, best = i, c
                nv[j] = best + row[j]
                bt[j] = best_i
            v = nv
            back[t] = bt
        st = max(range(3), key=lambda j: v[j])
        path = np.empty(m, dtype=np.int8)
        for t in range(m - 1, -1, -1):
            path[t] = st
            if t:
                st = back[t][st]
        out[a:b] = np.array(CLASSES)[path]
    return out


def force_dead(logp, rule_dead):
    """Hybrid mode: in-play vs dead comes from the rules model, the ML only picks the team.
    Frames the rules call dead can only be D; frames they call live can only be H or A."""
    out = logp.copy()
    out[rule_dead, 0] = out[rule_dead, 1] = -1e9
    out[~rule_dead, 2] = -1e9
    return out


def decode_hybrid(probs, rule_dead, graded_runs, cost_team):
    """The hybrid's labels: D where the rules model says dead, otherwise the H/A sequence that
    best fits the learned probabilities minus cost_team per change of team.  The dead cost is
    irrelevant (force_dead leaves no choice there) and is set to 1."""
    logp = np.log(np.clip(probs, 1e-6, 1.0))
    return viterbi(force_dead(logp, rule_dead), graded_runs, cost_team, 1)


def graded_runs_of(half_blocks):
    return [(int(a), int(b) + 1) for _, a, b in half_blocks]


def score(gt, labels):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels))
    return {"spell": r["spell_f1_mean"], "macro": r["macro_f1"],
            "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"],
            "team": r["team_acc_gt_alive"], "ad": r["alive_dead_f1"],
            "iou": [r["spell_f1"][i]["f1"] for i in pm.SPELL_IOUS]}
