"""DFL test bench: score the possession model on the 7 open Bundesliga matches.

Why: the public 5 minutes hold 17 possessions -- too few to tell a real improvement from
noise, and the rules forbid tuning on them.  The DFL/IDSSE open data (CC-BY 4.0, credit the
DFL; Bassek et al., Scientific Data 2025, doi:10.1038/s41597-025-04505-y) has 7 full matches
at 25 fps whose ball frames carry the league's own BallStatus (in/out of play) and
BallPossession (home/away) -- the same two questions the model answers.

The DFL tracking is far cleaner than the challenge's automatic tracking, so each match is
DEGRADED to the challenge's noise profile first.  The profile is measured from the challenge
ATD itself (an input, not an answer): hidden-ball and visible-ball stretch lengths, the ball
glitch rate, player presence and track-segment lengths.

Usage:
    python validation/dfl_bench.py                 # degraded, v10 vs v12 and each change
    python validation/dfl_bench.py --clean         # also score on the clean DFL tracking
    python validation/dfl_bench.py --matches J03WPY J03WR9

Data: other-games/dfl/ (see other-games/SOURCES.md).  Parsed matches are cached in
other-games/dfl/cache/.
"""

import argparse
import contextlib
import glob
import io
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DFL = os.path.join(ROOT, "other-games", "dfl")
CACHE = os.path.join(DFL, "cache")
sys.path.insert(0, os.path.join(ROOT, "solution"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import load_data  # noqa: E402
import model as M  # noqa: E402
import possession_metrics as pm  # noqa: E402

HALF_GAP = 500          # ungraded frames inserted between the halves
DFL_L, DFL_W = 105.0, 68.0

# ---------------------------------------------------------------- parsing

FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
FRAME = re.compile(r'<Frame N="(\d+)" T="[^"]*" X="([-\d.]+)" Y="([-\d.]+)"(?: Z="[-\d.]+")?'
                   r' D="[-\d.]+" S="([-\d.]+)"'
                   r'(?:.*?BallPossession="(\d)")?(?:.*?BallStatus="(\d)")?')


def match_files(match_id):
    pos = glob.glob(os.path.join(DFL, "DFL_04_03_positions_raw_observed_*%s.xml" % match_id))
    info = glob.glob(os.path.join(DFL, "DFL_02_01_matchinformation_*%s.xml" % match_id))
    if not pos or not info:
        raise FileNotFoundError("DFL files for %s not found in %s" % (match_id, DFL))
    return pos[0], info[0]


def home_team_id(info_path):
    with open(info_path, encoding="utf-8") as handle:
        return re.search(r'HomeTeamId="([^"]+)"', handle.read()).group(1)


def parse_match(match_id):
    """Arrays in the shape the model's load_all() returns, plus the true labels."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, match_id + ".npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}

    pos_path, info_path = match_files(match_id)
    home = home_team_id(info_path)
    sets = []   # (section, team, person, rows)
    rows = None
    with open(pos_path, encoding="utf-8") as handle:
        for line in handle:
            if "<Frame " in line:
                m = FRAME.search(line)
                rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)),
                             float(m.group(4)), int(m.group(5) or 0), int(m.group(6) or 0)))
            elif "<FrameSet" in line:
                m = FRAMESET.search(line)
                rows = []
                sets.append((m.group(1), m.group(2), m.group(3), rows))

    sections = ["firstHalf", "secondHalf"]
    ball_sets = {sec: np.array(r) for sec, team, _, r in sets if team == "BALL"}
    first = {sec: int(ball_sets[sec][:, 0].min()) for sec in sections}
    length = {sec: int(ball_sets[sec][:, 0].max()) - first[sec] + 1 for sec in sections}
    offset = {"firstHalf": 0, "secondHalf": length["firstHalf"] + HALF_GAP}
    n = offset["secondHalf"] + length["secondHalf"]

    persons = sorted({p for _, team, p, _ in sets if team != "BALL"})
    slot = {p: k for k, p in enumerate(persons)}
    team_of = {p: team for _, team, p, _ in sets if team != "BALL"}
    team_of_slot = np.array([M.HOME if team_of[p] == home else M.AWAY for p in persons],
                            dtype=np.int8)

    players = np.full((n, len(persons), 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    gt = np.full(n, "X", dtype="<U1")
    for sec, team, person, r in sets:
        a = np.array(r)
        idx = offset[sec] + a[:, 0].astype(int) - first[sec]
        keep = (idx >= offset[sec]) & (idx < offset[sec] + length[sec])
        a, idx = a[keep], idx[keep]
        x01 = (a[:, 1] + DFL_L / 2) / DFL_L
        y01 = (a[:, 2] + DFL_W / 2) / DFL_W
        speed = a[:, 3] / 3.6           # DFL speed is km/h; the ATD's is m/s
        if team == "BALL":
            ball[idx, 0], ball[idx, 1], ball[idx, 2] = x01, y01, speed
            status, poss = a[:, 5].astype(int), a[:, 4].astype(int)
            lab = np.where(status == 0, "D", np.where(poss == 1, "H", np.where(poss == 2, "A", "D")))
            gt[idx] = lab
        else:
            k = slot[person]
            players[idx, k, 0], players[idx, k, 1], players[idx, k, 2] = x01, y01, speed

    half_blocks = np.array([(1, 0, length["firstHalf"] - 1),
                            (2, offset["secondHalf"], n - 1)])
    out = {"players": players, "ball": ball, "team_of_slot": team_of_slot, "gt": gt,
           "half_blocks": half_blocks}
    np.savez_compressed(cache, **out)
    return out


# ---------------------------------------------------------------- degradation

def challenge_profile():
    """Noise statistics of the challenge's own automatic tracking (an input, not an answer)."""
    cache = os.path.join(CACHE, "challenge_profile.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    d = load_data.load_all()
    g = load_data.graded_mask()
    bvis = np.isfinite(d["ball"][:, 0])

    def lengths(mask):
        s, e = M.run_bounds(mask)
        return e - s

    pres = np.isfinite(d["players"][:, :, 0])
    seg = np.concatenate([lengths(pres[:, j] & g) for j in range(pres.shape[1])])
    prof = {"hidden": lengths(~bvis & g), "visible": lengths(bvis & g),
            "player_seg": seg[seg > 0],
            "presence": np.array([pres[g].sum(1).mean() / 22.0])}
    os.makedirs(CACHE, exist_ok=True)
    np.savez(cache, **prof)
    return prof


def renewal(n, rng, on_lengths, off_lengths, start_on):
    """Alternate on/off stretches drawn from the two length samples."""
    mask = np.zeros(n, dtype=bool)
    i, on = 0, start_on
    while i < n:
        k = int(rng.choice(on_lengths if on else off_lengths))
        if on:
            mask[i:i + k] = True
        i += max(k, 1)
        on = not on
    return mask


def degrade(data, prof, seed):
    """Make clean DFL tracking look like the challenge's automatic tracking."""
    rng = np.random.default_rng(seed)
    players = data["players"].copy()
    ball = data["ball"].copy()
    n, m, _ = players.shape

    # ball: hidden/visible stretches with the challenge's own length distribution, split by
    # whether play is live.  A long outage is a stoppage in the challenge's labels (its
    # corrected tracking cannot bridge one), so during live play only the challenge's
    # shorter-than-10 s hidden stretches are drawn; during stoppages, any length.
    live = np.isin(data["gt"], ["H", "A"])
    short = prof["hidden"][prof["hidden"] < M.HIDDEN_LONG_FRAMES]
    vis_live = renewal(n, rng, prof["visible"], short, bool(rng.random() < 0.6))
    vis_dead = renewal(n, rng, prof["visible"], prof["hidden"], bool(rng.random() < 0.6))
    visible = np.where(live, vis_live, vis_dead)
    ball[~visible] = np.nan
    # ball glitches: the tracker briefly locks onto a player (~1% of visible frame pairs jump)
    starts = np.flatnonzero(visible & (rng.random(n) < 0.005))
    for s in starts:
        k = int(rng.integers(1, 26))
        e = min(n, s + k)
        j = int(rng.integers(m))
        ok = np.isfinite(players[s:e, j, 0])
        ball[s:e, 0][ok] = players[s:e, j, 0][ok]
        ball[s:e, 1][ok] = players[s:e, j, 1][ok]
    # No per-frame position noise: measured, the challenge ball is as smooth frame to frame
    # as the clean DFL ball (median |second difference| 0.011 m vs 0.010 m).  Its errors are
    # gaps and glitches, which are modelled above.

    # players: drop stretches so ~20.5 of 22 are tracked, segments as long as the challenge's
    seg = prof["player_seg"]
    presence = float(prof["presence"][0])
    mean_gap = max(1.0, seg.mean() * (1 - presence) / presence)
    gaps = np.maximum(1, rng.exponential(mean_gap, 5000).astype(int))
    for j in range(m):
        keep = renewal(n, rng, seg, gaps, True)
        players[~keep, j, :] = np.nan
    graded = data["gt"] != "X"
    out = {**data, "players": players.astype(np.float32), "ball": ball.astype(np.float32)}
    out["visible_share"] = np.array([np.isfinite(ball[graded, 0]).mean()])
    return out


# ---------------------------------------------------------------- running the model

FLAGS = ["USE_PHANTOM_MASK", "USE_BACKDATE", "USE_STRONG", "USE_RESTART_TEAM",
         "USE_DEAD_BACKDATE", "USE_HEADING", "USE_VITERBI",
         "USE_EXIT_BACKDATE", "USE_STOP_BACKDATE", "USE_SANDWICH_MERGE"]
PARAMS = {"SANDWICH_MAX_S": M.SANDWICH_MAX_S}
# every switch and parameter the configs touch, with its value in model.py: labels_for resets
# all of them before applying a config, so nothing leaks from one config to the next
DEFAULTS = {f: getattr(M, f) for f in FLAGS}
V13_FLAGS = ["USE_STOP_BACKDATE", "USE_SANDWICH_MERGE", "USE_EXIT_BACKDATE"]

V10 = {f: False for f in FLAGS}
V12 = {**DEFAULTS, **{f: False for f in V13_FLAGS}}
CONFIGS = [("v10", V10), ("v12", V12), ("v13 (current defaults)", DEFAULTS)]
for f in FLAGS:
    if f not in V13_FLAGS:
        CONFIGS.append(("v10 + " + f[4:].lower(), {**V10, f: True}))
for f in V13_FLAGS:
    CONFIGS.append(("v12 + " + f[4:].lower(), {**V12, f: True}))
for f in V13_FLAGS:
    if not DEFAULTS[f]:
        CONFIGS.append(("v13 + " + f[4:].lower(), {**DEFAULTS, f: True}))


def use_match(data):
    n = len(data["ball"])
    M.N_FRAMES = n
    M.HALF_BLOCKS = [tuple(int(v) for v in row) for row in data["half_blocks"]]
    M.L, M.W = DFL_L, DFL_W
    M.GK_MAX_X = 0.6 * DFL_L
    M.load_all = lambda: data
    M.compute_features = lambda: {"phase_team": np.full(n, -1), "setpiece_team": np.full(n, -1),
                                  "setpiece_start": np.zeros(n, dtype=bool)}


def labels_for(data, flags, signal_cache):
    for f, v in {**DEFAULTS, **PARAMS, **flags}.items():
        setattr(M, f, v)
    key = flags["USE_PHANTOM_MASK"]
    with contextlib.redirect_stdout(io.StringIO()):
        if key not in signal_cache:
            signal_cache[key] = M.build_signals()
        s = signal_cache[key]
        dead, sig = M.detect_dead(s)
        team = M.detect_possession(s, dead, sig)
    n = len(team)
    labels = np.full(n, "X", dtype="<U1")
    labels[~dead & (team == M.HOME)] = "H"
    labels[~dead & (team == M.AWAY)] = "A"
    labels[dead] = "D"
    graded = np.zeros(n, dtype=bool)
    for _, a, b in M.HALF_BLOCKS:
        graded[a:b + 1] = True
        seg = labels[a:b + 1]
        known = seg != "X"
        if known.any() and not known.all():
            idx = np.arange(len(seg))
            carry = np.maximum.accumulate(np.where(known, idx, -1))
            carry[carry < 0] = idx[known][0]
            labels[a:b + 1] = seg[carry]
    labels[~graded] = "X"
    return labels


def score(gt, labels):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels))
    return {"spell": r["spell_f1_mean"], "macro": r["macro_f1"],
            "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"],
            "team": r["team_acc_gt_alive"], "ad": r["alive_dead_f1"],
            "iou": [r["spell_f1"][i]["f1"] for i in pm.SPELL_IOUS]}


def check_labels(data):
    """Sanity: BallPossession 1 should be the home team -- the nearest player to the ball
    during in-play frames should mostly be from the team the label names."""
    p, b, t, gt = data["players"], data["ball"], data["team_of_slot"], data["gt"]
    with np.errstate(all="ignore"):
        d = np.hypot((p[:, :, 0] - b[:, None, 0]) * DFL_L, (p[:, :, 1] - b[:, None, 1]) * DFL_W)
    d = np.where(np.isfinite(d), d, np.inf)
    near = t[np.argmin(d, axis=1)]
    close = np.min(d, axis=1) < 1.0
    agree_h = np.mean(near[(gt == "H") & close] == M.HOME)
    agree_a = np.mean(near[(gt == "A") & close] == M.AWAY)
    return agree_h, agree_a


def all_match_ids():
    return sorted(re.search(r"(J03\w+)\.xml", f).group(1)
                  for f in glob.glob(os.path.join(DFL, "DFL_04_03_*.xml")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--clean", action="store_true", help="also score on clean DFL tracking")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ids = args.matches or all_match_ids()
    prof = challenge_profile()
    print("challenge noise profile: ball visible %.0f%% (median hidden %d frames), "
          "players tracked %.1f/22" % (
              100 * prof["visible"].sum() / (prof["visible"].sum() + prof["hidden"].sum()),
              np.median(prof["hidden"]), 22 * prof["presence"][0]))

    conditions = [("degraded", True)] + ([("clean", False)] if args.clean else [])
    results = {(c, name): [] for c, _ in conditions for name, _ in CONFIGS}
    for k, mid in enumerate(ids):
        data = parse_match(mid)
        ah, aa = check_labels(data)
        gt = data["gt"]
        print("%s: %d frames, %d possessions in the labels, label check H %.2f / A %.2f"
              % (mid, len(gt), pm.evaluate(pm.label_codes(gt), pm.label_codes(gt))["n_gt_spells"],
                 ah, aa))
        for cond, noisy in conditions:
            d = degrade(data, prof, args.seed * 1000 + k) if noisy else data
            use_match(d)
            cache = {}
            for name, flags in CONFIGS:
                results[(cond, name)].append(score(gt, labels_for(d, flags, cache)))

    for cond, _ in conditions:
        print("\n== %s tracking, mean over %d matches" % (cond, len(ids)))
        print("%-26s %8s %8s %8s %8s %12s   %s" % ("config", "spellF1", "macroF1", "teamAcc",
                                                   "aliveF1", "spells/true", "spellF1 per IoU"))
        base = np.mean([r["spell"] for r in results[(cond, "v10")]])
        for name, _ in CONFIGS:
            rs = results[(cond, name)]
            sp = np.mean([r["spell"] for r in rs])
            print("%-26s %8.4f %8.4f %8.3f %8.3f %6d/%-5d   %s   (%+.4f vs v10; better in %d/%d)" % (
                name, sp, np.mean([r["macro"] for r in rs]), np.mean([r["team"] for r in rs]),
                np.mean([r["ad"] for r in rs]), sum(r["n_pred"] for r in rs),
                sum(r["n_gt"] for r in rs),
                " ".join("%.3f" % v for v in np.mean([r["iou"] for r in rs], axis=0)),
                sp - base, sum(r["spell"] > b["spell"] + 1e-9 for r, b in
                               zip(rs, results[(cond, "v10")])), len(rs)))

    for f, v in {**DEFAULTS, **PARAMS}.items():
        setattr(M, f, v)


if __name__ == "__main__":
    main()
