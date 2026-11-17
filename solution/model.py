"""The possession model, v12 (v11 + stoppage back-dating, ball heading; Viterbi
decoding built but OFF).  v11 = v10 + switch back-dating, strong single-touch
evidence, restart team from the last touch, phantom-track masking).

v10 additions in v11 (each behind a USE_* switch; see the v11 tunables):

Switch back-dating -- take-over evidence confirms a switch only after the
evidence window fills, so switches landed 11-150 frames late.  A confirmed
switch is now moved back to the frame the ball left the previous owner (the last
frame they were at the ball and nearer it than the gainer), up to 3 s back and
never past the previous change or a stoppage.

Strong evidence -- a hard strike or a driven ball killed dead at a player's
feet, with the opponent well clear, switches on 2 of 3 frames and cuts through
the post-switch lock.  Short spells (< ~0.3 s) were otherwise lost to the
7-frame window and the 15-frame lock.

Restart team -- a team clearly at the ball at the restart is taking it (the
observed taker comes first).  When nobody is clearly at the ball and the ball
went out, the restarting team is the opponent of the last toucher (or the
defending team when it went out across the goal mouth).

Phantom tracks -- two IDs locked onto one body (centroids < 0.25 m apart for a
full second) mask the newer ID; a goalkeeper ID more than 0.6 pitch-lengths from
its own goal line is masked.  Neither case teleports, so player_mask missed them.

v9 additions in v10:

Ball velocity units -- bvx/bvy were finite differences in m/FRAME (25x too
small), which made the ball-to-player relative speed ~the player's own speed and
kept the dribble arm ("they move with the ball") permanently off.  With the
per-second scale the dribble arm is live: a player running > 1.5 m/s with the
ball at their feet moving WITH them (relative speed < 1.5 m/s), clearly closer
to the ball than the opponent, is controlling it -- exactly the challenge's own
definition of a controlled gain.  The matching dead-ball exit (a live dribble
ends a stoppage) was measured and REMOVED: players keep the ball at their feet
throughout a break (hydration stops, walking the ball to a free-kick spot), and
the corrected tracking keeps the ball dead until the restart strike.

v8 additions in v9:

Player-track reliability (pmask) -- the automatic tracking occasionally re-locks
an ID onto a different player.  A real player cannot move more than ~0.4 m/frame,
so a larger per-frame jump is a teleport (a glitch), and a track segment that
begins after an absence and starts far from where the track was left (an average
speed over the gap no player can sustain) is a re-association onto another
object.  Teleport frames are masked.  A re-associated segment is masked only up
to its first internal teleport: the excursion in which the ID sits on the wrong
player ends with the track jumping back to the real one (a real player does not
teleport), so the mask is bounded by that jump.  A re-associated segment with NO
internal teleport is left alone -- the re-association may be the track finding
its real player after a glitch on the far side of the gap (e.g. a goalkeeper's
ID sitting on the wrong goal for a long stretch), and masking it would remove
the real player.  The masked positions are excluded from every ball-to-player
distance, so a phantom of the wrong team sitting on the ball cannot fire
take-over evidence (the v15650 false switch: a goalkeeper's ID duplicated onto
an attacker next to the ball the goalkeeper actually had).

An "arrival" take-over arm (the ball committed toward the new owner ~0.5 s
before it arrives -- the ground truth's own switch semantics, measured at every
public-window transition) was built and measured: it fires on ~28% of alive
frames (whenever the ball moves near any player the extrapolated landing point
is near an opponent) and cannot be separated from its false positives without
gutting it, while spellF1 is forgiving of switch LATENESS (a 230-frame-late
switch still overlaps a 1,000-frame spell at IoU ~0.78) but punishes false and
missed spells.  It was therefore dropped; the trap/kick/dribble arms plus the
sticky incumbent carry the transitions.

Two coupled state machines over the 199,319 frames.

Alive/dead (label D) -- two states:
  ALIVE    open play.
  DEAD     stoppage. Entered on sustained evidence: ball out of bounds; ball hidden
           while players are stopped (a box scramble is spared -- its players are
           visibly jostling); ball in flight (smoothed speed) and clear of everyone
           while players are stopped (a shot that has beaten the keeper and is
           leaving play); ball parked/isolated while players are stopped; the
           players frozen for several seconds with no live ball (covers a wandering
           wrong ball track during a stoppage); or -- the catch-all for a tracking
           outage -- the raw ball unseen for a long stretch (>= HIDDEN_LONG_FRAMES).
           A slow build-up is NOT a stoppage: the ball is visibly moving with a
           dribbling controller, so the freeze catch-all stays off.  DEAD ends only
           when play genuinely resumes: a real strike (the ball at kick speed, in
           bounds, either freshly accelerated out of rest or gaining several m/s
           over the last second after a carry -- the ball drifting back in from
           out of bounds at 5-7 m/s is fast but gains speed only gradually, and
           does NOT end a stoppage; nor does a strike while the ball is still out
           of bounds, nor a gentle strike within ~2 s of the ball having been
           out -- that is the ball coming back in, retrieved from the net after
           a goal or rolled back in at the byline, while a set piece taken from
           the lines is a HARD, freshly accelerated strike). A stoppage is never
           exited by the ball merely sitting near
           a player while the players mill about (that is how a goal kick waiting
           to be taken looks), nor by a player dribbling or juggling it during the
           break -- only a genuine strike ends it.

"Players stopped" is judged on the ATD speed field (a sprint detector: it reads ~0
whenever nobody exceeds 3 m/s). A second, slower channel -- the fraction of
players above 1.5 m/s -- additionally spares the jostling of a box scramble, which
the sprint-only channel would read as "everyone stopped".

The long-hidden rule exists because the corrected tracking cannot bridge a long
ball outage: when the raw ball is unseen for ~10 s or more, the tracking marks the
ball dead (it only bridges short gaps whose reappearance matches the ball's
constant-velocity continuation). 10.0 s is anchored to the Laws of the Game
(see HIDDEN_LONG_FRAMES).  Disclosure: the value was earlier checked against the
public window (longest GT-alive hidden stretch ~9.3 s); see the technical document.

Possession (H/A) -- the corrected tracking's owner behaves like a sticky possession
state, not a nearest-player rule: during a loose ball or a scramble the owner stays
with the last team that controlled it, even when an opponent is physically closest.
So possession changes only when the other team actively *gains* the ball, on
evidence smoothed over a short window: a trap (a ball ARRIVING with speed rapidly
slows at their feet -- a ball that was already slow rolling to a halt near them is
a loose ball, not a gain), a kick (the ball is struck from their feet), or a
dribble (they move with the ball).  Passive proximity -- the ball merely sitting
near the opponent on a loose ball -- is deliberately NOT take-over evidence.  The
gainer must be clearly the closest team (MARGIN) -- a loose ball skimming past an
opponent is not a take-over. While the ball is loose, in flight or hidden,
possession stays with the incumbent. When play restarts after a stoppage, the
restarting team -- the team that strikes or throws the ball back into play -- is
the first controller: the tracking's owner starts the new phase with them. The
smart-tagging phase prior is deliberately NOT used to switch possession (it tags
phases of play, not ball ownership, and contradicts the stickiness).

Ball reliability -- the automatic ball track teleports when the tracker swaps
objects. A real ball cannot move more than ~1.2 m/frame, so larger jumps mark the
observation unreliable; the Kalman filter rejects jump observations rather than
re-initialising on them, so a wrong track keeps the ball "hidden" until the real
ball comes back near the predicted position.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load_data import N_FRAMES, HALF_BLOCKS, PITCH, load_all
from features import compute_features, HOME, AWAY

L, W = PITCH
FPS = 25.0

# ---------------------------------------------------------------- tunables (domain-set)
CTRL_R = 1.8        # m -- ball this close to a player is "at their feet"
MARGIN = 0.7        # m -- a take-over needs the ball clearly closer to the taker
SLOW_V = 1.5        # m/s -- below this a controlled ball is stationary
FLIGHT_V = 7.0      # m/s -- above this the ball is in flight (pass/shot/clearance)
KICK_V = 7.0        # m/s -- a restart kick (a real ball strike out of a stoppage);
                    # high enough that the ball drifting back from out-of-bounds
                    # (~6.6 m/s) does NOT count as a kick, low enough to catch a
                    # gently struck goal kick (~7.2 m/s)
KICK_FRAMES = 3     # sustained frames above KICK_V that count as a kick
KICK_ACC = 3.0      # m/s gained over the last 0.16 s that makes it a real strike --
                    # a genuine kick accelerates the ball out of (near) rest; a ball
                    # drifting back in from out of bounds gains speed only gradually
                    # and must NOT end a stoppage
KICK_GAIN_FRAMES = 25  # ~1.0 s window for the longer-horizon strike test
KICK_GAIN = 4.0     # m/s gained over the last second that makes a fast ball a real
                    # strike: tracking smears a gentle restart over several frames,
                    # so a carry-then-strike gains ~4-7 m/s over 1 s while an
                    # in-flight ball gains ~0
KICK_OOB_GUARD_FRAMES = 50  # ~2.0 s: a strike this soon after the ball was out of
                    # bounds is suspect -- only a HARD strike counts (a set piece
                    # taken from the corner arc or the six-yard box, freshly
                    # accelerated); the gentle gain of a ball rolling back in at
                    # the byline (returned by a ball boy) does not
GLITCH_D = 2.0      # m/frame -- above this the raw ball observation teleported
GLITCH_V = 40.0     # m/s -- above this the raw ball speed is a glitch
ACT_ON = 0.08       # smoothed fraction of players > 3 m/s that counts as "active"
ACT15_ON = 0.30     # smoothed fraction of players > 1.5 m/s (jostling in a scramble)
PMAX_ON = 4.0       # fastest player above this also counts as "active"
OOB_MARGIN = 0.5    # m beyond the lines that counts as out of bounds

# freeze catch-all: players essentially motionless for this long with no live ball
FREEZE_ACT = 0.05   # smoothed fraction of players > 3 m/s
FREEZE_PMAX = 2.5   # m/s -- fastest player below this = nobody is really moving
FREEZE_FRAMES = 75  # ~3.0 s of sustained freeze

# long-hidden catch-all: the raw ball unseen for this many consecutive frames.
# Football anchor: the longest the ball may legitimately be held -- and so hidden
# against a body -- in open play is a goalkeeper's hold, capped by Law 12.2 at
# 6 s when this match was played (8 s since July 2025).  10 s = that cap plus
# ~2 s for the release to be picked up again by the tracking.
HIDDEN_LONG_FRAMES = 250  # ~10.0 s of the raw ball being unseen -> DEAD

ENTER_STOP_FRAMES = 30    # ~1.2 s: ball hidden + players stopped -> DEAD
ENTER_ISOL_FRAMES = 70    # ~2.8 s: ball isolated/parked + players stopped -> DEAD
ENTER_FLIGHT_FRAMES = 50  # ~2.0 s: ball in flight, isolated, players stopped -> DEAD
                    # (a long pass in open play flies ~1 s with everyone watching;
                    # only a shot leaving play stays this long)
ENTER_OOB_FRAMES = 5      # sustained OOB -> DEAD almost immediately
# take-over (possession change) evidence: a trap / kick / dribble by the
# opponent -- an ACTIVE gain, not passive proximity -- smoothed over a short window
TOV_WIN = 7               # ~0.28 s smoothing window for take-over evidence
TOV_MIN = 0.30            # fraction of the window with evidence that switches
TRAP_ARRIVE_V = 3.5       # m/s -- a trap is a ball ARRIVING with speed and being
                          # stopped at a player's feet; a ball that was already slow
                          # rolling to a halt near a player is a loose ball, not a
                          # take-over (the sticky owner keeps it)
LOCK_FRAMES = 15          # ~0.6 s after a switch during which no further switch
MIN_DEAD_S = 1.0
MIN_ALIVE_S = 1.2

# player-track glitches: teleports and re-associations onto another player
GLITCH_P_D = 2.0     # m/frame -- above this a player track teleported
REASSOC_GAP = 2      # frames of absence before a segment start is a re-association
REASSOC_V = 0.45     # m/frame -- average speed over the gap above this is
                     # impossible for a player (11.25 m/s), so the track did not
                     # travel the jump: it was re-locked onto another object
REASSOC_OFF = 3.0    # m -- offset of the re-association jump threshold

# ---------------------------------------------------------------- v11 additions
# Each v11 rule has an on/off switch so it can be ablated on its own.  None of
# these constants was read off the public window: they come from the physics of
# the game (body width, pitch and goal dimensions, pass flight times).

# (1) switch back-dating: a confirmed take-over is moved back to the frame the
# ball left the previous owner's foot, instead of the frame the evidence window
# filled up.
USE_BACKDATE = True
BACKDATE_MAX = 75    # ~3.0 s: the longest pass flight we will reach back across

# (2) strong single-touch evidence: an unmistakable kick or trap switches on a
# 2-of-3-frame window and overrides the post-switch lock, so a short spell (a
# one-touch interception that is lost again) is not swallowed.
USE_STRONG = True
STRONG_D = 1.5            # m -- ball unmistakably at the gainer's feet
STRONG_MARGIN = 1.4       # m -- and the gainer far clearer than the opponent (2 x MARGIN)
STRONG_KICK_V = 10.0      # m/s -- a hard strike
STRONG_KICK_ACC = 6.0     # m/s gained within 0.16 s
STRONG_TRAP_ARRIVE_V = 8.0  # m/s -- a driven ball arriving
STRONG_TRAP_DEC = 3.0     # m/s lost within 0.2 s -- killed dead
STRONG_WIN = 3            # frames (causal window)
STRONG_MIN = 2            # of which this many carry evidence

# (3) restart team from the last touch: when the ball went out of play, the
# opponent of the last toucher restarts (throw-in, corner, goal kick); when it
# went out across the goal mouth (a goal, or a shot over the bar) the defending
# team restarts (kick-off or goal kick).  Other stoppages (fouls, offside) fall
# back to the nearest-team rule at the restart.
USE_RESTART_TEAM = True
GOAL_HALF_W = 3.66        # m -- half the goal width (7.32 m)
RESTART_OOB_LOOKBACK = 25  # frames before a dead run in which its OOB may begin
RESTART_OOB_WIN = 50      # frames into a dead run in which the OOB must appear
LAST_TOUCH_WIN = 125      # ~5 s before going out in which to find the last touch

# (4) phantom tracks that do not teleport: two IDs sitting on one body, and a
# goalkeeper ID far from its own goal.
USE_PHANTOM_MASK = True

# (5) stoppage back-dating: the ball goes out or vanishes before the players
# react, so every stoppage used to be detected late.  A stoppage that starts
# because the ball went out, or vanished while the players stopped, is moved back
# to the frame the ball crossed the line or disappeared.  (The 10 s long-hidden
# entry stays a point entry: a hidden ball can still be in play for ~9 s.)
USE_DEAD_BACKDATE = True

# (6) heading: when a switch is back-dated, it starts at the first frame after
# the ball left the old owner on which the ball is heading to the new team (the
# nearest player to where the ball will be 0.5 s and 1 s ahead is theirs).
USE_HEADING = True
HEADING_T = (0.5, 1.0)    # s ahead

# (7) Viterbi decoding of possession: instead of the threshold state machine,
# choose, for each stretch of play, the H/A sequence that best agrees with the
# per-frame gain evidence (trap/kick/dribble) minus a fixed cost per switch.  A
# brief touch by the other team that is followed by the holder's own play is
# then not a switch (switching there and back costs twice).
# OFF: measured worse.  The evidence only marks GAINS (trap/kick/dribble); a team
# quietly keeping the ball produces none, so the decoder hands its stretches to
# whichever team has more gain evidence nearby.  Viterbi needs per-frame scores
# that also cover holding the ball -- i.e. learned probabilities.
USE_VITERBI = False
SWITCH_COST = TOV_MIN * TOV_WIN   # = 2.1 frames of evidence: the same bar the
                                  # state machine sets for one switch
DUP_R = 0.25              # m -- two real bodies cannot be this close ...
DUP_FRAMES = 25           # ... for a full second
GK_MAX_X = 0.6 * L        # m -- a keeper this far from their own goal line is a wrong ID


# ---------------------------------------------------------------- helpers

def runs_of(mask):
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def run_bounds(mask):
    """Vectorised runs_of: (starts, ends) of the True runs, ends exclusive."""
    m = np.asarray(mask, dtype=np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def smooth_mask(mask, min_true_s, min_false_s):
    mask = mask.copy()
    min_true = int(min_true_s * FPS)
    min_false = int(min_false_s * FPS)
    changed = True
    while changed:
        changed = False
        for a, b in runs_of(mask):
            if b - a < min_true:
                mask[a:b] = False
                changed = True
        for a, b in runs_of(~mask):
            if b - a < min_false:
                mask[a:b] = True
                changed = True
    return mask


def rolling_mean(x, k):
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.nan_to_num(x, nan=0.0)
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return sliding_window_view(pad, k).mean(axis=1)


def runs_ge(mask, k):
    """True at i when mask is True for the k frames ending at i (inclusive)."""
    m = np.asarray(mask, dtype=bool)
    n = len(m)
    idx = np.flatnonzero(~m)
    pos = np.arange(n)
    last = np.searchsorted(idx, pos, side="right") - 1
    lf = np.where(last >= 0, idx[np.maximum(last, 0)], -1)
    runlen = pos - lf
    return m & (runlen >= k)


def shift(x, k):
    """x shifted forward by k frames (x[i-k]); the first k entries are NaN."""
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


def frames_since(mask):
    """Frames since the last True in mask (inf where mask was never True)."""
    idx = np.flatnonzero(mask)
    pos = np.arange(len(mask))
    last = np.searchsorted(idx, pos, side="right") - 1
    out = np.full(len(mask), np.inf)
    ok = last >= 0
    out[ok] = pos[ok] - idx[last[ok]]
    return out


def sliding_fraction(mask, k):
    """Fraction of True in the window of k frames ending at i (inclusive)."""
    from numpy.lib.stride_tricks import sliding_window_view
    m = np.asarray(mask, dtype=float)
    pad = np.pad(m, (k - 1, 0), mode="constant")
    return sliding_window_view(pad, k).mean(axis=1)


# ---------------------------------------------------------------- ball recovery (jump-rejecting)

def kalman_ball(bx, by, bvis, fps=FPS, obs_noise=0.35, accel_noise=6.0):
    """Constant-velocity Kalman filter. Jump observations are REJECTED (treated as
    missing), not re-initialised on, so a tracker identity switch keeps the ball
    hidden instead of gluing the filter to the wrong object."""
    n = len(bx)
    rx = np.full(n, np.nan)
    ry = np.full(n, np.nan)
    rvx = np.full(n, np.nan)
    rvy = np.full(n, np.nan)
    runc = np.full(n, np.inf)

    dt = 1.0 / fps
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    R = np.eye(2) * obs_noise ** 2
    Q = np.zeros((4, 4))
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
                pred = F @ x
                innov = z - H @ pred
                S = H @ P @ H.T + R
                # reject observations too far from the prediction (tracker glitch)
                if np.sqrt(innov @ np.linalg.solve(S, innov)) > 8.0:
                    x = pred
                    P = F @ P @ F.T + Q
                    rx[i] = np.nan  # hidden this frame
                    ry[i] = np.nan
                    runc[i] = np.inf
                    continue
                K = P @ H.T @ np.linalg.inv(S)
                x = pred + K @ innov
                P = (np.eye(4) - K @ H) @ P
        else:
            if init:
                x = F @ x
                P = F @ P @ F.T + Q
            else:
                continue
        rx[i] = x[0]
        ry[i] = x[1]
        rvx[i] = x[2]
        rvy[i] = x[3]
        runc[i] = np.sqrt(P[0, 0] + P[1, 1])
    return rx, ry, rvx, rvy, runc


# ---------------------------------------------------------------- player-track reliability

def player_mask(px, py):
    """Reliability mask for the player tracks.  True where the position can be
    trusted for ball-to-player distances.

    A real player cannot move more than ~0.4 m/frame, so a larger per-frame jump
    is a teleport (the tracker glitched) and the landing frame is unreliable.
    A track segment that begins after an absence and starts far from where the
    track was last seen -- an average speed over the gap no player can sustain --
    is a re-association onto a different object.  Such a segment is masked only
    up to its first internal teleport: the excursion in which the ID sits on the
    wrong player ends with the track jumping back to the real one (a real player
    does not teleport), so the mask is bounded by that jump.  A re-associated
    segment with no internal teleport is NOT masked: the re-association may be
    the track finding its real player after a glitch on the far side of the gap
    (a goalkeeper's ID can sit on the wrong goal for a long stretch), and
    masking it would remove the real player from the game.
    """
    n, m = px.shape
    pres = np.isfinite(px) & np.isfinite(py)
    pmask = pres.copy()
    d = np.full((n, m), np.inf)
    d[1:] = np.hypot(px[1:] - px[:-1], py[1:] - py[:-1])
    d[~pres] = np.inf
    tele = pres & (d > GLITCH_P_D)
    pmask &= ~tele
    for j in range(m):
        prev_end = 0
        for a, b in runs_of(pres[:, j]):
            if a > 0 and a - prev_end >= REASSOC_GAP:
                gap = a - prev_end
                jump = np.hypot(px[a, j] - px[prev_end - 1, j],
                                py[a, j] - py[prev_end - 1, j])
                if jump > gap * REASSOC_V + REASSOC_OFF:
                    t = np.flatnonzero(tele[a:b, j])
                    if len(t):
                        pmask[a:a + int(t[0]) + 1, j] = False
            prev_end = b
    return pmask


def half_ranges(n):
    """(lo, hi) frame ranges of the two halves (split between the graded blocks)."""
    split = (HALF_BLOCKS[0][2] + HALF_BLOCKS[1][1]) // 2
    return [(0, split), (split, n)]


def goal_sides(px, team_of_slot):
    """Per half and team: (own goal-line x, goalkeeper slot).  The keeper is the
    regularly present slot whose median x sits closest to either goal line; that
    line is the team's own goal for the half."""
    out = {}
    for h, (lo, hi) in enumerate(half_ranges(len(px))):
        for team in (HOME, AWAY):
            slots = np.flatnonzero(team_of_slot == team)
            xs = px[lo:hi][:, slots]
            presence = np.isfinite(xs).mean(axis=0)
            cand = presence > 0.3
            if not cand.any():
                continue
            with np.errstate(all="ignore"):
                med = np.nanmedian(xs[:, cand], axis=0)
            edge = np.minimum(med, L - med)
            k = int(np.nanargmin(edge))
            gk = int(slots[cand][k])
            out[(h, team)] = (0.0 if med[k] < L / 2 else L, gk)
    return out


def phantom_mask(px, py, team_of_slot, sides):
    """Unreliable player positions that do NOT teleport (player_mask misses them).

    Duplicates: two IDs on one body.  Two real players' centroids cannot stay
    within DUP_R of each other for a full second, so one of them is a copy; the
    copy is the ID whose current track segment began more recently (it is the
    one that was re-locked onto the other's body).  Ties are left alone.

    Wrong-goal keeper: a goalkeeper ID more than GK_MAX_X from its own goal line
    is sitting on someone else."""
    n, m = px.shape
    pres = np.isfinite(px) & np.isfinite(py)
    bad = np.zeros((n, m), dtype=bool)

    seg_start = np.zeros((n, m), dtype=np.int64)
    for j in range(m):
        for a, b in zip(*run_bounds(pres[:, j])):
            seg_start[a:b, j] = a

    for i in range(m):
        for j in range(i + 1, m):
            both = pres[:, i] & pres[:, j]
            if both.sum() < DUP_FRAMES:
                continue
            with np.errstate(all="ignore"):
                close = both & (np.hypot(px[:, i] - px[:, j], py[:, i] - py[:, j]) < DUP_R)
            if close.sum() < DUP_FRAMES:
                continue
            for a, b in zip(*run_bounds(close)):
                if b - a < DUP_FRAMES or seg_start[a, i] == seg_start[a, j]:
                    continue
                newer = i if seg_start[a, i] > seg_start[a, j] else j
                bad[a:b, newer] = True

    for h, (lo, hi) in enumerate(half_ranges(n)):
        for team in (HOME, AWAY):
            if (h, team) not in sides:
                continue
            goal_x, gk = sides[(h, team)]
            x = px[lo:hi, gk]
            with np.errstate(all="ignore"):
                bad[lo:hi, gk] |= np.isfinite(x) & (np.abs(x - goal_x) > GK_MAX_X)
    return bad


# ---------------------------------------------------------------- geometry & features

def build_signals():
    """Compute every per-frame signal the two state machines need."""
    data = load_all()
    players = data["players"]
    ball = data["ball"]
    team_of_slot = data["team_of_slot"]

    px = players[:, :, 0] * L
    py = players[:, :, 1] * W
    pspeed = players[:, :, 2]
    bx = ball[:, 0] * L
    by = ball[:, 1] * W
    bvis = ~np.isnan(ball[:, 0])

    # player-track reliability: glitched tracks are excluded from every
    # ball-to-player distance (see player_mask)
    pmask = player_mask(px, py)
    sides = goal_sides(px, team_of_slot)
    if USE_PHANTOM_MASK:
        pmask &= ~phantom_mask(px, py, team_of_slot, sides)

    # raw ball speed & displacement (glitch detection)
    bspeed_raw = np.full(N_FRAMES, np.nan)
    bspeed_raw[1:] = np.hypot(np.diff(bx), np.diff(by)) * FPS
    disp = np.full(N_FRAMES, 0.0)
    disp[1:] = np.hypot(np.diff(bx), np.diff(by))
    bvalid = bvis & (disp <= GLITCH_D) & (bspeed_raw < GLITCH_V)

    # ball recovery
    rx, ry, rvx, rvy, runc = kalman_ball(bx, by, bvis)
    trust = np.isfinite(rx) & (runc < 8.0)
    # effective ball position: raw where valid, recovered where trustworthy
    ex = np.where(bvalid, bx, np.where(trust, rx, np.nan))
    ey = np.where(bvalid, by, np.where(trust, ry, np.nan))
    bseen = ~np.isnan(ex)
    # effective ball speed & velocity (m/s; the finite difference is per frame,
    # so it is scaled by FPS)
    bspd = np.where(bvalid, bspeed_raw, np.nan)
    bvx = np.where(bvalid, np.gradient(np.where(bvalid, bx, np.nan)) * FPS, np.nan)
    bvy = np.where(bvalid, np.gradient(np.where(bvalid, by, np.nan)) * FPS, np.nan)
    # fill velocity from the Kalman where the raw ball is invalid
    bspd = np.where(bseen & np.isnan(bspd), np.hypot(rvx, rvy), bspd)
    bvx = np.where(bseen & np.isnan(bvx), rvx, bvx)
    bvy = np.where(bseen & np.isnan(bvy), rvy, bvy)
    bspd = np.where(bseen, bspd, np.nan)

    oob = bvalid & ((bx < -OOB_MARGIN) | (bx > L + OOB_MARGIN) |
                    (by < -OOB_MARGIN) | (by > W + OOB_MARGIN))
    # the ball over a line at all (no margin): where a ball that ends up out
    # actually crossed
    crossed = bvalid & ((bx < 0) | (bx > L) | (by < 0) | (by > W))

    # distances from effective ball to every player
    with np.errstate(all="ignore"):
        dist = np.hypot(px - ex[:, None], py - ey[:, None])
    dist = np.where(pmask, dist, np.inf)
    dall = dist.min(axis=1)
    home_slots = team_of_slot == HOME
    away_slots = team_of_slot == AWAY
    dH = dist[:, home_slots].min(axis=1)
    dA = dist[:, away_slots].min(axis=1)
    # where the ball is heading: the nearest team to the ball's straight-line
    # position HEADING_T seconds ahead (kept on the pitch).  +1 = home nearer at
    # every horizon, -1 = away nearer at every horizon, 0 = mixed / unknown.
    heading = np.zeros(N_FRAMES, dtype=np.int8)
    votes = []
    for t in HEADING_T:
        qx = np.clip(ex + bvx * t, 0.0, L)
        qy = np.clip(ey + bvy * t, 0.0, W)
        with np.errstate(all="ignore"):
            dq = np.hypot(px - qx[:, None], py - qy[:, None])
        dq = np.where(pmask, dq, np.inf)
        qH = dq[:, home_slots].min(axis=1)
        qA = dq[:, away_slots].min(axis=1)
        votes.append(np.where(np.isfinite(qx) & (qH < qA), 1,
                              np.where(np.isfinite(qx) & (qA < qH), -1, 0)))
        del dq
    votes = np.array(votes)
    heading[(votes == 1).all(axis=0)] = 1
    heading[(votes == -1).all(axis=0)] = -1

    # nearest player of each team: index, speed, velocity
    def nearest_idx(dist_team):
        idx = np.argmin(dist_team, axis=1)
        rows = np.arange(N_FRAMES)
        return idx, dist_team[rows, idx]

    hidx, _ = nearest_idx(dist[:, home_slots])
    aidx, _ = nearest_idx(dist[:, away_slots])
    hslot = np.flatnonzero(home_slots)
    aslot = np.flatnonzero(away_slots)
    rows = np.arange(N_FRAMES)
    sp_H = pspeed[rows, hslot[hidx]]
    sp_A = pspeed[rows, aslot[aidx]]

    # player velocities (finite differences, m/s)
    def pvel(p):
        v = np.full_like(p, np.nan)
        v[1:] = (p[1:] - p[:-1]) * FPS
        return v
    pvx = pvel(px)
    pvy = pvel(py)
    pvx_H = pvx[rows, hslot[hidx]]
    pvy_H = pvy[rows, hslot[hidx]]
    pvx_A = pvx[rows, aslot[aidx]]
    pvy_A = pvy[rows, aslot[aidx]]

    # relative speed between ball and nearest player of each team
    rel_H = np.hypot(bvx - pvx_H, bvy - pvy_H)
    rel_A = np.hypot(bvx - pvx_A, bvy - pvy_A)

    # nearest player overall: speed
    near_is_H = dH <= dA
    sp_near = np.where(near_is_H, sp_H, sp_A)

    # player activity.  act3 (>3 m/s) catches sprints; act15 (>1.5 m/s) also catches
    # the slow jostling of a box scramble, which act3 alone reads as "everyone
    # stopped" and would mis-flag as a stoppage.
    present = ~np.isnan(pspeed)
    ps = np.where(present, pspeed, np.nan)
    with np.errstate(all="ignore"):
        act3 = np.nanmean(np.where(present, pspeed > 3.0, np.nan), axis=1)
        act15 = np.nanmean(np.where(present, pspeed > 1.5, np.nan), axis=1)
        pmax = np.nanmax(ps, axis=1)
    act_sm = rolling_mean(np.nan_to_num(act3, nan=0.0), 25)
    act15_sm = rolling_mean(np.nan_to_num(act15, nan=0.0), 25)
    pmax_sm = rolling_mean(np.nan_to_num(np.nan_to_num(pmax, nan=0.0), nan=0.0), 25)

    # "players active": sprints (act), slow jostling (act15 -- spares a box
    # scramble from the ball-hidden entry), or a clearly moving fastest player.
    active = (act_sm > ACT_ON) | (act15_sm > ACT15_ON) | (pmax_sm > PMAX_ON)

    # smoothed ball speed and its recent change (shared by both state machines)
    bspd_sm = rolling_mean(bspd, 5)
    bspd_dec5 = shift(bspd_sm, 5) - bspd_sm   # + = ball has slowed over the last 0.2 s
    bspd_acc4 = bspd_sm - shift(bspd_sm, 4)   # + = ball has sped up over the last 0.16 s
    bspd_prev5 = shift(bspd_sm, 5)            # smoothed speed 0.2 s ago

    # smart tagging
    f = compute_features()

    return {
        "ex": ex, "ey": ey, "bseen": bseen, "bspd": bspd, "bvx": bvx, "bvy": bvy,
        "oob": oob, "bvalid": bvalid, "bvis_raw": bvis,
        "dall": dall, "dH": dH, "dA": dA,
        "sp_H": sp_H, "sp_A": sp_A, "rel_H": rel_H, "rel_A": rel_A,
        "sp_near": sp_near,
        "act_sm": act_sm, "act15_sm": act15_sm, "pmax_sm": pmax_sm, "active": active,
        "bspd_sm": bspd_sm, "bspd_dec5": bspd_dec5, "bspd_acc4": bspd_acc4,
        "bspd_prev5": bspd_prev5,
        "phase_team": f["phase_team"], "setpiece_team": f["setpiece_team"],
        "setpiece_start": f["setpiece_start"],
        "sides": sides, "crossed": crossed, "heading": heading,
    }


# ---------------------------------------------------------------- alive/dead state machine

ALIVE, DEAD, RESTART = 0, 1, 2


def detect_dead(s, entry_log=None):
    bseen = s["bseen"]
    bspd = s["bspd"]
    bvalid = s["bvalid"]
    oob = s["oob"]
    dall = s["dall"]
    act_sm = s["act_sm"]
    act15_sm = s["act15_sm"]
    pmax_sm = s["pmax_sm"]
    active = s["active"]
    bvis_raw = s["bvis_raw"]
    sp_near = s["sp_near"]

    controlled = bseen & (dall < CTRL_R)
    stationary = bseen & (bspd < SLOW_V)
    inflight = bseen & (bspd > FLIGHT_V)
    isolated = bseen & (dall > 4.0)

    # Two activity notions.  `active` (incl. act15 jostling) spares a box scramble
    # from the ball-hidden entry.  `active3` (sprints/fast player only) is used for
    # the in-flight entry: a shot that has beaten the keeper and is flying out of
    # play leaves the players momentarily stopped, which act15 would read as
    # "active" and block the entry.
    active3 = (act_sm > ACT_ON) | (pmax_sm > PMAX_ON)

    # a player clearly moving with the ball (a live dribble).  Keeps the freeze
    # catch-all off during a slow build-up, where the controller is visibly
    # dribbling even though the rest of the players are standing still.
    dribbling = bseen & (dall < 2.0) & (sp_near > 2.0) & (bspd > 1.5)
    dribble_sm = rolling_mean(dribbling.astype(float), 25)

    # entry evidence
    e_oob = oob
    e_hidden_stop = (~bseen) & (~active)                          # ball hidden, players stopped
    e_parked = (~active) & (isolated | (controlled & stationary))  # ball alone / parked, stopped
    # ball in flight, clear of everyone, players stopped: a shot/clearance leaving
    # play.  Uses the SMOOTHED speed -- a ball in flight does not drop below flight
    # speed for a single noisy frame, and the raw speed dips mid-flight.
    e_flight_stop = bseen & (s["bspd_sm"] > FLIGHT_V) & (dall > 3.0) & (~active3)
    # freeze catch-all: players essentially motionless for a long stretch and the
    # ball is not being dribbled.  Catches a wandering wrong ball track during a
    # stoppage; a slow build-up is spared (its controller is dribbling).
    e_freeze = (act_sm < FREEZE_ACT) & (pmax_sm < FREEZE_PMAX) & (dribble_sm < 0.15)
    # long-hidden catch-all: the raw ball has been unseen for >= HIDDEN_LONG_FRAMES
    # consecutive frames (see HIDDEN_LONG_FRAMES above).  The corrected tracking
    # bridges only short outages whose reappearance matches the ball's
    # constant-velocity continuation; a long outage is a dead ball, whatever the
    # players are doing.
    e_long_hidden = runs_ge(~bvis_raw, HIDDEN_LONG_FRAMES)

    # exit evidence.  A stoppage ends only when play genuinely resumes: a real
    # strike.  A strike is a VALID ball observation at kick speed for several
    # frames that is either freshly accelerated out of rest (a genuine strike
    # gains speed within 0.16 s) or gaining several m/s over the last second (a
    # gentle restart follows a dribble/carry, so the ball is already moving and
    # barely accelerating over 0.16 s, but it gains ~4-7 m/s over 1 s).  The ball
    # drifting back in from out of bounds at 5-7 m/s is fast but gains speed only
    # gradually, so it does NOT count; nor does a strike while the ball is still
    # out of bounds (the ball retrieved from the net after a goal appears out of
    # bounds at speed and dies away); nor a GENTLE strike within
    # KICK_OOB_GUARD_FRAMES of the ball having been out of bounds -- that is the
    # ball rolling back in at the byline.  A genuine restart is struck from in
    # bounds, and a set piece taken from the lines (the ball marginally out at
    # the corner arc or the six-yard box moments before) is a HARD, freshly
    # accelerated strike.  Nor does an in-flight ball count (a shot gains
    # nothing -- it is already flying); nor a phantom Kalman track (not a valid
    # raw observation).  A live dribble does NOT end a stoppage either: players
    # keep the ball at their feet (or keepy-uppy it) throughout a break -- a
    # hydration stop, a player walking the ball to a free-kick spot -- and the
    # corrected tracking keeps the ball dead until the restart strike.
    kick_run = runs_ge(bseen & (bspd > KICK_V), KICK_FRAMES)
    bspd_gain = s["bspd_sm"] - shift(s["bspd_sm"], KICK_GAIN_FRAMES)
    since_oob = frames_since(oob)
    near_oob = since_oob <= KICK_OOB_GUARD_FRAMES
    kick_strike = (kick_run & bvalid & ~oob
                   & (~near_oob | (s["bspd_acc4"] > KICK_ACC))
                   & ((s["bspd_acc4"] > KICK_ACC) | (bspd_gain > KICK_GAIN)))

    crossed = s["crossed"]
    dead = np.zeros(N_FRAMES, dtype=bool)
    state = ALIVE
    last_exit = -1
    oob_n = stop_n = isol_n = flight_n = freeze_n = 0
    for i in range(N_FRAMES):
        if state == ALIVE:
            oob_n = oob_n + 1 if e_oob[i] else 0
            stop_n = stop_n + 1 if e_hidden_stop[i] else 0
            isol_n = isol_n + 1 if e_parked[i] else 0
            flight_n = flight_n + 1 if e_flight_stop[i] else 0
            freeze_n = freeze_n + 1 if e_freeze[i] else 0
            reason = None
            if oob_n >= ENTER_OOB_FRAMES:
                state = DEAD
                dead[max(0, i - ENTER_OOB_FRAMES + 1):i + 1] = True
                reason = "oob"
            elif stop_n >= ENTER_STOP_FRAMES:
                state = DEAD
                dead[max(0, i - ENTER_STOP_FRAMES + 1):i + 1] = True
                reason = "hidden_stop"
            elif flight_n >= ENTER_FLIGHT_FRAMES:
                state = DEAD
                dead[max(0, i - ENTER_FLIGHT_FRAMES + 1):i + 1] = True
                reason = "flight_stop"
            elif isol_n >= ENTER_ISOL_FRAMES:
                state = DEAD
                dead[max(0, i - ENTER_ISOL_FRAMES + 1):i + 1] = True
                reason = "parked"
            elif freeze_n >= FREEZE_FRAMES:
                state = DEAD
                dead[max(0, i - FREEZE_FRAMES + 1):i + 1] = True
                reason = "freeze"
            elif e_long_hidden[i]:
                # point entry: the frame the ball has been hidden for
                # HIDDEN_LONG_FRAMES straight is the first dead frame.  The
                # preceding hidden stretch is NOT marked retroactively -- the
                # tracking bridges short hidden stretches, and the longest it
                # bridges (~9.3 s) can still be alive.
                state = DEAD
                dead[i] = True
                reason = "long_hidden"
            if state == DEAD:
                oob_n = stop_n = isol_n = flight_n = freeze_n = 0
                if USE_DEAD_BACKDATE and reason in ("oob", "hidden_stop", "flight_stop"):
                    # the stoppage began when the ball crossed the line or
                    # disappeared, not when the players reacted: walk back over
                    # the frames just before the entry on which the ball was
                    # over a line or unseen, never past the last restart
                    j = i
                    while j - 1 > last_exit and dead[j - 1]:
                        j -= 1   # to the start of the entry window just marked
                    while j - 1 > last_exit and (crossed[j - 1] or not bvis_raw[j - 1]):
                        j -= 1
                    dead[j:i + 1] = True
                if entry_log is not None:
                    entry_log.append((i, reason))
        else:  # DEAD -- play is stopped; it resumes on a real strike
            dead[i] = True
            if kick_strike[i]:
                state = ALIVE
                last_exit = i

    dead = smooth_mask(dead, MIN_DEAD_S, MIN_ALIVE_S)
    sig = {"controlled": controlled, "stationary": stationary,
           "inflight": inflight, "isolated": isolated}
    return dead, sig


# ---------------------------------------------------------------- restart team (last touch)

def restart_teams(s, dead):
    """{first alive frame after a stoppage: restarting team} for stoppages caused
    by the ball going out of play.

    Out across the goal mouth (between the posts, beyond the goal line): a goal,
    or a shot over the bar -- either way the DEFENDING team restarts (kick-off or
    goal kick).  Out anywhere else: the opponent of the team that last touched
    it (throw-in, corner, goal kick).  The last touch is the most recent frame
    before the ball went out with a player at the ball (within CTRL_R) and one
    team clearly nearer (MARGIN); a contested last touch is left unknown.

    Stoppages with no ball out of play (fouls, offside, injuries) are not in the
    result; the caller falls back to the nearest team at the restart."""
    oob, ex, ey, bseen = s["oob"], s["ex"], s["ey"], s["bseen"]
    dH, dA, sides = s["dH"], s["dA"], s["sides"]
    halves = half_ranges(N_FRAMES)
    out = {}
    for a, b in zip(*run_bounds(dead)):
        if b >= N_FRAMES:
            continue
        lo = max(0, a - RESTART_OOB_LOOKBACK)
        hi = min(b, a + RESTART_OOB_WIN)
        f = np.flatnonzero(oob[lo:hi])
        if not len(f):
            continue
        f0 = lo + int(f[0])
        x, y = ex[f0], ey[f0]
        if (x < 0 or x > L) and abs(y - W / 2) < GOAL_HALF_W + OOB_MARGIN:
            h = 0 if f0 < halves[1][0] else 1
            goal_x = 0.0 if x < 0 else L
            for team in (HOME, AWAY):
                if sides.get((h, team), (None,))[0] == goal_x:
                    out[b] = team
            continue
        t0 = max(0, f0 - LAST_TOUCH_WIN)
        dmin = np.minimum(dH[t0:f0], dA[t0:f0])
        near = np.flatnonzero(bseen[t0:f0] & (dmin < CTRL_R))
        if not len(near):
            continue
        t = t0 + int(near[-1])
        if dH[t] < dA[t] - MARGIN:
            out[b] = AWAY
        elif dA[t] < dH[t] - MARGIN:
            out[b] = HOME
    return out


# ---------------------------------------------------------------- Viterbi possession decoder

def viterbi_possession(r_H, r_A, alive, prior, cost):
    """For each stretch of play, the H/A sequence maximising
        sum(evidence for the team in possession) - cost * (number of switches),
    starting with a bonus of one switch cost for the restarting team (prior).
    Returns (team array, switches as (frame, new, old, previous change))."""
    team = np.full(N_FRAMES, -1, dtype=np.int8)
    switches = []
    r = (r_H, r_A)          # indexed by team: HOME = 0, AWAY = 1
    for a, b in zip(*run_bounds(alive)):
        n = b - a
        came_over = np.zeros((n, 2), dtype=bool)   # True = arrived by switching
        p = prior.get(a, -1)
        v = [(cost if p == t else 0.0) + r[t][a] for t in (0, 1)]
        for k in range(1, n):
            i = a + k
            nv = [0.0, 0.0]
            for t in (0, 1):
                stay, switch = v[t], v[1 - t] - cost
                if switch > stay:
                    came_over[k, t] = True
                    nv[t] = switch + r[t][i]
                else:
                    nv[t] = stay + r[t][i]
            v = nv
        if v[0] == v[1]:
            st = p if p >= 0 else 0   # no evidence either way: the restarting team
        else:
            st = 0 if v[0] > v[1] else 1
        for k in range(n - 1, -1, -1):
            team[a + k] = st
            if k > 0 and came_over[k, st]:
                st = 1 - st
        changes = np.flatnonzero(team[a + 1:b] != team[a:b - 1]) + a + 1
        prev = a
        for c in changes:
            switches.append((int(c), int(team[c]), int(team[c - 1]), prev))
            prev = int(c)
    return team, switches


# ---------------------------------------------------------------- possession state machine

def detect_possession(s, dead, sig):
    dH, dA = s["dH"], s["dA"]
    bspd = s["bspd"]
    bseen = s["bseen"]
    rel_H, rel_A = s["rel_H"], s["rel_A"]
    sp_H, sp_A = s["sp_H"], s["sp_A"]
    bspd_sm = s["bspd_sm"]
    bspd_dec5 = s["bspd_dec5"]
    bspd_acc4 = s["bspd_acc4"]
    bspd_prev5 = s["bspd_prev5"]

    def takeover_evidence(dX, dO, rel_X, sp_X):
        """Evidence that team X is actively GAINING control of the ball: a trap
        (a ball ARRIVING with speed rapidly slows at X's feet), a kick (ball
        struck from X's feet), or a dribble (X moving with the ball).  X must be
        clearly the closest team (MARGIN) -- a loose ball skimming past an
        opponent is not a take-over.  Passive proximity ("the ball is near X")
        is deliberately NOT evidence: on a loose ball the opponent is often
        physically closest while the corrected tracking still owns it for the
        incumbent."""
        trap = (bseen & (dX < 2.0) & (bspd_sm < 6.0) & (bspd_dec5 > 0.8)
                & (bspd_prev5 > TRAP_ARRIVE_V) & (dX < dO - MARGIN))
        kick = (bseen & (dX < 2.0) & (bspd > 8.0) & (bspd_acc4 > 4.0)
                & (dX < dO - MARGIN))
        drib = (bseen & (dX < 1.8) & (bspd_sm > 2.0) & (bspd_sm < 7.0)
                & (rel_X < 1.5) & (sp_X > 1.5) & (dX < dO - MARGIN))
        return trap | kick | drib

    with np.errstate(all="ignore"):
        tov_H = takeover_evidence(dH, dA, rel_H, sp_H)
        tov_A = takeover_evidence(dA, dH, rel_A, sp_A)
    tov_H_eff = rolling_mean(tov_H.astype(float), TOV_WIN) >= TOV_MIN
    tov_A_eff = rolling_mean(tov_A.astype(float), TOV_WIN) >= TOV_MIN

    def strong_evidence(dX, dO):
        """An unmistakable gain: a hard strike or a driven ball killed dead, with
        the ball right at X's feet and the opponent well clear."""
        clear = bseen & (dX < STRONG_D) & (dX < dO - STRONG_MARGIN)
        kick = clear & (bspd > STRONG_KICK_V) & (bspd_acc4 > STRONG_KICK_ACC)
        trap = (clear & (bspd_prev5 > STRONG_TRAP_ARRIVE_V)
                & (bspd_dec5 > STRONG_TRAP_DEC) & (bspd_sm < 4.0))
        return kick | trap

    if USE_STRONG:
        with np.errstate(all="ignore"):
            st_H = strong_evidence(dH, dA)
            st_A = strong_evidence(dA, dH)
        need = (STRONG_MIN - 0.5) / STRONG_WIN
        str_H_eff = sliding_fraction(st_H, STRONG_WIN) >= need
        str_A_eff = sliding_fraction(st_A, STRONG_WIN) >= need
    else:
        str_H_eff = str_A_eff = np.zeros(N_FRAMES, dtype=bool)

    restart = restart_teams(s, dead) if USE_RESTART_TEAM else {}

    team = np.full(N_FRAMES, -1, dtype=np.int8)
    poss = -1
    lock = 0
    last_change = 0
    switches = []   # (frame, new team, old team, earliest frame it may reach back to)
    alive = ~dead

    def other(t):
        return AWAY if t == HOME else HOME

    for i in range(N_FRAMES):
        if not alive[i]:
            # the ball is dead: possession resets; the restart sets it fresh
            poss = -1
            lock = 0
            continue
        if poss < 0 and (i == 0 or not alive[i - 1]):
            last_change = i
            # play is restarting.  The team that strikes/throws the ball back in
            # is the one clearly nearest it at the restart -- an observed taker.
            # Only when nobody is clearly at the ball do we fall back on the
            # inferred restart team from the last touch (restart_teams): on the
            # full match the taker was clearly visible in 18 of the 20 restarts
            # where the two disagreed, so observation beats inference.
            lo = max(0, i - 2)
            if dH[lo:i + 1].min() < dA[lo:i + 1].min() - MARGIN:
                poss = HOME
                team[i] = poss
                lock = LOCK_FRAMES
                continue
            if dA[lo:i + 1].min() < dH[lo:i + 1].min() - MARGIN:
                poss = AWAY
                team[i] = poss
                lock = LOCK_FRAMES
                continue
            if i in restart:
                poss = restart[i]
                team[i] = poss
                lock = LOCK_FRAMES
                continue
        strong_other = poss >= 0 and (str_A_eff[i] if poss == HOME else str_H_eff[i])
        if lock > 0 and not strong_other:
            lock -= 1
            if poss >= 0:
                team[i] = poss
            continue
        if poss < 0:
            # no incumbent: the first team with take-over evidence takes it
            if tov_H_eff[i] or str_H_eff[i]:
                poss = HOME
            elif tov_A_eff[i] or str_A_eff[i]:
                poss = AWAY
            if poss >= 0:
                team[i] = poss
                lock = LOCK_FRAMES
                last_change = i
            continue
        # incumbent: switch only on confirmed take-over evidence by the other team
        # (or an unmistakable kick/trap, which also cuts through the lock)
        if strong_other or (tov_A_eff[i] if poss == HOME else tov_H_eff[i]):
            switches.append((i, other(poss), poss, last_change))
            poss = other(poss)
            lock = LOCK_FRAMES
            last_change = i
        team[i] = poss

    if USE_VITERBI:
        # replace the state machine's answer: same evidence, same restart rule,
        # decoded as a whole sequence with a cost per switch
        prior = {}
        for a in run_bounds(alive)[0]:
            lo = max(0, a - 2)
            if dH[lo:a + 1].min() < dA[lo:a + 1].min() - MARGIN:
                prior[a] = HOME
            elif dA[lo:a + 1].min() < dH[lo:a + 1].min() - MARGIN:
                prior[a] = AWAY
            elif a in restart:
                prior[a] = restart[a]
        team, switches = viterbi_possession(tov_H.astype(float), tov_A.astype(float),
                                            alive, prior, SWITCH_COST)

    if USE_BACKDATE:
        # Move each switch back to the frame the ball left the previous owner:
        # the last frame, within BACKDATE_MAX and after the previous change, on
        # which the old team was at the ball (within CTRL_R) and nearer it than
        # the new team.  With USE_HEADING the switch then starts at the first
        # later frame on which the ball is heading to the new team.  Nothing is
        # moved when no such frame exists (a loose ball nobody held), and never
        # across a dead frame (the previous change bounds it).
        heading = s["heading"]
        for i, new, old, bound in switches:
            b0 = max(bound + 1, i - BACKDATE_MAX)
            if b0 >= i:
                continue
            d_old = dH if old == HOME else dA
            d_new = dH if new == HOME else dA
            held = bseen[b0:i] & (d_old[b0:i] < CTRL_R) & (d_old[b0:i] < d_new[b0:i])
            idx = np.flatnonzero(held)
            if len(idx):
                start = b0 + int(idx[-1]) + 1
                if USE_HEADING:
                    toward = np.flatnonzero(heading[start:i] == (1 if new == HOME else -1))
                    if len(toward):
                        start += int(toward[0])
                team[start:i] = new

    # fill leading unknowns inside alive stretches with the following team
    for i in range(N_FRAMES):
        if alive[i] and team[i] < 0:
            j = i
            while j < N_FRAMES and alive[j] and team[j] < 0:
                j += 1
            if j < N_FRAMES and team[j] >= 0:
                team[i:j] = team[j]
    return team


# ---------------------------------------------------------------- assembly

def build_labels():
    s = build_signals()
    dead, sig = detect_dead(s)
    team = detect_possession(s, dead, sig)

    labels = np.full(N_FRAMES, "X", dtype="<U1")
    alive = ~dead
    labels[alive & (team == HOME)] = "H"
    labels[alive & (team == AWAY)] = "A"
    labels[dead] = "D"

    graded = np.zeros(N_FRAMES, dtype=bool)
    for _, a, b in HALF_BLOCKS:
        graded[a:b + 1] = True

    # alive frames still without a team inside a graded half: carry the nearest
    # known team (the scorer forward-fills X anyway; doing it here also protects
    # macroF1, the tie-break)
    for _, a, b in HALF_BLOCKS:
        seg = labels[a:b + 1]
        unknown = (seg != "H") & (seg != "A") & (seg != "D")
        if unknown.any():
            idx = np.arange(len(seg))
            known = ~unknown
            if known.any():
                carry = np.maximum.accumulate(np.where(known, idx, -1))
                carry[carry < 0] = idx[known][0]
                filled = seg[carry]
                # only replace unknowns that are not D
                seg = np.where(unknown, filled, seg)
                labels[a:b + 1] = seg

    # second half is graded from 110808 but kickoff is at 113742: all D before it
    labels[110808:113742] = "D"
    # ungraded frames stay X
    labels[~graded] = "X"
    return labels, dead, team, s, sig


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path


if __name__ == "__main__":
    labels, dead, team, s, sig = build_labels()
    # default: the evaluation folder; pass a path to write the submission file
    # directly, e.g.  python solution/model.py my_team.csv
    out = (sys.argv[1] if len(sys.argv) > 1 else
           os.path.join(HERE, "..", "evaluation", "results", "v12_model.csv"))
    write_submission(out, labels)
    counts = {c: int(np.sum(labels == c)) for c in "HADX"}
    print("counts:", counts)
    print("wrote", out)
