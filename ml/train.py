"""Train and evaluate the hybrid possession model on other matches.

    python ml/train.py                 # leave-one-match-out evaluation + final model
    python ml/train.py --quick         # a few matches, fewer trees (smoke test)

Matches used (never the challenge match):
  dfl:<id>     7 Bundesliga matches, degraded to the challenge's noise profile exactly as in
               the test bench (validation/dfl_bench.py)
  metrica:<n>  Metrica sample games 1 and 2, as they come: Metrica is the challenge's tracking
               provider and its sample tracking is equally raw (validation/metrica.py)

Steps:
  1. Per match: run the rules model on it and build the per-frame features (cached).
  2. Leave-one-match-out: train on all other matches, predict P(H), P(A), P(D) on the
     held-out one.  Every score is on a match the model never saw.
  3. Choose the team switch cost on those held-out predictions (each match weighted
     equally).
  4. Report the hybrid against the rules model on the same matches, DFL and Metrica
     separately.
  5. Train the final model on all matches and save it with its settings to ml/artifacts/.

The only thing read from the challenge match is the noise profile of its automatic tracking
(an input, not an answer), used to degrade the DFL matches.

The features include the rules model's own signals and answers, and the cache in
other-games/ml-cache/ is keyed by match only: after any change to solution/model.py, delete
that folder before training.
"""

import argparse
import json
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import common as C
import dfl_bench as B
import metrica as MT

ART = os.path.join(C.HERE, "artifacts")
CACHE = os.path.join(C.ROOT, "other-games", "ml-cache")
COSTS_TEAM = (3, 5, 7, 10, 14, 20, 40)

# Model size, chosen on the held-out DFL + Metrica matches: 250/31 -> 500/63 helped on every
# measure; 800/127 did not help further (worse on Metrica).
TREES, LEAVES = 500, 63


def classifier(quick):
    return HistGradientBoostingClassifier(
        max_iter=60 if quick else TREES, learning_rate=0.1, max_leaf_nodes=LEAVES,
        min_samples_leaf=200, l2_regularization=1.0, early_stopping=False,
        random_state=C.SEED)


# ---------------------------------------------------------------- matches

def all_keys():
    keys = ["dfl:" + m for m in B.all_match_ids()]
    if os.path.isdir(MT.DATA):
        keys += ["metrica:" + g for g in MT.GAMES]
    return keys


def load(key, seed, prof):
    """The match as the model sees it (DFL degraded, Metrica as it comes) and its labels."""
    src, mid = key.split(":")
    if src == "dfl":
        data = B.parse_match(mid)
        k = B.all_match_ids().index(mid)            # same degradation seed as the bench
        return B.degrade(data, prof, seed * 1000 + k), data["gt"]
    data = MT.parse_game(mid)
    return data, data["gt"]


def match_features(key, seed, prof):
    """Full unswapped features (for predicting) plus a subsampled team-mirrored copy (extra
    training data).  Cached."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "feat_%s_seed%d.npz" % (key.replace(":", "_"), seed))
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    d, gt = load(key, seed, prof)
    B.use_match(d)
    s, dead, team = C.rules_outputs()
    X, names = C.feature_matrix(C.base_columns(s, dead, team))
    Xs, _ = C.feature_matrix(C.base_columns(s, dead, team, swap=True))
    out = {"X": X, "y": C.label_index(gt), "Xs": Xs[::C.STEP],
           "ys": C.label_index(gt, swap=True)[::C.STEP], "gt": gt,
           "names": np.array(names), "half_blocks": d["half_blocks"]}
    np.savez_compressed(path, **out)
    return out


def rules_labels(key, seed, prof):
    """The rules model alone, with its current defaults."""
    d, _ = load(key, seed, prof)
    B.use_match(d)
    return B.labels_for(d, B.DEFAULTS, {})


def training_rows(feats, keys):
    Xs, ys = [], []
    for key in keys:
        f = feats[key]
        for X, y in ((f["X"][::C.STEP], f["y"][::C.STEP]), (f["Xs"], f["ys"])):
            keep = y >= 0
            Xs.append(X[keep])
            ys.append(y[keep])
    return np.concatenate(Xs), np.concatenate(ys)


# ---------------------------------------------------------------- decoding & reporting

def rule_dead_of(f):
    names = [str(n) for n in f["names"]]
    return f["X"][:, names.index("rule_dead")] > 0.5


def decode_all(probs, feats, keys, ct):
    return {k: C.score(feats[k]["gt"],
                       C.decode_hybrid(probs[k], rule_dead_of(feats[k]),
                                       C.graded_runs_of(feats[k]["half_blocks"]), ct))
            for k in keys}


def mean_spell(res, keys):
    return float(np.mean([res[k]["spell"] for k in keys]))


def report(title, rows, keys):
    print("\n== %s (held out), mean over %d" % (title, len(keys)))
    print("%-20s %8s %8s %8s %8s %12s  %s" % ("model", "spellF1", "macroF1", "teamAcc",
                                               "aliveF1", "spells/true", "spellF1 per IoU"))
    base = rows[0][1]
    for name, res in rows:
        rs = [res[k] for k in keys]
        bs = [base[k] for k in keys]
        sp = np.mean([r["spell"] for r in rs])
        extra = "" if res is base else "  (%+.4f vs %s; better in %d/%d)" % (
            sp - np.mean([b["spell"] for b in bs]), rows[0][0],
            sum(r["spell"] > b["spell"] + 1e-9 for r, b in zip(rs, bs)), len(rs))
        print("%-20s %8.4f %8.4f %8.3f %8.3f %6d/%-5d %s%s" % (
            name, sp, np.mean([r["macro"] for r in rs]), np.mean([r["team"] for r in rs]),
            np.mean([r["ad"] for r in rs]), sum(r["n_pred"] for r in rs),
            sum(r["n_gt"] for r in rs),
            " ".join("%.3f" % v for v in np.mean([r["iou"] for r in rs], axis=0)), extra))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--retrain", action="store_true",
                    help="ignore cached held-out predictions and retrain every fold")
    ap.add_argument("--no-final", action="store_true", help="skip training the final model")
    args = ap.parse_args()

    keys = all_keys()
    if args.quick:
        keys = [k for k in keys if k.startswith("dfl")][:3] + \
               [k for k in keys if k.startswith("metrica")]
    dfl = [k for k in keys if k.startswith("dfl")]
    met = [k for k in keys if k.startswith("metrica")]
    prof = B.challenge_profile()
    tag = "all_seed%d%s_t%d_l%d" % (args.seed, "_quick" if args.quick else "", TREES, LEAVES)
    t0 = time.time()

    feats = {}
    for key in keys:
        feats[key] = match_features(key, args.seed, prof)
        print("features %-14s %d frames (%.0fs)" % (key, len(feats[key]["y"]), time.time() - t0))

    # leave-one-match-out (held-out predictions are cached: retraining is the slow part)
    probs = {}
    for key in keys:
        path = os.path.join(CACHE, "cv_%s_%s.npy" % (tag, key.replace(":", "_")))
        if os.path.exists(path) and not args.retrain:
            probs[key] = np.load(path)
            continue
        X, y = training_rows(feats, [k for k in keys if k != key])
        probs[key] = classifier(args.quick).fit(X, y).predict_proba(feats[key]["X"])
        np.save(path, probs[key])
        print("held out %-14s trained on %d rows (%.0fs)" % (key, len(y), time.time() - t0))

    # team switch cost, chosen on the held-out predictions
    grid = {ct: mean_spell(decode_all(probs, feats, keys, ct), keys) for ct in COSTS_TEAM}
    best = max(grid, key=grid.get)
    print("\nteam switch cost: mean held-out spellF1 over %d matches" % len(keys))
    for ct in COSTS_TEAM:
        print("  %3d  %.4f%s" % (ct, grid[ct], "  <- chosen" if ct == best else ""))

    rules = {k: C.score(feats[k]["gt"], rules_labels(k, args.seed, prof)) for k in keys}
    hyb = decode_all(probs, feats, keys, best)
    for title, sub in (("DFL matches, degraded", dfl), ("Metrica games", met)):
        if sub:
            report(title, [("rules", rules), ("hybrid", hyb)], sub)
    if args.no_final:
        return

    X, y = training_rows(feats, keys)
    clf = classifier(args.quick).fit(X, y)
    os.makedirs(ART, exist_ok=True)
    import joblib
    joblib.dump(clf, os.path.join(ART, "model.joblib"))
    with open(os.path.join(ART, "config.json"), "w") as handle:
        json.dump({"hybrid_cost_team": best, "trees": TREES, "leaves": LEAVES,
                   "seed": args.seed, "quick": args.quick, "matches": keys,
                   "features": [str(n) for n in feats[keys[0]]["names"]],
                   "heldout_spellF1_hybrid": grid[best]},
                  handle, indent=1)
    print("\nsaved the final model (trained on %d rows from %d matches) to %s (%.0fs total)" % (
        len(y), len(keys), ART, time.time() - t0))


if __name__ == "__main__":
    main()
