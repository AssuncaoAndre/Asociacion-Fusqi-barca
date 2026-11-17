"""Label the challenge match (FC Barcelona vs Athletic Club) with the hybrid model.

Usage
-----
    python ml/predict.py                                       # -> ml/output/ml_hybrid.csv
    python ml/predict.py "Asociación Fusqi.csv"   # regenerate the submission file

The output is a submission file in the organisers' format: a `frame,label` header and one
row per frame of the tactical video (199,319 rows), each label H / A / D / X.  Check it with
    python evaluation/validate_submission.py <file>

The hybrid
----------
The rules model (solution/model.py) decides in play vs dead; the learned model decides which
team has the ball while play is live.  The rules model was built around the challenge's
dead-ball convention (a long ball-tracking gap is dead), which the learned model's training
labels do not follow.  See ml/README.md for the numbers.

What it needs
-------------
- The challenge data in data/ (or $MTAH_DATA_DIR), exactly as for solution/model.py.
- A trained model in ml/artifacts/: model.joblib and config.json, written by
  `python ml/train.py`.
- The rules model the saved model was trained with: its signals and answers are the learned
  model's inputs.  config.json records which training matches and features were used.
- scikit-learn 1.9.1 (ml/requirements.txt): the saved model only loads in that version.
It does NOT need the DFL or Metrica data; those are only for training.

What it does
------------
1. Runs the rules model on the challenge match to get its per-frame signals and its own
   dead-ball and team answers.
2. Builds the same features as in training (ml/common.py) and checks they match the
   feature list saved with the model.
3. Predicts P(H), P(A), P(D) for every frame.
4. Decodes them with Viterbi: frames the rules call dead are fixed to D, the others can only
   be H or A, and every change of team costs the switch cost chosen in training.
5. Applies the same match-specific framing as the rules model: D before the second-half
   kick-off, X outside the two graded halves.

The output is deterministic: the same data and the same saved model give the same file,
byte for byte.
"""

import json
import os
import sys

import joblib
import numpy as np

import common as C
from common import M

ART = os.path.join(C.HERE, "artifacts")      # trained model + its settings
OUT = os.path.join(C.HERE, "output")         # default output folder (gitignored)

# The second half is graded from frame 110808, but the kick-off is at 113742: the ball is
# dead in between.  Same constant, for the same reason, as in solution/model.py.
SECOND_HALF_GRADED, SECOND_HALF_KICKOFF = 110808, 113742


def load_model():
    """The trained classifier and its settings (the team switch cost chosen in training and
    the exact list of feature names the model was trained on)."""
    config_path = os.path.join(ART, "config.json")
    model_path = os.path.join(ART, "model.joblib")
    if not (os.path.exists(config_path) and os.path.exists(model_path)):
        raise FileNotFoundError("no trained model in %s -- run: python ml/train.py" % ART)
    with open(config_path) as handle:
        cfg = json.load(handle)
    return joblib.load(model_path), cfg


def predict_labels():
    """Labels for every frame of the challenge match: a numpy array of 'H' / 'A' / 'D' / 'X',
    and the saved model settings."""
    clf, cfg = load_model()

    # 1. the rules model's signals and answers (they are inputs to the learned model)
    s, dead, team = C.rules_outputs()

    # 2. the same features as in training, in the same order
    X, names = C.feature_matrix(C.base_columns(s, dead, team))
    if list(names) != cfg["features"]:
        raise RuntimeError("features differ from the ones the model was trained on -- "
                           "retrain with: python ml/train.py")

    # 3-4. P(H), P(A), P(D) per frame, decoded with in play vs dead fixed by the rules
    graded_runs = C.graded_runs_of(M.HALF_BLOCKS)
    labels = C.decode_hybrid(clf.predict_proba(X), dead, graded_runs, cfg["hybrid_cost_team"])

    # 5. match framing: dead before the second-half kick-off, X outside the graded halves
    labels[SECOND_HALF_GRADED:SECOND_HALF_KICKOFF] = "D"
    graded = np.zeros(len(labels), dtype=bool)
    for a, b in graded_runs:
        graded[a:b] = True
    labels[~graded] = "X"
    return labels, cfg


def main():
    """Command line: an optional output path."""
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "ml_hybrid.csv")

    labels, cfg = predict_labels()
    M.write_submission(out, labels)

    print("hybrid: in play vs dead from the rules model, team switch cost %s"
          % cfg["hybrid_cost_team"])
    print("counts:", {c: int(np.sum(labels == c)) for c in "HADX"})
    print("wrote", out)


if __name__ == "__main__":
    main()
