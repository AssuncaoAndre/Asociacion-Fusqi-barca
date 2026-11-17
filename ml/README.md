# Hybrid possession model

A learned model on top of the rules model in `solution/`. **The rules model decides in play
vs dead; the learned model decides which team has the ball.** It learns, from other matches,
the probability that each frame is `H`, `A` or `D`, then turns those probabilities into team
labels with Viterbi decoding, with the dead frames fixed by the rules.

The rules model in `solution/` is untouched; this folder only reads from it.

The code offers only the hybrid. A full learned model, deciding dead ball as well, was built
and evaluated first; its numbers are kept below because they are why the hybrid exists (see
"Why the hybrid").

## How it works

1. **Features.** For every frame, the model reuses the rules model's signals and answers:
   - the distance from the ball to each team's nearest player;
   - ball speed, and how fast it is speeding up or slowing down;
   - whether the ball can be seen, and how long since or until it is next seen;
   - how much the players are moving;
   - where the ball is heading;
   - the ball's position on the pitch;
   - the rules model's own team and dead-ball answers.

   On top of those it adds context: averages over 0.5 s, 2 s and 5 s around the frame, and the
   values 0.5 s and 1 s before and after. 83 features in all (`common.py`).
2. **Classifier.** Gradient boosting (scikit-learn's `HistGradientBoostingClassifier`, 500
   trees of up to 63 leaves) predicts P(H), P(A), P(D) for each frame. Each training match is
   also used with the two teams swapped, so the model treats home and away alike.
3. **Decoding.** Frames the rules model calls dead are fixed to `D`; the others can only be
   `H` or `A`. On its own, the most likely team per frame flickers: about 4 times too many
   possessions. Viterbi picks the team sequence that best fits the probabilities minus a cost
   of 14 for every change of team. That is the step that turns good per-frame guesses into
   good possessions.

## Training data

Only other matches. The rules allow data about other games. Nothing about the challenge match
is learned; the only thing read from it is how noisy its tracking is (an input, not an
answer).

| Source | Matches | How it is used | Labels |
|---|---|---|---|
| DFL open data (Bundesliga 2022/23) | 7 | degraded to look like the challenge tracking, exactly as in the test bench (`validation/README.md`) | the league's own per-frame `BallStatus` and `BallPossession` |
| Metrica sample games 1 and 2 | 2 | as they come: Metrica is the challenge's tracking provider, and its ball is visible about 60% of the time, like the challenge's 59% | built from the Metrica events by `validation/metrica.py`, an approximation |

Credits: DFL data is CC-BY 4.0 (credit the DFL; cite Bassek et al., *Scientific Data* 2025,
doi:10.1038/s41597-025-04505-y). Metrica asks to be acknowledged as the source.

## Results

Every score comes from a match the model never saw: train on all the other matches, test on
the held-out one, repeated for each of the 9 (leave-one-match-out). The switch cost and model
size were chosen on those held-out results too. "ML + Viterbi" is the full learned model,
evaluated but not shipped.

**DFL matches (7, degraded):**

| Model | spellF1 | macroF1 | Right team | In play vs dead F1 | Beats rules v10 in |
|---|---|---|---|---|---|
| Rules v10 | 0.327 | 0.774 | 75.8% | 0.911 | |
| Rules, current defaults (v12) | 0.331 | 0.780 | 75.2% | 0.915 | 4 of 7 |
| **ML + Viterbi** | **0.507** | **0.883** | **83.9%** | **0.982** | **7 of 7** |
| **Hybrid** | 0.440 | 0.815 | 80.7% | 0.916 | **7 of 7** |

**Metrica games (2), the closest stand-in for the challenge:**

| Model | spellF1 | macroF1 | Right team | In play vs dead F1 | Beats rules v10 in |
|---|---|---|---|---|---|
| Rules v10 | 0.532 | 0.735 | 71.8% | 0.829 | |
| Rules, current defaults (v12) | 0.531 | 0.745 | 70.8% | 0.837 | 1 of 2 |
| **ML + Viterbi** | **0.610** | **0.775** | **75.9%** | **0.860** | **2 of 2** |
| **Hybrid** | 0.590 | 0.761 | 73.1% | 0.838 | **2 of 2** |

On both sets the learned model clearly beats the rules model. Possessions found per true
possession also improved: on Metrica the hybrid finds 522 of 654.

### What we tried, and what it changed

All on the held-out matches above. "Full" is ML + Viterbi; "hybrid" as described.

| Change | DFL full / hybrid | Metrica full / hybrid | Kept? |
|---|---|---|---|
| First version: DFL only, 250 trees of 31 leaves | 0.491 / 0.428 | 0.591 / 0.550 (transfer) | |
| + Metrica games in the training data | 0.497 / 0.433 | 0.593 / 0.563 | yes |
| + bigger model, 500 trees of 63 leaves | **0.507 / 0.440** | **0.610 / 0.590** | **yes** |
| bigger still, 800 trees of 127 leaves | 0.505 / 0.446 | 0.601 / 0.585 | no: worse on Metrica |
| a separate team-only model for the hybrid | 0.497 / 0.421 | 0.593 / 0.561 | no: slightly worse |

## Why the hybrid

On the challenge match, the full ML model labels far fewer frames as dead than the rules model:
about 39% of graded frames against 47%. We can see this without the answers. Looking at the
frames the rules model calls dead, the difference is clear:

| On frames the rules model calls dead | Ball hidden | ML's average P(dead) |
|---|---|---|
| Challenge match | 77% | 0.69 |
| Metrica games | 83–91% | 0.63–0.73 |
| DFL matches (degraded) | 43–44% | 0.90 |

So the challenge's stoppages look like Metrica's: the ball is mostly out of the tracker's
sight. In those stretches, the DFL and Metrica labels describe what really happened, and play
often goes on out of sight. The challenge's official labels follow a different convention:
**a long gap in ball tracking is dead**, even when the video shows play continuing (see the
technical document). The learned model has never seen that convention; the rules model was
built around it.

So in-play vs dead is best left to the rules model, and the learned model should decide only
the team. The public 5 minutes agree (a pipeline check only; nothing was tuned on them):

| On the public 5 minutes | spellF1 | macroF1 | Dead-ball recall |
|---|---|---|---|
| Rules (v12, current submission) | 0.568 | 0.920 | 0.897 |
| ML + Viterbi | 0.566 | 0.703 | **0.250** |
| **Hybrid** | **0.607** | **0.922** | 0.897 |

With only 17 possessions in that window, its spellF1 differences are mostly noise. Different
versions of the hybrid scored 0.59–0.63 there. The decision rests on the held-out matches
and on the convention described above.

**Rules disclosure.** The public 5 minutes confirmed the dead-ball problem we had already seen
without them. If the hybrid is submitted, say this in the technical document.

## Run it

```bash
pip install -r requirements.txt -r ml/requirements.txt   # scikit-learn 1.9.1 is pinned

python ml/train.py            # evaluation + final model (~20 min from scratch)
python ml/predict.py          # -> ml/output/ml_hybrid.csv
python ml/predict.py "Asociación Fusqi.csv"   # regenerate the submission file (write anywhere)
```

Training needs the DFL and Metrica data in `other-games/` (download commands in
`validation/README.md`). Prediction needs the challenge data in `data/` and a trained model in
`ml/artifacts/`. It does not need the DFL or Metrica data.

Other options:

```bash
python ml/train.py --quick              # fewer matches and trees: a quick smoke test
python ml/train.py --retrain            # ignore cached held-out predictions
python ml/train.py --no-final           # evaluate only, keep the saved final model
```

Features and held-out predictions are cached in `other-games/ml-cache/`. Re-running after
changing only the decoding takes seconds; retraining all 9 folds takes ~10–20 min.

**The cache does not know about the rules model.** The features include the rules model's
signals and answers, and the hybrid takes its dead-ball mask from it, so after any change to
`solution/model.py` (the v13 changes of 2026-11-11 included) delete `other-games/ml-cache/`
and retrain. The saved model in `ml/artifacts/` and the numbers above were built on rules v12.

## Reproducibility

- **Training is deterministic.** Fixed seed, fixed degradation. Retraining from scratch gave
  byte-identical labels.
- **The saved model** (`ml/artifacts/model.joblib`) and its settings
  (`ml/artifacts/config.json`: switch cost, model size, feature list, training matches) are
  what `predict.py` uses.
- **The model file only loads in scikit-learn 1.9.1.** That is why the version is pinned.

## Submission state

This folder **is** the submitted state:

- `ml/artifacts/` (`model.joblib` and `config.json`) is **committed on purpose**, together with
  the submission CSV. The rules require the repository to regenerate the submitted file exactly;
  retraining may not match byte for byte on another machine, so the committed model is what makes
  the file reproducible, and it also spares reviewers the DFL and Metrica downloads.
- The submission CSV was regenerated from the saved model with
  `python ml/predict.py "Asociación Fusqi.csv"` and is byte-identical to that output.
- The technical document (`../Asociación Fusqi.md`) covers the approach, the training data
  (DFL CC-BY 4.0 and Metrica, both credited), the dead-ball convention that motivates the
  hybrid, and the disclosure that the public window confirmed the choice.

If you retrain later, commit the new artifacts together with the CSV regenerated from them,
never one without the other.

## Files

| File | What it does |
|---|---|
| `common.py` | features, team mirroring, the Viterbi decoder, the hybrid constraint, scoring |
| `train.py` | builds features for every match, runs leave-one-match-out, chooses the switch cost, trains the final model |
| `predict.py` | labels the challenge match with the saved model and the rules model's dead-ball answer |
| `artifacts/` | the trained model and its settings (committed on purpose — see "Submission state") |
| `output/` | predicted label files (gitignored) |
| `requirements.txt` | the extra pinned dependencies |
| `../validation/metrica.py` | loads the Metrica games and builds their labels from the events |
| `../validation/dfl_bench.py` | loads and degrades the DFL matches (also the rules-model test bench) |

## Ideas for the next step

- **Teach the learned model the dead-ball convention.** In the training labels, mark a long
  ball-tracking gap as dead, as the challenge's labels do. Then check on the held-out matches
  whether a full learned model, deciding dead ball too, becomes usable on its own.
- **Add Metrica sample game 3.** It is in the same file format as the challenge data (EPTS +
  JSON events), and every extra Metrica-style match strengthens the closest stand-in.
- **Try other per-frame models** (for example LightGBM, or a small sequence model). The
  features, decoding and evaluation stay the same.
