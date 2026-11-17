# Who has the ball? — technical document

**Team:** Asociación Fusqi
**Challenge:** More than a Hack 2027, possession labelling (FC Barcelona vs Athletic Club, LaLiga 2024–25)
**Submission file:** `Asociación Fusqi.csv` — one label per frame of the tactical video (`frame,label`, 199,319 rows, labels `H` / `A` / `D` / `X`)
**Code:** this repository. Section "Reproduce" shows the exact commands.

---

## What we did, in short

For every frame we answer two questions: **is the ball in play?** and **if yes, which team has it?**

1. **Clean the tracking.** The automatic tracking sometimes jumps (ball or player IDs teleport). A real ball/player cannot move that fast in one frame, so those jumps are ignored and filled in from the recent path.
2. **Rules model (v12, `solution/model.py`).** Hand-written rules over the tracking decide in play vs dead (`D`), and give a first guess of the team. No training, no labels.
3. **Train a small ML model on other matches** (`ml/train.py`). It learns, per frame, the probability of `H`, `A`, `D` from simple signals (distance from the ball to each team's nearest player, ball speed, whether the ball is visible, how much players move, plus averages over the last/next few seconds).
4. **Combine them (the hybrid, `ml/predict.py`).** The rules model decides in play vs dead; the ML model decides only **which team** has the ball while play is live. A Viterbi decoder then picks the team sequence that best fits the probabilities, with a penalty for every change of team, so the labels do not flicker.
5. **Write the CSV.** `ml/predict.py` writes the submission file directly.

Why the hybrid and not the ML model alone: the challenge's official labels call a frame **dead whenever the ball tracking has a long gap**, even if the video shows play continuing. The ML model's training labels (from other leagues) do not follow that convention, so on this match it calls far too little dead ball. The rules model was built around that convention, so we let it keep the dead/alive decision. On the public 5 minutes the full ML model gets dead-ball recall 0.25; the hybrid keeps the rules model's 0.897.

---

## How it works

### Is the ball in play? (rules model)

Each half starts in play. A stoppage is believed only after a sign has lasted a while:

| Sign | Must last |
|---|---|
| Ball outside the lines | 0.2 s |
| Ball not visible **and** players stopped | 1.2 s |
| Ball flying away from everyone **and** players stopped | 2 s |
| Ball still or alone **and** players stopped | 2.8 s |
| Nobody really moving | 3 s |
| Ball not seen at all | 10 s |

The 10 s comes from the Laws of the Game (a goalkeeper may hold the ball 6 s; we add time for the tracker to see it again). A stoppage is moved back to the moment the ball went out or disappeared. Play restarts only when someone clearly kicks the ball.

### Which team has the ball? (ML model on top of the rules)

The rules model's own answer is sticky: a team keeps the ball until the other team clearly wins it — **trap** (ball arrives fast and is stopped at their feet), **kick**, or **dribble** (running with the ball), with that player clearly closer to the ball than any opponent. Loose balls, deflections and headers do not change possession. After a stoppage, the restarting team starts with the ball.

The ML model reuses those signals and answers as inputs and learns P(H), P(A), P(D) per frame with gradient boosting (scikit-learn, 500 trees). Frames the rules call dead are fixed to `D`; on the rest, Viterbi picks the H/A sequence that best fits the probabilities minus a cost of 14 per team change. That cost was chosen on held-out matches, not on the challenge.

---

## Training data

**Nothing from the provided match was used to train or tune anything.** The only thing read from it is how noisy its tracking is (an input, used to make the DFL data look equally noisy). The video was watched only to understand the data and debug the pipeline — nothing seen was turned into a label, a correction, or a rule tied to a specific moment of the match. The public 5-minute answers were used only to check that the pipeline works — see the disclosure below.

| Source | Matches | Labels | How used |
|---|---|---|---|
| DFL open data (Bundesliga 2022/23, CC-BY 4.0) | 7 | the league's own per-frame `BallStatus` / `BallPossession` | degraded to the challenge's noise profile, then used for training and testing |
| Metrica sample games 1 and 2 | 2 | built from the Metrica events by `validation/metrica.py` (an approximation) | used as they come — Metrica is the challenge's tracking provider, so its raw data is the closest stand-in |

Credits: DFL data is CC-BY 4.0 — credit the DFL; cite Bassek et al., *Scientific Data* 2025, doi:10.1038/s41597-025-04505-y. Metrica sample data: "Please acknowledge the source" — Metrica Sports.

Every score we quote is on a match the model never saw (leave-one-match-out over the 9 matches):

| Model | DFL spellF1 (7 matches) | Metrica spellF1 (2 games) |
|---|---|---|
| Rules v12 | 0.331 | 0.531 |
| ML alone | 0.507 | 0.610 |
| **Hybrid (submitted)** | **0.440** | **0.590** |

The ML alone scores higher on those matches, but on the challenge match it calls far too little dead ball (see above), so the hybrid is what we submit.

**Disclosure about the public 5 minutes.** The rules say they are for checking the pipeline, not for tuning. We used them only to confirm the dead-ball problem we had already seen without them (the full ML model labels only 25% of the true dead frames there; the hybrid keeps the rules model's 89.7%). No number in the pipeline was set from those answers. On the public window the submitted file scores **spellF1 0.607, macroF1 0.922** (baseline 0.243); with only 17 possessions there, those differences are noise.

---

## Possession rules — where we drew the line

- A team keeps the ball until the other team **clearly wins it**: trap, kick or dribble, with their player at the ball and at least 0.7 m closer than any opponent, seen on several frames within ~0.25 s.
- **Loose balls do not change possession** — being nearest to a loose ball is not enough; a ball in the air after a pass stays with the team that kicked it.
- **Restarts** (throw-in, corner, goal kick, free kick) go to the team with a player clearly at the ball; if nobody is, to the opponent of the last toucher (or the defending team after a goal/shot over the bar).
- **Dead ball**: the signs in the table above; a long gap in ball tracking (10 s) is dead, matching the challenge's own convention.

---

## Limitations

- Long gaps in ball tracking are marked dead even when the video shows play continuing — the official labels do the same, because the corrected tracking cannot fill those gaps either.
- Restarts after fouls and offsides go to the team nearest the ball; the tracking does not say who was penalised.
- The ML model has never seen the challenge's dead-ball convention, which is why dead/alive is left to the rules model.
- The tracking sets the ceiling: when the raw positions are wrong in a way the clean-up does not catch, the labels are wrong too.

---

## The code — what each file does

| File | What it does |
|---|---|
| `solution/load_data.py` | reads the tracking (ATD), smart tagging and homography; caches the parse |
| `solution/features.py` | reads the smart tagging (parsed, deliberately not used for labels) |
| `solution/model.py` | **the rules model (v12)**: cleans the tracking, decides in play vs dead, sticky possession team |
| `solution/score.py` | scores a CSV on the public 5 minutes, with detail |
| `solution/sanity.py` | whole-match sanity report (label shares, dead runs) |
| `ml/common.py` | **shared ML pieces**: builds the per-frame features, the Viterbi decoder, the hybrid constraint, scoring |
| `ml/train.py` | **training code**: builds features for the 9 other matches, leave-one-match-out evaluation, picks the switch cost, trains and saves the final model to `ml/artifacts/` |
| `ml/predict.py` | **submission code**: loads the saved model, runs the rules model on the challenge match, predicts P(H/A/D), decodes the hybrid, writes the submission CSV |
| `ml/artifacts/model.joblib` | the trained model (committed on purpose — with the CSV it makes the file reproducible) |
| `ml/artifacts/config.json` | the model's settings: switch cost, feature list, training matches |
| `ml/requirements.txt` | extra pinned dependencies for the ML part (scikit-learn 1.9.1) |
| `validation/dfl_bench.py` | loads the 7 DFL matches and degrades them to the challenge's noise profile (also the test bench) |
| `validation/metrica.py` | loads the 2 Metrica sample games and builds their labels from the events |
| `evaluation/` | the organisers' scorer, format validator and the public 5-minute ground truth |
| `games.py`, `DATA.md` | challenge constants and where to get the data |

The match data is **not** in this repository (it may not be redistributed — see `DATA.md`).

---

## Reproduce

```bash
# 1. install (Python 3.11+)
python -m venv .venv && .venv/bin/pip install -r requirements.txt -r ml/requirements.txt

# 2. get the match data (see DATA.md), then either unpack it into ./data or:
export MTAH_DATA_DIR=/path/to/the/data        # the folder that contains tactical/ and broadcast/

# 3. regenerate the submission file — byte for byte the one submitted:
python ml/predict.py "Asociación Fusqi.csv"   # ~1 minute

# 4. check the format (exactly what the official scorer checks):
python evaluation/validate_submission.py "Asociación Fusqi.csv"

# 5. optional: score on the public 5 minutes
python solution/score.py "Asociación Fusqi.csv"
```

**Training** (only needed to retrain the model; the saved model in `ml/artifacts/` is what produced the CSV):

```bash
# download the other-matches data into other-games/ (commands in validation/README.md):
#   other-games/dfl/                  7 Bundesliga matches from the DFL open data (CC-BY 4.0)
#   other-games/metrica-sample-data/  git clone https://github.com/metrica-sports/sample-data
python ml/train.py        # ~20 min: evaluation + final model -> ml/artifacts/
```

Prediction is deterministic: same data + same pinned versions + same saved model = the same file, byte for byte. No GPU, no internet needed at prediction time. The saved model only loads in scikit-learn 1.9.1, which is why it is pinned.
