# DFL test bench

Scores the possession model on 7 full Bundesliga matches instead of the public 5 minutes.

The public 5 minutes hold only 17 possessions, so one possession more or less moves spellF1 by
0.03–0.06, and the rules forbid tuning on them. The bench gives you about 3,000 labelled
possessions from other matches, which the rules allow.

## Setup (once)

1. Install the Python dependencies of the repo (`pip install -r requirements.txt`).
2. Have the challenge data in `data/tactical/` as usual (see `DATA.md`). The bench reads the
   challenge tracking once to measure how noisy it is (see "How it works").
3. **Download the outside datasets.** They are not in the repository: a fresh clone has no
   `other-games/` folder, and you must download the data yourself. Everything goes in
   `other-games/` at the root of the repository:

   ```text
   other-games/
   ├─ dfl/                    7 Bundesliga matches    (needed by the bench, ~2.5 GB)
   ├─ metrica-sample-data/    3 Metrica sample games  (optional, not used by the bench yet)
   └─ skillcorner-opendata/   20 A-League matches     (optional, not used by the bench yet)
   ```

   Run these from the root of the repository.

   **DFL (required).** All files go straight into `other-games/dfl/`, without subfolders:

   ```bash
   mkdir -p other-games/dfl && cd other-games/dfl
   curl -s "https://huggingface.co/api/datasets/pysport/idsse-data/tree/main" \
     | python3 -c "import json,sys; [print(f['path']) for f in json.load(sys.stdin) if f['path'].startswith('DFL_')]" \
     | xargs -P 4 -I{} curl -sSL --retry 3 -o {} "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/{}"
   cd ../..
   ```

   You should end up with 21 files: one `DFL_02_01_matchinformation_*`, one
   `DFL_03_02_events_raw_*` and one `DFL_04_03_positions_raw_observed_*` per match.

   **Metrica (optional):**

   ```bash
   git clone --depth 1 https://github.com/metrica-sports/sample-data.git other-games/metrica-sample-data
   ```

   **SkillCorner (optional).** The tracking files are stored with Git LFS. Without `git-lfs`,
   the clone only contains small placeholder files, so fetch the real ones separately:

   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/SkillCorner/opendata.git other-games/skillcorner-opendata
   cd other-games/skillcorner-opendata
   B=$(git rev-parse --abbrev-ref HEAD)
   git ls-files 'data/matches/*/*tracking_extrapolated.jsonl' \
     | xargs -P 4 -I{} curl -sSL --retry 3 -o {} "https://media.githubusercontent.com/media/SkillCorner/opendata/$B/{}"
   cd ../..
   ```

   Each tracking file should be about 90 MB. If they are around 130 bytes, they are still
   placeholders.

   `other-games/SOURCES.md` describes each dataset and its licence, if you have a copy of it.
   In short: credit the DFL and cite Bassek et al., *Scientific Data* 2025
   (doi:10.1038/s41597-025-04505-y), CC-BY 4.0; acknowledge Metrica Sports; credit SkillCorner.

`other-games/` is gitignored, so these downloads never reach the public repository. Don't force
them in: each dataset has its own terms.

## Run it

```bash
python validation/dfl_bench.py                       # all 7 matches, degraded tracking
python validation/dfl_bench.py --clean               # also score on the clean DFL tracking
python validation/dfl_bench.py --matches J03WPY      # one match (quick check, ~15 s)
python validation/dfl_bench.py --seed 1              # different random degradation
```

The first run parses each match (about 10 s each) and caches it in `other-games/dfl/cache/`.
After that, a full run takes about 90 s.

## Reading the output

```
== degraded tracking, mean over 7 matches
config                      spellF1  macroF1  teamAcc  aliveF1  spells/true   spellF1 per IoU
v10                          0.3269   0.7739    0.758    0.911   2557/3061    0.477 0.405 ...
v10 + dead_backdate          0.3324   0.7801    0.752    0.915   2550/3061    ...   (+0.0055 vs v10; better in 6/7)
```

| Column | Meaning |
|---|---|
| `config` | `v10` = every v11/v12/v13 change off. `v12` = the v11/v12 changes on, the v13 ones off (the file submitted first). `v13 (current defaults)` = the defaults in `solution/model.py`. `v10 + x` = v10 with only change `x` on; `v12 + x` = v12 with only v13 change `x` on; `v13 + x` = the defaults plus a v13 change that is off by default. |
| `spellF1` | the ranking metric, averaged over the 7 matches |
| `macroF1` | the tie-break metric |
| `teamAcc` | share of in-play frames given to the right team |
| `aliveF1` | how well in-play vs dead ball is detected |
| `spells/true` | possessions we produced vs possessions in the labels, summed over the matches |
| `spellF1 per IoU` | spellF1 at overlap thresholds 0.3, 0.4, 0.5, 0.6, 0.7. Low values at 0.6–0.7 mean possession edges are off; low values everywhere mean possessions are missing or wrong. |
| `vs v10`, `better in` | the difference from v10, and in how many of the 7 matches the config beat v10 |

**What counts as a real difference.** On a single match, spellF1 differences under about 0.04 are
noise. Averaged over 7 matches, that drops to about 0.015. A small gain that wins in 6 or 7 of
the 7 matches is a useful signal; a small gain that wins in 3 or 4 is not.

**Trust the differences, not the levels.** DFL decides possession its own way, which is not
exactly the challenge's way, so DFL scores sit well below the challenge's. Compare configs with
each other; don't compare a DFL number with a public-window number.

## Testing a change

1. Put the change behind a `USE_...` switch in `solution/model.py`, like the existing ones,
   switched off.
2. Add the switch name to `FLAGS` and to `V13_FLAGS` in `validation/dfl_bench.py`. A
   `v12 + <name>` row (and, while it is off by default, a `v13 + <name>` row) then appears.
3. Run the bench and look at the degraded results first: they are closest to the challenge.
   Also score the Metrica sample games (`python ml/train.py --no-final` prints the rules
   model on them), which are the closest stand-in for the challenge tracking.
4. Keep the change only if it beats the current defaults in most matches. Then switch it on
   in `solution/model.py`, regenerate the submission and update the technical document with
   the bench result.

The bench never edits `solution/model.py`. It only flips the switches in memory and restores
them at the end. Every switch in `FLAGS` is reset before each config, so a switch that is
not listed there would leak from one config to the next.

**Changing the rules model invalidates the learned model's cache.** `ml/` uses the rules
model's signals and answers as features, and `other-games/ml-cache/` is keyed by match only.
After a change to `solution/model.py`, delete that folder and retrain (`python ml/train.py`).

### v13 results (2026-11-11)

Three changes were tried on top of v12. Mean spellF1, DFL degraded (7 matches) and Metrica (2):

| config | DFL | better in | Metrica | better in |
|---|---|---|---|---|
| v12 | 0.3307 | | 0.5306 | |
| v12 + stop_backdate | 0.3425 | 7/7 | 0.5306 | 1/2 (no change) |
| v12 + sandwich_merge 1.0 s | 0.3625 | 6/7 | 0.5375 | 2/2 |
| v12 + stop + sandwich 1.5 s (= **v13**) | **0.3862** | **7/7** | **0.5391** | **2/2** |
| v12 + stop + sandwich 2.0 s | 0.3846 | 7/7 | 0.5323 | 1/2 |
| v13 + exit_backdate | 0.3876 | 7/7 | 0.5368 | 1/2 |

`exit_backdate` lands the restart on the right frame (median lag 4 frames -> 0) but moves
spellF1 only within noise, so it stays off. On the challenge match v13 differs from v12 on
0.48% of graded frames: 257 possessions instead of 283, the same 112 stoppages, and an
identical public window.

## How it works

1. **Labels.** Every DFL ball frame carries `BallStatus` (0 = dead) and `BallPossession`
   (1 = home, 2 = away). These become `D`, `H` and `A`. The two halves are joined with 500
   ungraded (`X`) frames between them. At start-up the bench prints a check per match: how
   often the player nearest the ball belongs to the team the label names. It should be about
   0.9 for both teams.
2. **Degradation.** DFL tracking is far cleaner than the challenge's automatic tracking, so
   each match is made to look like the challenge data. All the statistics are measured from
   the challenge tracking itself, which is an input, not an answer:
   - the ball is hidden in stretches drawn from the challenge's own hidden and visible stretch
     lengths, with only stretches under 10 s during live play (a longer gap counts as a
     stoppage in the challenge's labels);
   - about 1% of visible ball frames jump onto a random player for up to 1 s, like the
     tracker's glitches;
   - player frames are dropped so about 20.5 of 22 players are tracked, with track stretches
     as long as the challenge's;
   - there is no per-frame position noise: the challenge ball is measured to be as smooth frame
     to frame as the DFL ball.

   The degradation is random but seeded, so the same `--seed` gives the same result.
3. **Running the model.** The unmodified model runs on each match. The bench only swaps in the
   match's data, frame count, halves and pitch size (105 × 68 m).
4. **Scoring** uses the organisers' own `evaluation/possession_metrics.py`.

## Known limits

- **It is a little easier than the real data.** The degraded ball is visible about 70% of the
  time, against 59% in the challenge.
- **It does not test phantom player IDs.** DFL tracking has no ID swaps, so the phantom-mask
  change always shows "no change" here.
- **It does not test anything from smart tagging or video.** It only covers what the tracking
  can show.
- **The labels are DFL's.** A change that matches DFL's moment of switching may not match the
  challenge's, which is why the bench is evidence, not proof.
