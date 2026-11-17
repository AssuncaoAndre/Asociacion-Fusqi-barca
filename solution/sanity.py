"""Whole-match sanity check for a submission CSV: label shares on graded frames,
longest D runs, H/A spell count."""
import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load_data import N_FRAMES, HALF_BLOCKS

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "..", "evaluation", "results", "v6_model.csv")
labels = np.loadtxt(path, delimiter=",", skiprows=1, dtype=str, usecols=1)
assert len(labels) == N_FRAMES, len(labels)
graded = np.zeros(N_FRAMES, bool)
for _, a, b in HALF_BLOCKS:
    graded[a:b + 1] = True
g = labels[graded]
print("file:", os.path.basename(path))
for c in "HADX":
    print("  share", c, round(float(np.mean(g == c)), 4))
dead = labels == "D"
runs = []
i = 0
while i < N_FRAMES:
    if dead[i]:
        j = i
        while j < N_FRAMES and dead[j]:
            j += 1
        if graded[i:j].any():
            runs.append((i, j, (j - i) / 25.0))
        i = j
    else:
        i += 1
runs.sort(key=lambda r: -r[2])
print("  longest D runs (start,end,s):", [(a, b, round(s, 1)) for a, b, s in runs[:15]])
print("  n D runs:", len(runs), " total D s:", round(sum(r[2] for r in runs), 1))
team = np.where(labels == "H", 1, np.where(labels == "A", 2, 0))
spells = 0
i = 0
while i < N_FRAMES:
    if graded[i] and team[i] in (1, 2):
        j = i
        while j < N_FRAMES and graded[j] and team[j] == team[i]:
            j += 1
        spells += 1
        i = j
    else:
        i += 1
print("  H/A spells (graded):", spells)
