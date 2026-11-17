"""Score a submission CSV on the public GT and print the ranked row + key detail.

Usage: python solution/score.py [path.csv]
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)

import numpy as np
import possession_metrics as pm
from load_data import load_gt_public, N_FRAMES


def score(path):
    gt = pm.label_codes(load_gt_public())
    pred = pm.label_codes(pm.read_label_csv(path))
    res = pm.evaluate(gt, pred)
    print("file:", os.path.basename(path))
    print("  macroF1   {:.4f}".format(res["macro_f1"]))
    print("  spellF1   {:.4f}  (headline IoU0.5 {:.4f})".format(
        res["spell_f1_mean"], res["spell_f1_at_headline"]))
    print("  per-IoU:  " + "  ".join("{:.1f}:{:.3f}".format(i, res["spell_f1"][i]["f1"])
                                    for i in pm.SPELL_IOUS))
    print("  spells    gt {}  pred {}  ({:+d})".format(
        res["n_gt_spells"], res["n_pred_spells"], res["spell_count_error"]))
    print("  dur       pred {:.1f}s  gt {:.1f}s".format(
        res["mean_spell_duration_s"], res["gt_mean_spell_duration_s"]))
    print("  per class " + "  ".join("{} F1 {:.3f} (P {:.3f} R {:.3f})".format(
        n, res["per_class"][n]["f1"], res["per_class"][n]["precision"],
        res["per_class"][n]["recall"]) for n in "HAD"))
    print("  alive/dead F1 {:.3f}   team acc (gt-alive) {:.3f}".format(
        res["alive_dead_f1"], res["team_acc_gt_alive"]))
    print("  frag      {:.2f}   abstain on graded {}".format(
        res["fragmentation"], res["n_abstain_on_graded"]))
    for tol in pm.TAUS:
        t = res["transitions"][tol]
        print("  trans tau={:<3} P {:.3f} R {:.3f} F1 {:.3f}  matched {} missed {} spurious {}".format(
            tol, t["precision"], t["recall"], t["f1"], t["n_matched"], t["n_missed"],
            t["n_spurious"]))
    return res


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "evaluation", "results", "v1_model.csv")
    score(path)
