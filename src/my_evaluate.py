import argparse
import string
from typing import List, Dict, Union

import pandas as pd
import numpy as np

from file_io import *


def is_none_gts(gts: List[str]) -> bool:
    return not gts


def is_none_preds(preds: List[str]) -> bool:
    return (
        preds is None
        or len(preds) == 0
        or (
            len(preds) == 1
            and (
                list(preds)[0] is None
                or list(preds)[0] is np.nan
                or list(preds)[0].lower() in {"", "none", "null"}
            )
        )
    )


def true_positives(preds: List[str], gts: List[str]) -> int:
    tp = 0
    for pred in preds:
        if pred in gts:
            tp += 1

    return tp


def precision(preds: List[str], gts: List[str]) -> float:
    # when nothing is predicted, precision 1 irrespective of the ground truth value
    if is_none_preds(preds):
        if is_none_preds(gts):
            return 1.0
        else:
            return 0.0

    # When the ground truth object is not none
    try:
        return true_positives(preds, gts) / len(preds)
    except TypeError as e:
        e.with_traceback()
        return 0.0


def recall(preds: List[str], gts: List[str]) -> float:
    # When the ground truth object is none return 1 even if there are predictions (edge case)
    if is_none_preds(gts):
        if is_none_preds(preds):
            return 1.0
        else:
            return 0.0

    # When the ground truth object is not none
    try:
        return true_positives(preds, gts) / len(gts)
    except TypeError as e:
        e.with_traceback()
        print(f"recall 0.0 gts={gts}")
        return 0.0


def f1_score(p: float, r: float) -> float:
    if p < 0.00001 and r < 0.00001:
        return 0.0
    try:
        return (2 * p * r) / (p + r)
    except ZeroDivisionError:
        return 0.0


def rows_to_dict(rows: List[Dict]) -> Dict:
    return {(r["SubjectEntity"], r["Relation"]): r["ObjectEntitiesID"] for r in rows}


def evaluate_per_sr_pair(pred_rows, gt_rows) -> List[Dict[str, float]]:
    pred_dict = rows_to_dict(pred_rows)
    gt_dict = rows_to_dict(gt_rows)

    results = []

    for subj, rel in gt_dict.keys():
        # get the ground truth objects
        gts = gt_dict[(subj, rel)]

        # get the predictions
        preds = pred_dict[(subj, rel)]

        # calculate the scores
        p = precision(preds, gts)
        r = recall(preds, gts)
        f1 = f1_score(p, r)

        results.append(
            {"SubjectEntity": subj, "Relation": rel, "p": p, "r": r, "f1": f1}
        )

        # if p > 1.0 or r > 1.0:
        #     print(f"{subj} {rel} {p} {r} {f1} {gts} {preds}")

    return sorted(results, key=lambda x: (x["Relation"], x["SubjectEntity"]))


def combine_scores_per_relation(scores_per_sr: List[Dict[str, float]]) -> dict:
    scores = {}
    for r in scores_per_sr:
        if r["Relation"] not in scores:
            scores[r["Relation"]] = []
        scores[r["Relation"]].append(
            {
                "p": r["p"],
                "r": r["r"],
                "f1": r["f1"],
            }
        )

    for rel in scores:
        scores[rel] = {
            "p": sum([x["p"] for x in scores[rel]]) / len(scores[rel]),
            "r": sum([x["r"] for x in scores[rel]]) / len(scores[rel]),
            "f1": sum([x["f1"] for x in scores[rel]]) / len(scores[rel]),
        }

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Precision, Recall and F1-score of predictions"
    )

    parser.add_argument(
        "-p",
        "--predictions",
        type=str,
        required=True,
        help="Path to the predictions file (required)",
    )
    parser.add_argument(
        "-g",
        "--ground_truth",
        type=str,
        required=True,
        help="Path to the ground truth file (required)",
    )

    args = parser.parse_args()

    pred_rows = read_lm_kbc_jsonl(args.predictions)
    gt_rows = read_lm_kbc_jsonl(args.ground_truth)

    scores_per_sr_pair = evaluate_per_sr_pair(pred_rows, gt_rows)
    scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)

    scores_per_relation["*** Relation Average ***"] = {
        "p": sum([x["p"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
        "r": sum([x["r"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
        "f1": sum([x["f1"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
    }

    scores_per_relation_pd = pd.DataFrame(scores_per_relation)

    print(scores_per_relation_pd.transpose().round(3))
    average_pd = scores_per_relation_pd.a


if __name__ == "__main__":
    main()
