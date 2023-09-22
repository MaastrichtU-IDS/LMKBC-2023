import argparse
import string
from typing import List, Dict, Union

import pandas as pd
import numpy as np

from file_io import *
import util
import csv
from tqdm import tqdm
import config

def true_positives(preds: List, gts: List) -> int:
    tp = 0
    for pred in preds:
        if (pred in gts):
            tp += 1

    return tp


def precision(preds: List[str], gts: List[str]) -> float:
    # when nothing is predicted, precision 1 irrespective of the ground truth value
    try:
        if len(preds)==0:
            return 1
        # When the predictions are not empty
        return min(true_positives(preds, gts) / len(preds), 1.0)
    except TypeError:
        return 0.0    


def recall(preds: List[str], gts: List[str]) -> float:
    try:
        # When ground truth is empty return 1 even if there are predictions (edge case)
        if len(gts)==0 or gts==[""]:
            return 1.0
        # When the ground truth is not empty
        return true_positives(preds, gts) / len(gts)
    except TypeError:
        return 0.0

def f1_score(p: float, r: float) -> float:
    try:
        return (2 * p * r) / (p + r)
    except ZeroDivisionError:
        return 0.0


def rows_to_dict(rows: List[Dict]) -> Dict:
    return {(r["SubjectEntity"], r["Relation"]): r["ObjectEntities"] for r in rows}


def evaluate_per_sr_pair(pred_rows, gt_rows) -> List[Dict[str, float]]:
    pred_dict = rows_to_dict(pred_rows)
    gt_dict = rows_to_dict(gt_rows)
    
    results = []

    for subj, rel in gt_dict:
        # get the ground truth objects
        gts = gt_dict[(subj, rel)]
        
        # get the predictions
        preds = pred_dict[(subj, rel)]

        # calculate the scores
        p = precision(preds, gts)
        r = recall(preds, gts)
        f1 = f1_score(p, r)

        results.append({
            "SubjectEntity": subj,
            "Relation": rel,
            "p": p,
            "r": r,
            "f1": f1
        })

        # if p > 1.0 or r > 1.0:
        #     print(f"{subj} {rel} {p} {r} {f1} {gts} {preds}")

    return sorted(results, key=lambda x: (x["Relation"], x["SubjectEntity"]))


def combine_scores_per_relation(scores_per_sr: List[Dict[str, float]]) -> dict:
    scores = {}
    for r in scores_per_sr:
        if r["Relation"] not in scores:
            scores[r["Relation"]] = []
        scores[r["Relation"]].append({
            "p": r["p"],
            "r": r["r"],
            "f1": r["f1"],
        })

    for rel in scores:
        scores[rel] = {
            "p": sum([x["p"] for x in scores[rel]]) / len(scores[rel]),
            "r": sum([x["r"] for x in scores[rel]]) / len(scores[rel]),
            "f1": sum([x["f1"] for x in scores[rel]]) / len(scores[rel]),
        }

    return scores




def evaluate_list(gt_rows, pred_rows):

    # calculate the precision and recall score of each triple
    scores_per_sr_pair = evaluate_per_sr_pair(pred_rows, gt_rows)
    # calculate the precision and recall score of each relation
    scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)
    # calculate average score
    scores_per_relation["Average"] = {
        "p": sum([x["p"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
        "r": sum([x["r"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
        "f1": sum([x["f1"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
    }

    # scores_per_relation_pd = pd.DataFrame(scores_per_relation)
    return scores_per_relation



def evaluate_pure(output, test_fn):
    #  read json file
    pred_rows = read_lm_kbc_jsonl(output)
    gt_rows = read_lm_kbc_jsonl(test_fn)

    # calculate the precision and recall score of each triple
    scores_per_sr_pair = evaluate_per_sr_pair(pred_rows, gt_rows)
    # calculate the precision and recall score of each relation
    scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)
    # calculate average score
    scores_per_relation["Average"] = {
        "p": sum([x["p"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
        "r": sum([x["r"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
        "f1": sum([x["f1"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
    }
    # return  scores_per_relation

def evaluate(output, test_fn):
    #  read json file
    pred_rows = read_lm_kbc_jsonl(output)
    gt_rows = read_lm_kbc_jsonl(test_fn)

    # calculate the precision and recall score of each triple
    scores_per_sr_pair = evaluate_per_sr_pair(pred_rows, gt_rows)
    # calculate the precision and recall score of each relation
    scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)
    # calculate average score
    scores_per_relation["Average"] = {
        "p": sum([x["p"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
        "r": sum([x["r"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
        "f1": sum([x["f1"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
    }

    scores_per_relation_pd = pd.DataFrame(scores_per_relation)
    scores_per_relation_pd = scores_per_relation_pd.transpose().round(3)

    # print(scores_per_relation_pd.transpose().round(3))
    return  scores_per_relation
    # add a new field which indicate if the predicted object is true of false
    # the correctness of predicted object can be used in Next-Sentence task, if applicable



def assign_label(output_fn, test_fn) -> List:
    pred_rows = read_lm_kbc_jsonl(output_fn)
    gt_rows = read_lm_kbc_jsonl(test_fn)

    pred_dict = rows_to_dict(pred_rows)
    # return as {(subject_entity,relation):object_entity}
    gt_dict = rows_to_dict(gt_rows)

    for row in pred_rows:
        relation = row['Relation']
        preds = row['ObjectEntities']
        subject = row["SubjectEntity"]
        object_labels = []
        gts = gt_dict[(subject, relation)]
        #  if the predicted object is correct, then mark as 1, elsewise 0
        for pred in preds:
            if pred in gts:
                object_labels.append(1)
            else:
                object_labels.append(0)

        row["ObjectLabels"] = object_labels
        row["TrueObjectEntities"] = gts

    util.file_write_json_line(output_fn, pred_rows, 'w')
    return pred_rows



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

    scores_per_relation["*** Average ***"] = {
        "p": sum([x["p"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
        "r": sum([x["r"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
        "f1": sum([x["f1"] for x in scores_per_relation.values()])
        / len(scores_per_relation),
    }

    print(pd.DataFrame(scores_per_relation).transpose().round(3))



def adaptive_threshold(args, rel_thres_fn):
    origin_result_dict = evaluate(args.output_fn, args.valid_fn)
    predefine_fine = 'res/object_number.tsv'
    with open(predefine_fine) as f:
        topk = csv.DictReader(f, delimiter="\t")
        topk_max_dict = { row["Relation"]:eval(row['Val'])[1] for row in topk}
    threshold_initial_dict = dict()
    for k,v in topk_max_dict.items():
        threshold_initial_dict[k] = 1/float(v)
    pred_rows = util.file_read_json_line(args.output_fn)
    groud_rows = util.file_read_json_line(args.valid_fn)
    relation_list_pred = dict()
    relation_list_groud = dict()
    best_topk_dict=dict()
    for row in pred_rows:
        relation = row['Relation']
        if relation not in relation_list_pred:
            relation_list_pred[relation]=[]
        relation_list_pred[relation].append(row)

    for row in groud_rows:
        relation = row['Relation']
        if relation not in relation_list_groud:
            relation_list_groud[relation]=[]
        relation_list_groud[relation].append(row)

    relation_threshold_dict = dict()
    relation_index= dict()
    # print('threshold_initial_dict',threshold_initial_dict)
    for relation, pred_list in tqdm(relation_list_pred.items()):
        groud_list = relation_list_groud[relation]
        origin_topk = topk_max_dict[relation]
        best_f1=0
        best_precision = 0
        best_recal= 0 
        origin_threshold = threshold_initial_dict[relation]
        best_threshold=0
        threshold_step = 0.01
        best_index = 100
        for i in range(1,int(0.5//threshold_step)):
            threshold = threshold_step*i
            # try_times=0
            for row in pred_list:
                score_index = 0 
                for i, score in enumerate(row['ObjectEntitiesScore']):
                    if score <= threshold:
                        score_index = i
                        break

                row['ObjectEntities'] =row['ObjectEntities'][:score_index]
                row['ObjectEntitiesID'] =row['ObjectEntitiesID'][:score_index]

            eval_dict = evaluate.evaluate_list(groud_list, pred_list)[relation]
            f1 = eval_dict['f1']
            p = eval_dict['p']
            r = eval_dict['r']
            if f1> best_f1:
                best_f1=f1
                best_threshold =threshold
                best_precision= p 
                best_recal= r
                best_index= score_index
                # try_times = 0
            # else:
            #     try_times+=1
            #     if try_times > 3:
            #         break

  
        relation_index[relation] = best_index
        relation_threshold_dict[relation] = best_threshold
    
        origin_result_dict[relation]["best_precision"]=best_precision
        origin_result_dict[relation]["best_recal"]=best_recal
        origin_result_dict[relation]["best_f1"]=best_f1
        origin_result_dict[relation]["best_threshold"]=best_threshold

    pred_rows = util.file_read_json_line(args.output_fn)
    for row in pred_rows:
        relation = row[config.KEY_REL]
        row[config.KEY_OBJS] = row[config.KEY_OBJS][:relation_index[relation]]
        row[config.KEY_OBJS_ID] = row[config.KEY_OBJS_ID][:relation_index[relation]]
    util.file_write_json_line(args.output_fn+'.ths',pred_rows)
        #origin_result_dict[relation]["origin_threshold"]=origin_threshold

    with open(rel_thres_fn,'w') as f:
        json.dump(relation_threshold_dict,f,indent = 2)
    origin_result_dict["Average"]["best_f1"] =  sum([x["best_f1"] if "best_f1" in x else 0 for x in origin_result_dict.values()])/(len(origin_result_dict)-1)
    origin_result_dict["Average"]["best_precision"] =  sum([x["best_precision"] if "best_precision" in x else 0 for x in origin_result_dict.values()])/(len(origin_result_dict)-1)
    origin_result_dict["Average"]["best_recal"] =  sum([x["best_recal"] if "best_recal" in x else 0 for x in origin_result_dict.values()])/(len(origin_result_dict)-1)
    result_dict = {
        "args":args.__dict__,
        "metric":origin_result_dict
        }
    util.file_write_json_line(config.RESULT_FN, [result_dict],'auto')
    scores_per_relation_pd = pd.DataFrame(origin_result_dict)
    print(scores_per_relation_pd.transpose().round(3).to_string(max_cols=12))



    # for relation, v in best_topk_dict.items():
    #     print(relation,v[1],v[0], topk_max_dict[relation] )
    # # print(json.dumps(best_topk_dict, indent=4))
    # average_f1 = sum(map(lambda x:x[1], best_topk_dict.values()))/ len(best_topk_dict)
    # print("average_f1",average_f1)
    #   

if __name__ == "__main__":
    main()
