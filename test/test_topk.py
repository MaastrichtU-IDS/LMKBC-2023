

import csv


def adaptive_top_k():
    predefine_fine = 'res/object_number.tsv'
    with open(predefine_fine) as f:
        topk = csv.DictReader(f, delimiter="\t")
        topk_max_dict = { row["Relation"]:eval(row['Val'])[1] for row in topk}
        for row in topk:
            # print(row)
            print(row['Relation'],row['Test'])
    print(topk_max_dict)
    # evaluate(args.output, args.test_fn)

adaptive_top_k()
