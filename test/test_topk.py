import math

import chunk
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

def split_list():
    alist= [1,2,3,4,5,6,7,8]
    split_size = 3

    chunk_size = math.ceil(len(alist) / split_size )

    # if len(alist) % 3 != 0:
    #     chunk_size+=1
    result = [alist[i*chunk_size:(i+1)*chunk_size] for i in range(split_size)]

    test_list = alist[1:20]
    print(math.ceil(1.00000))
    print(test_list)
    print(result)

split_list()
# adaptive_top_k()
