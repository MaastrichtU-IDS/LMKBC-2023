import os
import config

RUN_OUTPUT_NAME = "filled-mask.jsonl"
OUTPUT_DIR = f'{config.OUTPUT_DIR}/filled-mask'
OUTPUT_FILE = f'{OUTPUT_DIR}/{RUN_OUTPUT_NAME}'
TEST_FILE = f'{config.DATA_DIR}/val.jsonl'

import torch

print(torch.cuda.is_available())

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}/train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''

    cmd_run_fillmask = f"""
   python src/ta_model.py  --test_fn {config.VAL_FN} --template_fn res/prompts.csv  --output {OUTPUT_FILE} --train_fn data/train.jsonl --pretrain_model_name bert-base-cased  --train_batch_size 128 --gpu 0 --top_k 30 --threshold 0.1  --dev_fn  data/train_tiny.jsonl --mode "train t est" --bin_dir best_ckpt  --train_epoch 30 --test_batch_size 512 --learning_rate 3e-5
    
    """

    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)


if __name__ == "__main__":
    run()
