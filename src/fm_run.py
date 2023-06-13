import os
import config

RUN_OUTPUT_NAME = "testrun-bert.jsonl"
OUTPUT_FILE = f'{config.OUTPUT_DIR}/{RUN_OUTPUT_NAME}'
TEST_FILE = f'{config.DATA_DIR}/train.jsonl'

import torch

print(torch.cuda.is_available())


def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}/train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''

    cmd_run_fillmask = f"""
   python src/fm_model.py  --test_fn {TEST_FILE} --template_fn res/prompts.csv  --output {OUTPUT_FILE} --train_fn data/train.jsonl --pretrain_model_name bert-base-cased  --train_batch_size 16 --gpu 0 --top_k 20 --threshold 0.1  --dev_fn  data/train_tiny.jsonl --mode "train test" --bin_dir best_ckpt  --train_epoch 30 --test_batch_size 64 --learning_rate 4e-5
    
    """

    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)


if __name__ == "__main__":
    run()
