import os
import config

RUN_OUTPUT_NAME = "filled-mask.jsonl"
OUTPUT_DIR = f'{config.OUTPUT_DIR}/filled-mask'
OUTPUT_FILE = f'{OUTPUT_DIR}/{RUN_OUTPUT_NAME}'
TEST_FILE = f'{config.DATA_DIR}/val.jsonl'

import torch

pretrain_model_name = 'bert-base-cased'
task = "fill-mask"
print(torch.cuda.is_available())
bin_dir = f"{config.BIN_DIR}/{task}/{pretrain_model_name}"
best_dir = f"{bin_dir}/best_ckpt"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(bin_dir):
    os.makedirs(bin_dir)

# best_dir = 'bin/token-align/bert-base-cased/best_ckpt'


def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}/train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''

    cmd_run_fillmask = f"""
    
   python src/fm_model.py  --test_fn {config.VAL_FN} --template_fn res/prompts.csv  --output {OUTPUT_FILE} --train_fn data/train.jsonl --pretrain_model_name bert-base-cased  --train_batch_size 256 --gpu 0 --top_k 30 --threshold 0.1  --dev_fn  data/train_tiny.jsonl --mode "tr ain test" --bin_dir {best_dir}  --train_epoch 30 --test_batch_size 1024 --learning_rate 3e-5
    
    """

    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)


if __name__ == "__main__":
    run()
