import os
import config

RUN_OUTPUT_NAME = "filled-mask-valid.jsonl"
OUTPUT_DIR = f'{config.OUTPUT_DIR}/filled-mask'
OUTPUT_FILE = f'{OUTPUT_DIR}/{RUN_OUTPUT_NAME}'

import torch


pretrain_model_name = config.bert_base_cased
task = "fill-mask"
print(torch.cuda.is_available())
model_save_dir = f"{config.BIN_DIR}/{pretrain_model_name}"
model_best_dir = model_save_dir + "/best_ckpt"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

model_load_dir = 'bin/pretrain_fill-mask/bert-base-cased/best_ckpt'
# model_load_dir = config.bert_base_cased
model_load_dir = model_best_dir

def run_token_redefine():
    OUTPUT_DIR = f'{config.OUTPUT_DIR}/filled-mask'
    OUTPUT_FILE = f'{OUTPUT_DIR}/{RUN_OUTPUT_NAME}'
    cmd_token_redefine = f"""
   python src/ta_model.py  --test_fn {config.VAL_FN} --template_fn res/prompts.csv  --output {OUTPUT_FILE} --train_fn data/train.jsonl --pretrain_model_name bert-base-cased  --train_batch_size 128 --gpu 0 --top_k 30 --threshold 0.1  --dev_fn  data/train_tiny.jsonl --mode "train t est" --bin_dir best_ckpt  --train_epoch 30 --test_batch_size 512 --learning_rate 3e-5
    --model_load_dir {config.bert_base_cased} --model_save_dir {model_save_dir} --model_best_dir  {model_best_dir}
    
    """

    print(cmd_token_redefine)
    os.system(cmd_token_redefine)


def run_pretrain_filled_mask():
    final_corpus_fn = f"{config.RES_DIR}/additional_corpus/final_corpus.txt"
    cmd_pretrain_filled_mask = f"""
    
   python src/pre_model.py  --test_fn {config.VAL_FN} --template_fn res/prompts.csv  --output {OUTPUT_FILE} --train_fn {final_corpus_fn}  --train_batch_size 16 --gpu 0 --top_k 30 --threshold 0.1  --dev_fn  data/train_tiny.jsonl --mode "train test eva ulate" --train_epoch 5 --learning_rate 3e-5 --model_load_dir {model_save_dir} --model_save_dir {model_save_dir}
    
    """

    print(cmd_pretrain_filled_mask)
    os.system(cmd_pretrain_filled_mask)


    

def run_fillmask():

    cmd_run_fillmask = f"""
    
   python src/fm_model.py  --test_fn {config.VAL_FN} --template_fn res/prompts0.csv  --output {OUTPUT_FILE} --train_fn data/train.jsonl --train_batch_size 256 --gpu 0 --top_k 15 --threshold 0.1  --dev_fn  data/train_tiny.jsonl --mode "train test" --train_epoch 50 --learning_rate 10e-5 --model_load_dir {model_save_dir} --model_save_dir {model_save_dir} --model_best_dir  {model_best_dir}
    
    """

    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)


if __name__ == "__main__":
    print("hello docker")
    run_token_redefine()
    run_pretrain_filled_mask()
    run_fillmask()
