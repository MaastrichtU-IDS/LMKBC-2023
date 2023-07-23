import os
import config

RUN_OUTPUT_NAME = "filled-mask.jsonl"
OUTPUT_DIR = f'{config.OUTPUT_DIR}/filled-mask'
OUTPUT_FILE = f'{OUTPUT_DIR}/{RUN_OUTPUT_NAME}'
TEST_FILE = f'{config.DATA_DIR}/val.jsonl'

import torch


pretrain_model_name = config.bert_base_cased
task = "pretrain_fill-mask"
print(torch.cuda.is_available())
model_save_dir = f"{config.BIN_DIR}/{task}/{pretrain_model_name}"
final_corpus_fn = f"{config.RES_DIR}/additional_corpus/fm_pretrain_2.txt"
model_best_dir = model_save_dir+'/best_ckpt'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# model_load_dir = f'{model_save_dir}/best_ckpt'
model_load_dir = config.bert_tiny


def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}/train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''

    cmd_run_fillmask = f"""
    
   python {config.SRC_DIR}/pre_model.py   --train_fn {final_corpus_fn}  --train_batch_size 16 --gpu  0  --train_epoch 10 --learning_rate 3e-5   --model_load_dir {model_load_dir} --model_save_dir {model_save_dir} --model_best_dir  {model_best_dir}


    """

    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)


if __name__ == "__main__":
    run()
