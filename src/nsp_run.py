import config
import os

task = "next-sentence"

RUN_OUTPUT_NAME = "{task}.jsonl"
OUTPUT_DIR = f'{config.OUTPUT_DIR}/{task}'
OUTPUT_FILE = f'{OUTPUT_DIR}/{RUN_OUTPUT_NAME}'

import torch


pretrain_model_name = config.bert_base_cased

print(torch.cuda.is_available())
model_save_dir = f"{config.BIN_DIR}/{task}/{pretrain_model_name}"
model_best_dir = model_save_dir + "/best_ckpt"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# model_load_dir = 'bin/pretrain_fill-mask/bert-base-cased/best_ckpt'
# model_load_dir = config.bert_base_cased
model_load_dir = model_best_dir


# train_fn = 'output/filled-mask/train.jsonl'
# dev_fn = 'output/filled-mask/dev.jsonl'

train_fn = 'output/filled-mask/filled-mask-train.jsonl'
dev_fn = 'output/filled-mask/filled-mask-valid.jsonl'


def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}/train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''

    cmd_run_fillmask = f"""
    
   python src/nsp_model.py  --test_fn {dev_fn} --template_fn res/prompts0.csv  --output {OUTPUT_FILE} --train_fn {train_fn}  --train_batch_size 16 --gpu 0 --top_k 15 --threshold 0.1  --dev_fn   {dev_fn} --mode "tra in  test" --train_epoch 10  --is_debug true --learning_rate 5e-5 --model_load_dir {model_load_dir} --model_save_dir {model_save_dir} --model_best_dir  {model_best_dir}
    
    """

    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)


if __name__ == "__main__":
    run()
