import os
import config
os.chdir(config.ROOT_PATH)

RUN_OUTPUT_NAME = "filled-mask-valid.jsonl"
OUTPUT_DIR = f'output/filled-mask'
OUTPUT_FILE = f'{OUTPUT_DIR}/{RUN_OUTPUT_NAME}'

import torch


pretrain_model_name = config.bert_base_cased
task = "fill-mask"
print(torch.cuda.is_available())
model_save_dir = f"bin/{task}/{pretrain_model_name}"
model_best_dir = model_save_dir + "/best_ckpt"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# model_load_dir = 'bin/pretrain_fill-mask/bert-base-cased/best_ckpt'
# model_load_dir = config.bert_base_cased
model_load_dir = model_best_dir


def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}/train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''

    cmd_run_fillmask = f"""
    
   python src/fm_model.py  --test_fn {config.VAL_FN} --template_fn res/prompts0.csv  --output {OUTPUT_FILE} --train_fn {config.TRAIN_FN} --train_batch_size 64 --gpu 0 --top_k 20 --threshold 0.1  --dev_fn  {config.VAL_FN} --mode "tr ain test p redict" --train_epoch 50 --learning_rate 5e-5 --model_load_dir {model_load_dir} --model_save_dir {model_save_dir} --model_best_dir  {model_best_dir} --pretrain_model {config.bert_base_cased}  --silver_data False  --filter False      --token_recode true  
    """

    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)
   

if __name__ == "__main__":
    run()
