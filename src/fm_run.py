import os
import config
os.chdir(config.ROOT_PATH)


import torch


test_mode = False
# test_mode = True
# label = "token_recode_std"
label = "pretrain-token_recode_std"
# label = "fine-tune"
if test_mode:
    do_train= False
    do_test = True
    do_valid = False
    RUN_OUTPUT_NAME = "predictions.jsonl"
else:
    RUN_OUTPUT_NAME = "filled-mask-valid.jsonl"
    do_test = False
    do_valid = True
    do_train= False

OUTPUT_DIR = f'output/filled-mask/{label}'
OUTPUT_FILE = f'{OUTPUT_DIR}/{RUN_OUTPUT_NAME}'


pretrain_model_name = config.bert_large_cased

print(torch.cuda.is_available())
model_save_dir = f"bin/{label}/{pretrain_model_name}/fm"
model_best_dir = model_save_dir + "/best_ckpt"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# model_load_dir = 'bin/pretrain_fill-mask/bert-base-cased/best_ckpt'
# model_load_dir = config.bert_base_cased
# model_load_dir = config.bert_large_cased
# model_load_dir = model_best_dir
# model_load_dir = 'bin/pretrain_fill-mask/bert-base-cased/best_ckpt'
# model_load_dir = 'bin/baseline/fill_mask/bert-large-cased/best_ckpt'
# model_load_dir = 'bin/pretrain-val_test/bert-base-cased/best_ckpt'
model_load_dir = 'bin/pretrain-val_test-recode/bert-base-cased/best_ckpt'
# model_load_dir = 'bin/bert-large-cased/checkpoint-105960'

model_best_dir = model_save_dir + "/best_ckpt"
# model_best_dir = model_save_dir + "/best_ckpt"


def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}/train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''

    cmd_run_fillmask = f"""
    
   python src/fm_model.py  --test_fn {config.test_fp} --valid_fn {config.VAL_FN}  --template_fn res/prompts0.csv  --output {OUTPUT_FILE} --train_fn {config.TRAIN_FN} --train_batch_size 64 --gpu 0 --top_k 40 --threshold 0.1  --dev_fn  {config.VAL_FN} --train_epoch 20 --learning_rate 2e-5 --model_load_dir {model_load_dir} --model_save_dir {model_save_dir} --model_best_dir  {model_best_dir} --pretrain_model {config.bert_base_cased}  --silver_data false  --filter false      --token_recode 0   --do_train 0   --do_valid {do_valid}  --do_test {do_test}  --recode_type std

    """

    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)
   

if __name__ == "__main__":
    run()
