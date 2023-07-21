import os
import config
import torch
print(torch.cuda.is_available())

TASK_NAME = "text-generation"
RUN_OUTPUT_NAME = "test-opt-1.3b.jsonl"
OUTPUT_DIR = f'{config.OUTPUT_DIR}/{TASK_NAME}'
OUTPUT_FILE = f'{OUTPUT_DIR}/{RUN_OUTPUT_NAME}'
TEST_FILE = f'{config.DATA_DIR}/val.jsonl'


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
pretrain_model_name = config.opt_350m
task = "text-generation"
print(torch.cuda.is_available())
model_save_dir = f"{config.BIN_DIR}/{task}/{pretrain_model_name}"
model_best_dir = model_save_dir + "/best_ckpt"




model_load_dir = 'bin/pretrain_fill-mask/bert-base-cased/best_ckpt'
model_load_dir = pretrain_model_name
# model_load_dir = model_best_dir


def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}/train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''

    cmd_run_fillmask = f"""

   python src/ge_model.py  --test_fn {config.VAL_FN} --template_fn res/question-prompts0.csv  --output {OUTPUT_FILE} --train_fn data/train.jsonl --pretrain_model_name facebook/opt-350m  --train_batch_size 16 --gpu 0    --dev_fn  data/train_tiny.jsonl --mode "tr ain  test evaulate" --bin_dir best_ckpt  --train_epoch 50 --test_batch_size 32 --learning_rate 5e-5 --few_shot 3 --model_load_dir {model_load_dir} --model_save_dir {model_save_dir} --model_best_dir  {model_best_dir}
   
    """

    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)


if __name__ == "__main__":
    run()
