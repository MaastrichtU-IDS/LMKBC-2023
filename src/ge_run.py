import os
import config

TASK_NAME = "text-generation"
RUN_OUTPUT_NAME = "test-opt-1.3b.jsonl"
OUTPUT_DIR = f'{config.OUTPUT_DIR}/{TASK_NAME}'
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

   python src/ge_model.py  --test_fn {config.VAL_FN} --template_fn res/question-prompts0.csv  --output {OUTPUT_FILE} --train_fn data/train.jsonl --pretrain_model_name facebook/opt-350m  --train_batch_size 16 --gpu 0    --dev_fn  data/train_tiny.jsonl --mode "tra in test" --bin_dir best_ckpt  --train_epoch 100 --test_batch_size 32 --learning_rate 5e-6 --few_shot 3

    """

    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)


if __name__ == "__main__":
    run()
