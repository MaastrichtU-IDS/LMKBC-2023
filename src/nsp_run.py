import os
import config

RUN_OUTPUT_NAME = "test_run-bert.jsonl"
OUTPUT_FILE = f'{config.OUTPUT_DIR}/{RUN_OUTPUT_NAME}'


VAL_FILE = f'{config.DATA_DIR}/triple_classification_val.jsonl'
TRAIN_FILE = f'{config.DATA_DIR}/triple_classification_train.jsonl'
DEV_FILE = f'{config.DATA_DIR}/triple_classification_dev.jsonl'


def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}/train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''

    cmd_run_fillmask = f"""
   python src/nsp_model.py  --test_fn {VAL_FILE} --template_fn res/prompts.csv  --output {OUTPUT_FILE} --train_fn {TRAIN_FILE} --pretrain_model_name bert-base-cased  --train_batch_size 16 --gpu 0 --top_k 20 --threshold 0.1  --dev_fn  {DEV_FILE} --mode "train test" --bin_dir best_ckpt  --train_epoch 1 --test_batch_size 64 --learning_rate 4e-5
    
    """

    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)


if __name__ == "__main__":
    run()
