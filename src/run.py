import os
import config

RUN_OUTPUT_NAME = "testrun-bert.jsonl"
OUTPUT_FILE = f'{config.OUTPUT_DIR}\\{RUN_OUTPUT_NAME}'
TEST_FILE = f'{config.DATA_DIR}\\train.jsonl'


def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}\\train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''

    cmd_run_fillmask = f"""
   python src\\model_fillmask.py  --test_fn {TEST_FILE} --template_fn res\\prompts.csv  --output {OUTPUT_FILE} --train_fn data/train.jsonl --pretrain_model_name bert-large-cased-whole-word-masking --train_batch_size 32 --gpu 0 --top_k 8 --threshold 0.1  --dev_fn  data/train_tiny.jsonl --mode "predictt" --bin_dir {config.BIN_DIR}\\best_ckpt  --train_epoch 5 --test_batch_size 128
    
    """

    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)


if __name__ == "__main__":
    run()
