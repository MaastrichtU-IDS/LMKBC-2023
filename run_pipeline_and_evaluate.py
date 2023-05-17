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
   python model_fillmask.py  --test_fn {TEST_FILE} --template_fn prompts.csv  --output {OUTPUT_FILE} --train_fn data/train.jsonl --pretrain_model_name bert-large-cased --train_batch_size 128 --gpu 0 --top_k 15 --threshold 0.1  --dev_fn  data/train_tiny.jsonl --mode "test" --bin_dir {config.BIN_DIR}\\best_ckpt  --train_epoch 10 --test_batch_size 64
    
    """
    #     cmd_run_qa = f'''
    #             python baseline.py  --input {TEST_FILE} --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output {OUTPUT_FILE} --train_data data/train.jsonl --model facebook/opt-1.3b --batch_size 4 --gpu 0  --dev_data  data/train_tiny.jsonl
    #             '''

    # cmd_evaluate = f"""
    # python evaluate.py -p {OUTPUT_FILE} -g {TEST_FILE}"
    # """
    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)
    # print("run finished, then evaluate.......................")
    # print(cmd_evaluate)
    # os.system(cmd_evaluate)


if __name__ == "__main__":
    run()
