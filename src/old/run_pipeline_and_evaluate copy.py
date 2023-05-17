import os
import config

RUN_OUTPUT_NAME = "testrun-bert.jsonl"
OUTPUT_FILE = f'{config.OUTPUT_DIR}\\{RUN_OUTPUT_NAME}'


def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}\\train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''
    cmd_run_fillmask = f"""
   python baseline.py  --input data/val.jsonl --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output {OUTPUT_FILE} --train_data data/train.jsonl --model bert-large-cased --batch_size 128 --gpu 0 --top_k 10 --threshold 0.8  --dev_data  data/train_tiny.jsonl  --train_epoch 10
    """
    cmd_run_qa = f'''
            python baseline.py  --input data/val.jsonl --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output {OUTPUT_FILE} --train_data data/train.jsonl --model facebook/opt-1.3b --batch_size 4 --gpu 0  --dev_data  data/train_tiny.jsonl
            '''

    cmd_evaluate = f"""
   python evaluate.py -p {config.DATA_DIR}/val.jsonl -g {OUTPUT_FILE}"
    """
    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)
    print("run finished, then evaluate.......................")
    print(cmd_evaluate)
    os.system(cmd_evaluate)


if __name__ == "__main__":
    run()
