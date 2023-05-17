import os
import config


def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}\\train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''
    cmd = f"""
   python baseline.py  --input data/val.jsonl --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output {config.OUTPUT_DIR}\\testrun-bert.jsonl --train_data data/train.jsonl --model bert-large-cased --batch_size 32 --gpu 0
    """
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    run()
