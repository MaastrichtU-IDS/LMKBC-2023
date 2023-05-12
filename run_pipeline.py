import os


ROOT_PATH = str(os.path.abspath(os.path.dirname(__file__)).split("src")[0])

SRC_PATH = f"{ROOT_PATH}\src"
DATA_PATH = f"{ROOT_PATH}\data"
OUTPUT_PATH = f"{ROOT_PATH}\output"
LM_PATH = f"{ROOT_PATH}\lm"

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

print(ROOT_PATH)


def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}\\train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''
    cmd = f"""
   python baseline.py  --input data/val.jsonl --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output {OUTPUT_PATH}\\testrun-bert.jsonl --train_data data/train.jsonl --model bert-base-cased --batch_size 32 --gpu 0
    """
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    run()
