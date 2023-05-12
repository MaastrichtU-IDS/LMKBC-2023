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
   python evaluate.py -p {DATA_PATH}/val.jsonl -g {OUTPUT_PATH}/testrun-bert.jsonl
    """
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    run()
