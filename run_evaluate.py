import os
import config


def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}\\train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''
    cmd = f"""
   python evaluate.py -p {config.DATA_PATH}/val.jsonl -g {config.OUTPUT_PATH}/testrun-bert.jsonl
    """
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    run()
