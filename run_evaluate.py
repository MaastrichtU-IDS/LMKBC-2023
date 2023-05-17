import os
import config


# result_name = "testrun-bert.jsonl"
def run():
    # cmd = f'''
    #     python {SRC_PATH}\pipeline.py -i {DATA_PATH}\\train_tiny.jsonl -o {OPTPUT_PATH}\prediction.jsonl -m "bert-large-cased"
    #     '''
    cmd = f"""
   python evaluate.py -p {config.DATA_DIR}/val.jsonl -g {config.OUTPUT_DIR}/testrun-bert.jsonl
    """
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    run()
