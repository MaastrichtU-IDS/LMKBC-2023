import os

ROOT_PATH = str(os.path.abspath(os.path.dirname(__file__)).split("src")[0])

SRC_PATH = f"{ROOT_PATH}\src"
DATA_PATH = f"{ROOT_PATH}\data"
OUTPUT_PATH = f"{ROOT_PATH}\output"
LM_PATH = f"{ROOT_PATH}\lm"

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
