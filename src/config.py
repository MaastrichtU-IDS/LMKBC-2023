import os

ROOT_PATH = str(os.path.abspath(os.path.dirname(__file__)).split("src")[0])

SRC_PATH = f"{ROOT_PATH}\src"
DATA_DIR = f"{ROOT_PATH}\data"
OUTPUT_DIR = f"{ROOT_PATH}\output"
LM_PATH = f"{ROOT_PATH}\lm"
BIN_DIR = f"{ROOT_PATH}\\bin"
LOGGING_DIR = f"{ROOT_PATH}\\logging"

check_dir_list = [
    BIN_DIR,
    OUTPUT_DIR,
    LOGGING_DIR,
]

for d in check_dir_list:
    if not os.path.exists(d):
        os.mkdir(d)
