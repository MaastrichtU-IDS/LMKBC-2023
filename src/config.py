import os

ROOT_PATH = str(os.path.abspath(os.path.dirname(__file__)).split("src")[0])

SRC_PATH = f"{ROOT_PATH}/src"
DATA_DIR = f"{ROOT_PATH}/data"
OUTPUT_DIR = f"{ROOT_PATH}/output"
RES_PATH = f"{ROOT_PATH}/res"
BIN_DIR = f"{ROOT_PATH}/bin"
LOGGING_DIR = f"{ROOT_PATH}/logging"


EMPTY_TOKEN = 'Empty Token'
MASK_TOKEN_SIZE = 50
MAX_LENGTH = 512

TRAIN_FN = f'{DATA_DIR}/train.jsonl'
TRAIN_TINY_FN = f'{DATA_DIR}/train_tiny.jsonl'
VAL_FN = f'{DATA_DIR}/val.jsonl'

KEY_OBJS = "ObjectEntities"
KEY_REL = "Relation"
KEY_SUB = "SubjectEntity"

check_dir_list = [
    BIN_DIR,
    OUTPUT_DIR,
    LOGGING_DIR,
]

for d in check_dir_list:
    if not os.path.exists(d):
        os.mkdir(d)
