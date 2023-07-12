import os

ROOT_PATH = str(os.path.abspath(os.path.dirname(__file__)).split("/src")[0])

SRC_DIR = f"{ROOT_PATH}/src"
DATA_DIR = f"{ROOT_PATH}/data"
OUTPUT_DIR = f"{ROOT_PATH}/output"
RES_DIR = f"{ROOT_PATH}/res"
BIN_DIR = f"{ROOT_PATH}/bin"
LOGGING_DIR = f"{ROOT_PATH}/logging"


EMPTY_TOKEN = 'Empty'
NULL_TOKEN = 'null'
EMPTY_STR = ''

MASK_TOKEN_SIZE = 50
FM_MAX_LENGTH = 32
TA_MAX_LENGTH = 64
GE_MAX_LENGTH = 512

bert_base_cased = 'bert-base-cased'
bert_large_cased = 'bert-large-cased'
opt_350m = 'opt-350m'

TRAIN_FN = f'{DATA_DIR}/train.jsonl'
TRAIN_TINY_FN = f'{DATA_DIR}/train_tiny.jsonl'
VAL_FN = f'{DATA_DIR}/val.jsonl'

KEY_OBJS = "ObjectEntities"
KEY_REL = "Relation"
KEY_SUB = "SubjectEntity"

TOKENIZER_PATH = 'res/tokenizer/bert'
check_dir_list = [
    BIN_DIR,
    OUTPUT_DIR,
    LOGGING_DIR,
]

for d in check_dir_list:
    if not os.path.exists(d):
        os.mkdir(d)
