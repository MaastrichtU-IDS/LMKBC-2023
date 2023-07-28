import os

ROOT_PATH = str(os.path.abspath(os.path.dirname(__file__)).split("/src")[0])

SRC_DIR = f"{ROOT_PATH}/src"
DATA_DIR = f"{ROOT_PATH}/data"
OUTPUT_DIR = f"{ROOT_PATH}/output"
RES_DIR = f"{ROOT_PATH}/res"
BIN_DIR = f"{ROOT_PATH}/bin"
LOGGING_DIR = f"{ROOT_PATH}/logging"

CACHE_DIR = f"{ROOT_PATH}/cache"  

TRANSFOER_CACHE_DIR = f'{CACHE_DIR}/transformers/'

mask_length = 3

EMPTY_TOKEN = 'Empty'
EMPTY_STR = ''

MASK_TOKEN_SIZE = 50
FM_MAX_LENGTH = 32
TA_MAX_LENGTH = 64
GE_MAX_LENGTH = 512

bert_base_cased = 'bert-base-cased'
bert_large_cased = 'bert-large-cased'
bert_tiny = 'prajjwal1/bert-tiny'
opt_350m = 'facebook/opt-350m'

PRETRAIN_FN = f'{DATA_DIR}/pretrain_corpus.jsonl'

TRAIN_FN = f'{DATA_DIR}/train.jsonl'
TRAIN_TINY_FN = f'{DATA_DIR}/train_tiny.jsonl'
VAL_FN = f'{DATA_DIR}/val.jsonl'

RESULT_FN= f'{RES_DIR}/results.jsonl'
KEY_OBJS = "ObjectEntities"
KEY_REL = "Relation"
KEY_SUB = "SubjectEntity"
OBJLABELS_KEY = 'ObjectLabels'

FROM_KG = 'from'
TO_KG = 'to'

TOKENIZER_PATH = f'{RES_DIR}/tokenizer/bert'

check_dir_list = [
    BIN_DIR,
    OUTPUT_DIR,
    LOGGING_DIR,
]

for d in check_dir_list:
    if not os.path.exists(d):
        os.mkdir(d)
