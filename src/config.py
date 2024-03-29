import os

ROOT_PATH = str(os.path.abspath(os.path.dirname(__file__)).split("/src")[0])
os.chdir(ROOT_PATH)

SRC_DIR = f"src"
DATA_DIR = f"data"
OUTPUT_DIR = f"output"
RES_DIR = f"res"
BIN_DIR = f"bin"
LOGGING_DIR = f"logging"

CACHE_DIR = f"cache"  

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
silver_dir = f"res/silver"

PRETRAIN_FN = f'{DATA_DIR}/pretrain_corpus.jsonl'

TRAIN_FN = f'{DATA_DIR}/train.jsonl'
TRAIN_TINY_FN = f'{DATA_DIR}/train_tiny.jsonl'
VAL_FN = f'{DATA_DIR}/val.jsonl'
test_fp = f'data/test.jsonl'
prompt_fp = f'res/prompts0.csv'
token_count_fp = 'res/token_count_wide.json'
RESULT_FN= f'{RES_DIR}/results.jsonl'
KEY_OBJS = "ObjectEntities"
KEY_REL = "Relation"
KEY_SUB = "SubjectEntity"
OBJLABELS_KEY = 'ObjectLabels'
KEY_SUB_ID='SubjectEntityID'
KEY_OBJS_ID= 'ObjectEntitiesID'
test_silver_fp = 'res/test_silver.jsonl'
FROM_KG = 'from'
TO_KG = 'to'

TOKENIZER_PATH = f'{RES_DIR}/tokenizer/bert'
OPT_TOKENIZER_PATH = f'{RES_DIR}/tokenizer/opt'

check_dir_list = [
    BIN_DIR,
    OUTPUT_DIR,
    LOGGING_DIR,
]

for d in check_dir_list:
    if not os.path.exists(d):
        os.mkdir(d)

relation_entity_type_dict =  {
            "CompanyHasParentOrganisation": [
                "Company",
                "Company"
            ],
            "CountryBordersCountry": [
                "Country",
                "Country"
            ],
            "CountryHasOfficialLanguage": [
                "Country",
                "Language"
            ],
            "PersonHasNumberOfChildren": [
                "Person",
                "Number"
            ],
            "PersonHasPlaceOfDeath": [
                "Person",
                "City"
            ],
            "PersonHasProfession": [
                "Person",
                "Profession"
            ],
            "PersonPlaysInstrument": [
                "Person",
                "Instrument"
            ],
            "PersonSpeaksLanguage": [
                "Person",
                "Language"
            ],
            "RiverBasinsCountry": [
                "River",
                "Country"
            ],
            "SeriesHasNumberOfEpisodes": [
                "Series",
                "Number"
            ],
            "StateBordersState": [
                "State",
                "State"
            ],
            "BandHasMember": [
                "Band",
                "Person"
            ],
            "CityLocatedAtRiver": [
                "City",
                "River"
            ],
            "FootballerPlaysPosition": [
                "Person",
                "Position"
            ],
            "PersonCauseOfDeath": [
                "Person",
                "Cause"
            ],
            "PersonHasAutobiography": [
                "Person",
                "Autobiography"
            ],
            "PersonHasEmployer": [
                "Person",
                "Company"
            ],
            "PersonHasNoblePrize": [
                "Person",
                "Prize"
            ],
            "PersonHasSpouse": [
                "Person",
                "Person"
            ],
            "CountryHasStates": [
                "Country",
                "State"
            ],
            "CompoundHasParts": [
                "Compound",
                "Part"
            ]
            }