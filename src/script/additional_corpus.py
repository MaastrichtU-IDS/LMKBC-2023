import csv
import json
import os
import random
import sys
import requests
from tqdm import tqdm
import transformers
import wikipedia
from glob import glob

parent_dir = os.path.abspath(os.path.join(os.getcwd(), 'src'))
print("parent path ", parent_dir)
print('cwd path', os.getcwd())
sys.path.append(parent_dir)

import config
import util
from multiprocessing import Process
from multiprocessing import Pool


enhance_tokenizer = transformers.AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)
origin_tokenizer = transformers.AutoTokenizer.from_pretrained(config.bert_base_cased)
additional_fp = config.TOKENIZER_PATH+"/added_tokens.json"
token_mapping_fp = config.RES_DIR+"/token_mapping.json"

def refresh_tokenizer():
    with open (additional_fp) as f:
        entity_set = json.load(f).keys()
    for entity in entity_set:
        entity = entity.strip()
        if entity not in origin_tokenizer.vocab:
            origin_tokenizer.add_tokens(entity.strip())
    origin_tokenizer.save_pretrained(config.TOKENIZER_PATH)

def build_entity_set_from_dataset(fns):
    entity_set = set()
    for fn in fns:
        with open(fn, "r") as file:
            for line in file:
                line_json = json.loads(line)
                object_entities = line_json['ObjectEntities']
                subject = line_json["SubjectEntity"]

                entity_set.add(subject)
                entity_set.update(object_entities)
    return entity_set


def extend_tokenizer(entity_set: set):
    for entity in entity_set:
        if entity not in enhance_tokenizer.get_vocab():
            enhance_tokenizer.add_tokens(entity)
    enhance_tokenizer.save_pretrained(config.TOKENIZER_PATH)
    return enhance_tokenizer


def get_additional_sentence(entity_set):
    for item in entity_set:
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
        data = requests.get(url).json()


def get_text(entity_list):
    content_list = []
    for entity in tqdm(entity_list):
        wiki_titles = wikipedia.search(entity)
        for title in wiki_titles:
            try:
                wiki_page = wikipedia.summary(title)
            except:
                # print("not wiki page")
                pass

            content = wiki_page
            content_list.append(content)
    local_sentence_fn = (
        f'{config.RES_DIR}/res/additional_corpus/{int(random.random*10000)}.txt'
    )
    with open(local_sentence_fn, 'w') as f:
        s = '\n'.join(content_list)
        f.writelines(s)


def country_state():
    r = 'CountryHasStates'
    states_fp = 'res/additional_corpus/states.csv'
    silver_fp = f'res/additional_corpus/silver/{r}.jsonl'
    entity_set=set()
    data_dict = dict()
    with open(states_fp) as f:
       csv_reader =  csv.DictReader(f)
       for row in csv_reader:
            country_name = row['country_name'].strip()
            state_name = row['name'].strip()
            if country_name not in data_dict:
                data_dict[country_name]=[]
            data_dict[country_name].append(state_name)
            entity_set.add(country_name)
            entity_set.add(state_name)
    extend_tokenizer(entity_set)
    data_list = []
    for country, states in data_dict.items():
        item = {
        config.KEY_SUB: country,
        config.KEY_REL:r,
        config.KEY_OBJS:states,
        }
        data_list.append(item)
    util.file_write_json_line(silver_fp, data_list)


def river():
    river_fp = 'res/additional_corpus/rivers.csv'
    entity_set=set()
    with open(river_fp) as f:
       csv_reader =  csv.DictReader(f, delimiter='    ')
       for row in csv_reader:
            country_name = row['Country']
            river_name = row['River']
            entity_set.update(country_name.strip().split(' '))
            entity_set.add(river_name)
    extend_tokenizer(entity_set)



def nobel_prize():
    r = 'PersonHasNoblePrize'
    origin_fp = 'res/additional_corpus/nobel_prize.jsonl'
    silver_fp = f'res/additional_corpus/silver/{r}.jsonl'
    json_lines = util.file_read_json_line(origin_fp)
    entity_set=set()
    '''
    {"Physics":"Robert B. Laughlin;Horst Ludwig Störmer;Daniel C. Tsui","Chemistry":"Walter Kohn;John Pople","Physiologyor Medicine":"Robert F. Furchgott;Louis Ignarro;Ferid Murad","Literature":"José Saramago","Peace":"John Hume;David Trimble","Economics":"Amartya Sen"}
{"Physics":"Gerard 't Hooft;Martinus J. G. Veltman","Chemistry":"Ahmed Zewail","Physiologyor Medicine":"Günter Blobel","Literature":"Günter Grass","Peace":"Médecins Sans Frontières","Economics":"Robert Mundell"}

    '''
    prize_dict = {
        "Physics":"Nobel Prize in Physics",
        "Chemistry":"Nobel Prize in Chemistry",
"Physiologyor Medicine":"Nobel Prize in Physiology or Medicine",
"Literature":"Nobel Prize in Literature",
"Peace":"Nobel Peace Prize",
"Economics":"Nobel Prize for Economics"
        }
    data_dict = dict()
    for row in json_lines:
        for k,v in row.items():
            if v in ('Cancelled due to World War II','-',None,'None','"—"'):
                continue
            names = v.split(';')
            names = [n.strip() for n in names]
            prize = prize_dict[k]
            for n in names:
                n=n.strip()
                if n in data_dict:
                    data_dict[n].append(prize)
                else:
                    data_dict[n]=[prize]
    data_list = [] 
    for name, prizes in data_dict.items():
        if len(prizes) > 4:
            continue
        item = {
        config.KEY_SUB: name,
        config.KEY_REL:r,
        config.KEY_OBJS:prizes,
        }
        data_list.append(item)

    util.file_write_json_line(silver_fp, data_list)
    entity_set.update(data_dict.keys())

    print(entity_set)
    extend_tokenizer(entity_set)

def official_language():

    # Country,Official,Regional language ,Minority language ,National language ,Widely spoken 
    official_language_fn = 'res/additional_corpus/official_language.csv'
    official_fn = 'res/additional_corpus/CountryHasOfficialLanguage.jsonl'
    entity_set=set()
    ds_list= []
    with open(official_language_fn) as f:
        csv_dict = csv.DictReader(f)
        for row in csv_dict: 
            country = row['Country'].strip()
            language = row['Official'].strip().split(' ')
            item = {
            config.KEY_SUB: country,
            config.KEY_REL:"CountryHasOfficialLanguage",
            config.KEY_OBJS: language,
            }
            ds_list.append(item)
            entity_set .add(country)
            entity_set.update(language)
    print(entity_set)  
    util.file_write_json_line(official_fn, ds_list      )    
    extend_tokenizer(entity_set)

def river_city():
    city_river_fp = 'res/additional_corpus/river_city.csv'
    relkation_fp = 'res/additional_corpus/CityLocatedAtRiver.jsonl'
    entity_set=set()
    ds_dict = dict()
    with open(city_river_fp) as f:
        csv_dict = csv.DictReader(f)
        for row in csv_dict: 
            river = row['riverLabel'].strip()
            cityLabel = row['cityLabel'].strip()
            entity_set .add(river)
            entity_set.add(cityLabel)
            if cityLabel not in ds_dict:
                ds_dict[cityLabel] = []
            ds_dict[cityLabel].append(river)
    ds_list=[]
    for k,v in ds_dict.items():
        item = {
            config.KEY_SUB: k,
            config.KEY_REL:"CityLocatedAtRiver",
            config.KEY_OBJS: v,
            }
        ds_list.append(item)
    util.file_write_json_line(relkation_fp,ds_list)

    # print(entity_set)    
    extend_tokenizer(entity_set)

def count_token_length():
    additional_fp = config.TOKENIZER_PATH+"/added_tokens.json"
    token_mapping_fp = config.RES_DIR+"/token_mapping.json"
    with open(additional_fp) as f:
        additional_entity_set = json.load(f).keys()
    print("additional_entity_set",len(additional_entity_set))
    token_count = dict()
    token_mapping = dict()
    token_conflict= 0 
    # token_frequency =  count_word()
    # for token in token_frequency:
    #     origin_tokenizer.add_tokens(token)
    for entity in tqdm(additional_entity_set):
        tokens = origin_tokenizer.tokenize(entity)
        # tokens = list(filter(lambda x:not x.startswith("##"),tokens))
        token_key = ' '.join(tuple(tokens[:3]))
        # token_mapping [token_key]= entity
        if token_key not in token_count:
            token_count[token_key] = 1
            token_mapping [token_key]= [entity]
        else:
            token_count[token_key] +=1
            token_mapping [token_key].append(entity)
    with open(token_mapping_fp,'w') as f:
        f.write(json.dumps(token_mapping,indent=4))
    token_conflict = list(sorted(filter(lambda x:x[1]>1, token_count.items()), key=lambda x:x[1],reverse=True)) 
    print("token_conflict",sum(map(lambda x:x[1],token_conflict)))
    print("token_conflict",token_conflict[:30])

def count_word():
    additional_fp = config.TOKENIZER_PATH+"/added_tokens.json"
    word_frequency_fp = config.RES_DIR+"/word_frequency.json"
    if os.path.exists(word_frequency_fp):
        with open (word_frequency_fp) as f:
            token_frequency = json.load(f)
        return token_frequency[:100]
    
    with open(additional_fp) as f:
        additional_entity_set = set(json.load(f).keys())
    
    gt_3_count=0
    token_mapping = dict()
    for entity in tqdm(additional_entity_set):
        words = entity.split(' ')
        for w in words:
            if w in origin_tokenizer.vocab:
                continue
            if w not in token_mapping:
                token_mapping[w] = 1
            token_mapping[w] += 1
    token_frequency = list(token_mapping.items())
    token_frequency.sort(key=lambda x:x[1],reverse=True)
    util.file_write_line(word_frequency_fp, json.dumps(token_frequency))
    token_frequency = token_frequency[:300]
    print("token_frequency",token_frequency)
    token_frequency = list(map(lambda x: x[0], token_frequency))
    print("token_frequency",token_frequency)
    return  token_frequency

def entitys():
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="bert-base-cased"
    )
    # entity_set = build_entity_set_from_dataset([config.TRAIN_FN, config.VAL_FN])
    # print("start building entity set")
    # tokenizer = extend_tokenizer(entity_set, bert_tokenizer)
    # print("start extend tokenizer")
    # tokenizer_path = f'{config.RES_PATH}/tokenizer/bert'
    # tokenizer.save_pretrained(tokenizer_path)
    # print("save tokenizer success")

def collect_specify_entity():
    collect_entity_pattern = {
        
        }
    for filename in glob(f'{config.RES_DIR}/silver/*.jsonl', recursive=True):
        print(filename)



if __name__ == "__main__":

    # nobel_prize()
    # official_language()
    # count_token_length()
    # count_word()
    # country_state()
    # refresh_tokenizer()
    # wiki_download()
    collect_specify_entity()
    # river_city()
