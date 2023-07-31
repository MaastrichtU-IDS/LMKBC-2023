

import csv
import json
import os
import random
import sys
import requests
from tqdm import tqdm
import transformers
import wikipedia

parent_dir = os.path.abspath(os.path.join(os.getcwd(), 'src'))
print("parent path ", parent_dir)
print('cwd path', os.getcwd())
sys.path.append(parent_dir)

import config
import util
from multiprocessing import Process
from multiprocessing import Pool

import json

silver_dir = f"{config.RES_DIR}/silver"


def extract(item_dict):
    fp = item_dict['fp']
    subject_name=item_dict['subject_name']
    object_name=item_dict['object_name']
    predicate=item_dict['predicate']
    sub_id_name = subject_name[:-5]
    obj_id_name = object_name[:-5]

    with open(fp) as f:
        json_list = json.load(f)

    res_dict=dict()
    entity_ids_dict=dict()
    has_id=False
    for item in json_list:
        sub = item[subject_name]
        obj = item[object_name]
        # if sub_id_name in item:
        #     has_id =True
        # if has_id:
        #     entity_ids_dict[sub] = extract_id( item[sub_id_name])
        #     entity_ids_dict[obj] = extract_id(item[obj_id_name])

        if sub not in res_dict:
            res_dict[sub] = set() 
        res_dict[sub].add(obj)
    
    res_list=[]
    for k,v in res_dict.items():
        item = {
            config.KEY_SUB:k,
            config.KEY_OBJS:v,
            config.KEY_REL:predicate
            }
        # if has_id:
        #     item[config.KEY_OBJS_ID]= [entity_ids_dict[vi] for vi in v],
        #     item[config.KEY_SUB_ID]=entity_ids_dict[k]

        res_list.append(item)
    print("size of predicate is ",len(res_list))
    return res_list



CountryBordersCountry = {
    "fp":f"{config.RES_DIR}/additional_corpus/CountriesBorderCountries.json",
    "subject_name":"country1Label",
    "object_name":"country2Label",
    "predicate":"CountryBordersCountry"
    }

FootballerPlaysPosition = {
    "fp":f"{config.RES_DIR}/additional_corpus/FootballPlayPosition_0_50000.json",
    "subject_name":"footballerLabel",
    "object_name":"positionLabel",
    "predicate":"FootballerPlaysPosition"
    }


BandHasMember = {
    "fp":f"{config.RES_DIR}/additional_corpus/BandHasMember.json",
    "subject_name":"bandLabel",
    "object_name":"memberLabel",
    "predicate":"BandHasMember"
    }

CompanyHasParentOrganisation = {
    "fp":f"{config.RES_DIR}/additional_corpus/CompanyHasParents.json",
    "subject_name":"companyLabel",
    "object_name":"parentOrgLabel",
    "predicate":"CompanyHasParentOrganisation"
    }

CompoundHasParts = {
    "fp":f"{config.RES_DIR}/additional_corpus/CompoundHasParts.json",
    "subject_name":"compoundLabel",
    "object_name":"partLabel",
    "predicate":"CompoundHasParts"
    }

PersonCauseOfDeath = {
    "fp":f"{config.RES_DIR}/additional_corpus/PersonCauseOfDeath_0_30000.json",
    "subject_name":"personLabel",
    "object_name":"causeOfDeathLabel",
    "predicate":"PersonCauseOfDeath"
    }

PersonHasAutobiography = {
    "fp":f"{config.RES_DIR}/additional_corpus/PersonHasAutobiography.json",
    "subject_name":"personLabel",
    "object_name":"autobiography",
    "predicate":"PersonHasAutobiography"
    }

PersonHasEmployer = {
    "fp":f"{config.RES_DIR}/additional_corpus/PersonHasEmployer_0_20000.json",
    "subject_name":"personLabel",
    "object_name":"employerLabel",
    "predicate":"PersonHasEmployer"
    }

PersonHasPlaceOfDeath = {
    "fp":f"{config.RES_DIR}/additional_corpus/PersonHasPlaceOfDeath_0_50000.json",
    "subject_name":"personLabel",
    "object_name":"placeOfDeathLabel",
    "predicate":"PersonHasPlaceOfDeath"
    }


PersonPlaysInstrument = {
    "fp":f"{config.RES_DIR}/additional_corpus/PersonPlaysInstrument0_50000.json",
    "subject_name":"personLabel",
    "object_name":"instrumentLabel",
    "predicate":"PersonPlaysInstrument"
    }

PersonSpeaksLanguage = {
    "fp":f"{config.RES_DIR}/additional_corpus/PersonSpeaksLanguage.json",
    "subject_name":"personLabel",
    "object_name":"languageLabel",
    "predicate":"PersonSpeaksLanguage"
    }

RiverBasinsCountry = {
    "fp":f"{config.RES_DIR}/additional_corpus/RiverBasinCountr_0_50000y.json",
    "subject_name":"riverBasinLabel",
    "object_name":"countryLabel",
    "predicate":"RiverBasinsCountry"
    }

SeriesHasNumberOfEpisodes = {
    "fp":f"{config.RES_DIR}/additional_corpus/SeriesHasNumberOfEpisodes.json",
    "subject_name":"seriesLabel",
    "object_name":"numEpisodes",
    "predicate":"SeriesHasNumberOfEpisodes"
    }
PersonHasProfession = {
    "fp":f"{config.RES_DIR}/additional_corpus/PersonHasProfession.json",
    "subject_name":"personLabel",
    "object_name":"professionLabel",
    "predicate":"PersonHasProfession"
    }


CityLocatedAtRiver ={
    "fp":f"{config.RES_DIR}/additional_corpus/CityLocatedAtRiver.json",
    "subject_name":"cityLabel",
    "object_name":"riverLabel",
    "predicate":"CityLocatedAtRiver",
    
    }

pre_list = [
        CountryBordersCountry,
        FootballerPlaysPosition,
    CityLocatedAtRiver,
    BandHasMember,
    CompanyHasParentOrganisation,
    CompoundHasParts,
    PersonCauseOfDeath,
    PersonHasAutobiography,
    PersonHasEmployer,
    PersonHasPlaceOfDeath,
    PersonPlaysInstrument,
    PersonSpeaksLanguage,
    RiverBasinsCountry,
    SeriesHasNumberOfEpisodes,
PersonHasProfession,
CityLocatedAtRiver,
        ]

def generate_silver_json():
    for p in pre_list:
        res_list = extract(p)
        pred_fp = silver_dir+f'/{p["predicate"]}.jsonl'
        util.file_write_json_line(pred_fp, res_list)

def extract_id(aUrl):
    return aUrl.split('/')[-1]

def generate_wiki_id_map():
    wiki_list=[] 
    for p in pre_list:
        p['subject_name'] = p['subject_name'].replace('Label','')
        p['object_name'] = p['object_name'].replace('Label','')
        res_list = extract(p)
        wiki_list.extend(res_list)
    entity_id_dict = dict()
    for p in wiki_list:
        entity_id_dict[p[config.KEY_SUB]] = extract_id(p[config.KEY_SUB])
        for e in p[config.KEY_OBJS]:
            entity_id_dict[e] = extract_id(e)
        p['object_name'] = p['object_name'].replace('Label','')

    pred_fp = silver_dir+f'/{p["predicate"]}.jsonl'
    util.file_write_json_line(pred_fp, wiki_list)


if __name__ == "__main__":
    generate_silver_json()
        



    
        