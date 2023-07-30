

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

def extract(fp, subject_name, object_name, predicate):
    with open(fp) as f:
        json_list = json.load(f)
    res_dict=dict()
    pred_fp = silver_dir+f'/{predicate}.jsonl'
    for item in json_list:
        sub = item[subject_name]
        obj = item[object_name]
        if sub not in res_dict:
            res_dict[sub] = [] 
        res_dict[sub].append(obj)
    
    res_list=[]
    for k,v in res_dict.items():
        item = {
            config.KEY_SUB:k,
            config.KEY_OBJS:v,
            config.KEY_REL:predicate
            }
        res_list.append(item)
    print("size of predicate is ",len(res_list))
    util.file_write_json_line(pred_fp, res_list)


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
    "predicate":"CityLocatedAtRiver"
    }


def generate_silver_json():
    pre_list = [
        
    CityLocatedAtRiver

        ]

    for p in pre_list:
        extract(**p)

def generate_wiki_id_map():

    pre_list = [
        
    CityLocatedAtRiver

        ]

    for p in pre_list:
        extract(**p)
        



    
        