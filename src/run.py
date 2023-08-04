import os
import config


TR='token_redefine'
PFM='pretrain_fill_mask'
FM ='fill_mask'


def run_pretrain_sentence_validation(para_dict):
    model_load_dir = para_dict['pretrain_model']
    model_save_dir =f"bin/{para_dict['label']}/{PFM}/{para_dict['pretrain_model']}"
    model_best_dir = model_save_dir + "/best_ckpt"
    input_fp = para_dict['input_fp']

    cmd_pretrain_filled_mask = f"""
    
   python src/pre_sv_model.py   --train_fn {input_fp}  --train_batch_size 16 --gpu 0   --train_epoch {para_dict['epoch']} --learning_rate 5e-5  --mask_strategy {para_dict['mask_strategy']} --model_load_dir {model_load_dir} --model_save_dir {model_save_dir} --model_best_dir  {model_best_dir}
    
    """
    print(cmd_pretrain_filled_mask)
    os.system(cmd_pretrain_filled_mask)
    return model_best_dir



def run_pretrain_filled_mask(para_dict):
    model_load_dir = para_dict['pretrain_model']
    model_save_dir =f"bin/{para_dict['label']}/{PFM}/{para_dict['pretrain_model']}"
    model_best_dir = model_save_dir + "/best_ckpt"
    input_fp = para_dict['input_fp']

    cmd_pretrain_filled_mask = f"""
    
   python src/pre_fm_model.py   --train_fn {input_fp}  --train_batch_size 32 --gpu 0   --train_epoch {para_dict['epoch']} --learning_rate 5e-5  --mask_strategy {para_dict['mask_strategy']} --model_load_dir {model_load_dir} --model_save_dir {model_save_dir} --model_best_dir  {model_best_dir}
    
    """
    print(cmd_pretrain_filled_mask)
    os.system(cmd_pretrain_filled_mask)
    return model_best_dir

def run_file_mask(para_dict):
    OUTPUT_FILE = f'output/filled-mask-valid.jsonl'

    model_save_dir =f"bin/{para_dict['label']}/{FM}/{para_dict['pretrain_model']}"
    model_best_dir = model_save_dir + "/best_ckpt"


    cmd_run_fillmask = f"""
    
   python src/fm_model.py  --test_fn {config.VAL_FN} --template_fn res/prompts0.csv  --output_fn {OUTPUT_FILE}    --train_fn data/train.jsonl --train_batch_size 64 --gpu 0  --top_k 30 --threshold 0.1  --dev_fn  data/train_tiny.jsonl --mode "train test" --train_epoch {para_dict['epoch']} --pretrain_model {para_dict['pretrain_model']}   --learning_rate 5e-5 --model_load_dir {para_dict['model_load_dir']} --model_save_dir {model_save_dir} --model_best_dir  {model_best_dir}  
    
    """
    print(cmd_run_fillmask)
    # last_best_dir =  pfm_model_best_dir
    os.system(cmd_run_fillmask)

    
def start_tasks(para_list):
    for para in para_list:
        pfm = para['pfm']
        fm = para['fm']
        pfm_model_best_dir = run_pretrain_filled_mask(pfm)
        fm['model_load_dir'] = pfm_model_best_dir
        run_file_mask(fm)

def task_0():
    pfm_epoch = 20
    fm_epoch = 50
    # pfm_epoch = 0.001
    # fm_epoch = 0.001
    pretrain_model = config.bert_base_cased
    pfm_input_fp="res/wikidata/Country-Language-State/filter.json"
    
    para_list=[
        {
            "pfm":{
                "mask_strategy":"random",
                "epoch":pfm_epoch,
                'pretrain_model':pretrain_model,
                'label':"random",
                'input_fp':pfm_input_fp,
            },
               "fm":{
                 "epoch":fm_epoch,
                     'label':"random",
                   'pretrain_model':pretrain_model,
            }
        },
        {
            "pfm":{
                "mask_strategy":"single",
                 "epoch":pfm_epoch,
                'label':"single",
                'pretrain_model':pretrain_model,
                 'input_fp':pfm_input_fp,
            },
            "fm":{
                 "epoch":fm_epoch,
                 'label':"single",
                'pretrain_model':pretrain_model,
            }
        },
        # {
        #     "pfm":{
        #         "mask_strategy":"fold",
        #          "epoch":pfm_epoch,
        #         'label':"fold",
        #         'pretrain_model':pretrain_model,
        #          'input_fp':pfm_input_fp,
        #     },
        #        "fm":{
        #          "epoch":fm_epoch,
        #            'label':"fold",
        #            'pretrain_model':pretrain_model,
        #     }
        # }
    ]


    
    return para_list



def task_1():
    cv_epoch = 20
    pfm_epoch = 20
    fm_epoch = 50
    # cv_epoch = 0.0001
    # pfm_epoch = 0.0001
    # fm_epoch = 0.001
    label = 'pfm_sc_fm'
    pretrain_model = config.bert_base_cased
    pfm_input_fp="res/wikidata/Country-Language-State/filter.json"
    
    para_list=[
        {
            "pfm":{
                "mask_strategy":"random",
                 "epoch":pfm_epoch,
                'label':label,
                'pretrain_model':pretrain_model,
                 'input_fp':pfm_input_fp,
            },
            "sv":{
                "mask_strategy":"random",
                "epoch":cv_epoch,
                'pretrain_model':pretrain_model,
                'label':label,
                'input_fp':pfm_input_fp,
            },
               "fm":{
                 "epoch":fm_epoch,
                'label':label,
                'pretrain_model':pretrain_model,
            }
        }
    ]
    
    for para in para_list:
        sv = para['sv']
        fm = para['fm']
        pfm = para['pfm']
        pfm_model_best_dir = run_pretrain_filled_mask(pfm)
        sv['model_load_dir'] = pfm_model_best_dir
        sv_model_best_dir = run_pretrain_sentence_validation(sv)
        fm['model_load_dir'] = sv_model_best_dir
        run_file_mask(fm)

        

if __name__ == "__main__":
    task_1()
               
