import os
import config


TR='token_redefine'
PFM='pretrain_fill_mask'
FM ='fill_mask'


def run_pretrain_filled_mask(para_dict):
    model_load_dir = config.bert_base_cased
    pfm_model_save_dir =f"bin/{para_dict['label']}/{PFM}/{para_dict['pretrain_model']}"
    pfm_model_best_dir = pfm_model_save_dir + "/best_ckpt"
    final_corpus_fn = f"res/wikidata/Country-Language-State/filter.json"
    # run_pretrain_filled_mask

    # final_corpus_fn = f"{config.RES_DIR}/additional_corpus/fm_pretrain_2.txt"

    cmd_pretrain_filled_mask = f"""
    
   python src/pre_model.py   --train_fn {final_corpus_fn}  --train_batch_size 32 --gpu 0   --train_epoch {para_dict['epoch']} --learning_rate 5e-5  --mask_strategy {para_dict['mask_strategy']} --model_load_dir {model_load_dir} --model_save_dir {pfm_model_save_dir} --model_best_dir  {pfm_model_best_dir}
    
    """
    print(cmd_pretrain_filled_mask)
    os.system(cmd_pretrain_filled_mask)
    return pfm_model_best_dir

def run_file_mask(para_dict):
    OUTPUT_FILE = f'output/filled-mask-valid.jsonl'

    fm_model_save_dir =f"bin/{para_dict['label']}/{FM}/{para_dict['pretrain_model']}"
    fm_model_best_dir = fm_model_save_dir + "/best_ckpt"
 

    cmd_run_fillmask = f"""
    
   python src/fm_model.py  --test_fn {config.VAL_FN} --template_fn res/prompts0.csv  --output_fn {OUTPUT_FILE}    --train_fn data/train.jsonl --train_batch_size 64 --gpu 0  --top_k 30 --threshold 0.1  --dev_fn  data/train_tiny.jsonl --mode "train test" --train_epoch {para_dict['epoch']} --learning_rate 5e-5 --model_load_dir {para_dict['model_load_dir']} --model_save_dir {fm_model_save_dir} --model_best_dir  {fm_model_best_dir}  
    
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
    pfm_epoch = 15
    fm_epoch = 50
    pretrain_model = config.bert_base_cased
    para_list=[
        {
            "pfm":{
                "mask_strategy":"random",
                "epoch":pfm_epoch,
                'pretrain_model':pretrain_model,
                'label':"random"
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
            },
            "fm":{
                 "epoch":fm_epoch,
                 'label':"single",
                'pretrain_model':pretrain_model,
            }
        },
        {
            "pfm":{
                "mask_strategy":"fold",
                 "epoch":pfm_epoch,
                'label':"fold",
                'pretrain_model':pretrain_model,
            },
               "fm":{
                 "epoch":fm_epoch,
                   'label':"fold",
                   'pretrain_model':pretrain_model,
            }
        }
    ]
    start_tasks(para_list)
        

if __name__ == "__main__":
    task_0()
               
