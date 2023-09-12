import os
import config

print(os.getcwd())
TR='token_redefine'
PFM='pretrain_fill_mask'
FM ='fill_mask'


def run_pretrain_sentence_validation(para_dict):
    model_load_dir = para_dict['pretrain_model']
    model_save_dir =f"bin/{para_dict['label']}/{PFM}/{para_dict['pretrain_model']}"
    model_best_dir = model_save_dir + "/best_ckpt"
    input_fp = para_dict['input_fp']

    cmd_pretrain_filled_mask = f"""
    
   python src/pre_sv_model.py   --train_fn {input_fp}  --train_batch_size 16 --gpu 0   --train_epoch {para_dict['epoch']} --learning_rate 5e-5  --mask_strategy {para_dict['mask_strategy']} --model_load_dir {model_load_dir} --model_save_dir {model_save_dir} --model_best_dir  {model_best_dir}  --token_recode {para_dict['token_recode']} 
    
    """
    print(cmd_pretrain_filled_mask)
    os.system(cmd_pretrain_filled_mask)
    return model_best_dir



def run_pretrain_filled_mask(para_dict:dict):
    model_load_dir = para_dict['pretrain_model']
    model_save_dir =f"bin/{para_dict['label']}/{PFM}/{para_dict['pretrain_model']}"
    model_best_dir = model_save_dir + "/best_ckpt"
    input_fp = para_dict['input_fp']

    cmd_pretrain_filled_mask = f"""
    
   python src/pre_fm_model.py   --train_fn {input_fp}  --train_batch_size 16 --gpu 0   --train_epoch {para_dict['epoch']} --learning_rate 5e-5  --model_load_dir {model_load_dir} --model_save_dir {model_save_dir} --model_best_dir  {model_best_dir}   --token_recode {para_dict.get('token_recode',False)}
    
    """
    print(cmd_pretrain_filled_mask)
    os.system(cmd_pretrain_filled_mask)
    return model_best_dir

def run_file_mask(para_dict):
    OUTPUT_FILE = f'output/filled-mask-valid.jsonl'

    model_save_dir =f"bin/{para_dict['label']}/{FM}/{para_dict['pretrain_model']}"
    model_best_dir = model_save_dir + "/best_ckpt"


    cmd_run_fillmask = f"""
    
   python src/fm_model.py  --test_fn {config.test_fp}  --valid_fn {config.VAL_FN}    --template_fn res/prompts0.csv  --output_fn {OUTPUT_FILE}    --train_fn data/train.jsonl --train_batch_size 64 --gpu 0  --top_k 30 --threshold 0.1  --dev_fn  data/train_tiny.jsonl  --do_train true  --do_valid true  --do_test false  --token_recode {para_dict.get('token_recode',False)} --train_epoch {para_dict['epoch']} --pretrain_model {para_dict['pretrain_model']}   --learning_rate 5e-5 --model_load_dir {para_dict['model_load_dir']} --model_save_dir {model_save_dir} --model_best_dir  {model_best_dir}   --filter {para_dict.get('filter',False)}  --label  {para_dict['label']}  --recode_type std
    
    """
    print(cmd_run_fillmask)
    # last_best_dir =  pfm_model_best_dir
    os.system(cmd_run_fillmask)

name_function_dict = {
    "pfm":run_pretrain_filled_mask,
    "fm":run_file_mask,
    }
def start_tasks(para_list):
    for tasks in para_list:
        model_load_dir = None
        for task_dict in tasks:
            if model_load_dir is not None:
                task_dict['model_load_dir']=model_load_dir
            else:
                task_dict['model_load_dir']=task_dict['pretrain_model']
            func = name_function_dict[task_dict['task']]
            model_load_dir = func(task_dict)
     

def task_3():
    pfm_epoch = 5
    fm_epoch = 5
    # pfm_epoch = 0.00001
    # fm_epoch = 0.001
    pretrain_model = config.bert_large_cased
    pfm_input_fp="res/wikidata/Country-Language-State/filter.json"
    
    task_list=[
            [
              {
                   "task":"fm",
                   "epoch":fm_epoch,
                   'label':"token_recode_fm",
                   'pretrain_model':pretrain_model,
                   'token_recode':True,
                   'model_load_dir':pretrain_model,
          
                } 
            ],
            [
                {
                "task":"fm",
                 "epoch":fm_epoch,
                'label':"baseline",
                   'pretrain_model':pretrain_model,
                   'model_load_dir':pretrain_model,
         
                }
            ],
               
    ]   
    return task_list

if __name__ == "__main__":
    task_list = task_3()
    start_tasks(task_list)
               
