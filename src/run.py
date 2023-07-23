import argparse
import os
import config


import torch

TR='token_redefine'
PFM='pretrain_fill_mask'
FM ='fill_mask'

def run_token_redefine():
    # run_token_redefine
    global last_best_dir
    tr_model_save_dir = f"{config.BIN_DIR}/train_full/{TR}/{pretrain_model_name}"
    tr_model_best_dir = tr_model_save_dir + "/best_ckpt"


    cmd_token_redefine = f"""
   python {config.SRC_DIR}/tr_model.py    --train_batch_size 128 --gpu 0   --train_epoch 30 --learning_rate 3e-5  --model_load_dir {last_best_dir} --model_save_dir {tr_model_save_dir} --model_best_dir  {tr_model_best_dir}
    
    """
    last_best_dir =  tr_model_best_dir
    print(cmd_token_redefine)
    os.system(cmd_token_redefine)

def run_pretrain_filled_mask():
    global last_best_dir
    pfm_model_save_dir =f"{config.BIN_DIR}/train_full/{PFM}/{pretrain_model_name}"
    pfm_model_best_dir = pfm_model_save_dir + "/best_ckpt"

    # run_pretrain_filled_mask
    final_corpus_fn = f"{config.RES_DIR}/additional_corpus/fm_pretrain_2.txt"
    cmd_pretrain_filled_mask = f"""
    
   python {config.SRC_DIR}/pre_model.py   --train_fn {final_corpus_fn}  --train_batch_size 16 --gpu {args.gpu}   --train_epoch 40 --learning_rate 3e-5   --model_load_dir {last_best_dir} --model_save_dir {pfm_model_save_dir} --model_best_dir  {pfm_model_best_dir}
    
    """
    last_best_dir =  pfm_model_best_dir
    print(cmd_pretrain_filled_mask)
    os.system(cmd_pretrain_filled_mask)

def run_file_mask():
    OUTPUT_FILE = f'{config.OUTPUT_DIR}/filled-mask-valid.jsonl'

    fm_model_save_dir =f"{config.BIN_DIR}/train_full/{FM}/{pretrain_model_name}"
    fm_model_best_dir = fm_model_save_dir + "/best_ckpt"
    global last_best_dir
    global test_fn

    cmd_run_fillmask = f"""
    
   python {config.SRC_DIR}/fm_model.py  --test_fn {test_fn} --template_fn res/prompts0.csv  --output_fn {OUTPUT_FILE} --train_fn {config.DATA_DIR}/train.jsonl --train_batch_size 256 --gpu {args.gpu}  --top_k 30 --threshold 0.1  --dev_fn  {config.DATA_DIR}/train_tiny.jsonl --mode "train test" --train_epoch 50 --learning_rate 5e-5 --model_load_dir {last_best_dir} --model_save_dir {fm_model_save_dir} --model_best_dir  {fm_model_best_dir}
    
    """
    # last_best_dir =  pfm_model_best_dir
    print(cmd_run_fillmask)
    os.system(cmd_run_fillmask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Model with Question and Fill-Mask Prompts"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default='train_full',
        help="train test",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default='0',
        help="train test",
    )
    #pretrain_model_name = config.bert_tiny
    pretrain_model_name = config.bert_large_cased

    para_dict=dict()

    args = parser.parse_args()
    last_best_dir = pretrain_model_name
    test_fn = config.VAL_FN
    if "train_full" in args.mode:
        run_token_redefine()
        run_pretrain_filled_mask()
        run_file_mask()

    if "train_tr" in args.mode:
        run_token_redefine()
        run_file_mask()
        
    if "train_pfm" in args.mode:
        run_pretrain_filled_mask()
        run_file_mask()

    if "train_fm" in args.mode:
        run_file_mask()

    if "test" in args.mode:
        test_fn=args.test_fn 
        run_file_mask()
               
