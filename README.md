
# Expanding the Vocabulary of BERT for Knowledge Base Construction

Knowledge Base Construction from Pre-trained Language Models (LM-KBC) 2nd Edition

This repository contains dataset for the LM-KBC challenge at ISWC 2023.

## run pretrain task with task_recode

```
python src/pre_fm_model.py   --train_fn res/pretrain.jsonl  --train_batch_size 16 --gpu  0  --train_epoch 20 --learning_rate 2e-5  --mask_strategy random  --model_load_dir bert-base-cased --model_save_dir bin/pretrain --model_best_dir  bin/pretrain/best_ckpt --token_recode true 

```

## run fine-tune task for valid set

```
python src/fm_model.py  --test_fn data/test.jsonl --valid_fn data/val.jsonl  --template_fn res/prompts0.csv  --output output/run-valid.jsonl --train_fn data/train.jsonl --train_batch_size 64 --gpu -1 --top_k 40  --train_epoch 20 --learning_rate 2e-5 --model_load_dir  bin/pretrain/best_ckpt --model_save_dir bin/fine-tune  --model_best_dir bin/fine-tune/best_ckpt  --pretrain_model bert-base-cased   --do_train true  --do_valid true  --do_test false   

```

## run fine-tune task for test set

```
python src/fm_model.py  --test_fn data/test.jsonl --valid_fn data/val.jsonl  --template_fn res/prompts0.csv  --output output/run-valid.jsonl --train_fn data/train.jsonl --train_batch_size 64 --gpu -1 --top_k 40  --train_epoch 20 --learning_rate 2e-5 --model_load_dir  bin/pretrain/best_ckpt --model_save_dir bin/fine-tune  --model_best_dir bin/fine-tune/best_ckpt  --pretrain_model bert-base-cased   --do_train true  --do_valid false  --do_test true  
```


### Baselines

baselines are released by official

Running instructions for the Huggingface baselines:
 - For BERT

```python src/baseline.py  --input data/val.jsonl --threshold 0.1 --fill_mask_prompts res/prompts.csv --question_prompts res/question-prompts.csv  --output output/testrun-bert-token_recode.jsonl --train_data data/train.jsonl --model bert-base-cased --batch_size 32 --gpu 0 --token_recode 1 ```

 - For OPT-1.3b

```python src/baseline.py  --input data/val.jsonl --fill_mask_prompts res/prompts.csv --question_prompts res/question-prompts.csv  --output output/testrun-opt.jsonl --train_data data/train.jsonl --model facebook/opt-1.3b --batch_size 8 --gpu 0```

### Evaluation script

Run instructions evaluation script:
  ```

  python src/evaluate.py -p data/val.jsonl -g output/text-generation/test-opt-1.3b.jsonl
  
  ```

The first parameter hereby indicates the prediction file, the second the ground truth file.
