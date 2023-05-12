# Knowledge Base Construction from Pre-trained Language Models (LM-KBC) 2nd Edition

This repository contains dataset for the LM-KBC challenge at ISWC 2023.

## Dataset v0.9

Preliminary release of the LM-KBC dataset, evaluation script, GPT-baseline



### Baselines

As baselines, we provide a script that can run masked LMs and causal LMs from Huggingface in the baseline.py, use these to generate entity surface forms, and use a Wikidata API for entity disambiguation.

Furthermore, we also provide a GPT-3 baseline that directly predicts Wikidata identifiers.

Running instructions for the Huggingface baselines:
For BERT

```python baseline.py  --input data/val.jsonl --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output testrun-bert.jsonl --train_data data/train.jsonl --model bert-large-cased --batch_size 32 --gpu 0```

For OPT-1.3b

```python baseline.py  --input data/val.jsonl --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output testrun-opt.jsonl --train_data data/train.jsonl --model facebook/opt-1.3b --batch_size 8 --gpu 0```

Run instructions GPT-3 baseline:

 ```python baseline-GPT3-IDs-directly.py" --input data/val.jsonl --output data/testrun-GPT3.jsonl -k YOUR_OPENAI_KEY_HERE```
 
### Evaluation script

Run instructions evaluation script:
  * ```python evaluate.py -p data/val.jsonl -g data/testrun-XYZ.jsonl```

The first parameter hereby indicates the prediction file, the second the ground truth file.

### Note

The released dataset is primarily for understanding the format. We are making a few quality checks and the changes to the final version will be minor. The final dataset will be added here within a few days.

### Coming soon

- More baseline scripts (GPT-3 baseline w/ Wikidata entity disambiguation)
- Dataset V1 with further cleaning
