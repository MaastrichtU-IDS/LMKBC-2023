from transformers import AutoTokenizer
from transformers.utils import PaddingStrategy

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

batch_sentences = ["Hello world", "My name is Bing", "I live in Redmond"]

# Tokenize the batch of sentences
batch_tokenized = tokenizer(batch_sentences, padding=True, truncation=True)
tokenizer.model_input_names = [
    "input_ids",
    "token_type_ids",
    "attention_mask",
    'label_ids',
]
print(batch_tokenized)

batch = [
    {"input_ids": [1, 2, 3], "label_ids": [2, 3, 4]},
    {"input_ids": [1, 2, 3, 4, 5, 6], "label_ids": [2, 3, 4, 5, 6, 7]},
    {
        "input_ids": [
            1,
        ],
        "label_ids": [
            2,
        ],
    },
]
padded = tokenizer.pad(batch, padding=PaddingStrategy.LONGEST)
# padded = tokenizer.pad(batch, padding=PaddingStrategy.LONGEST, return_tensors='pt')
print(padded)
