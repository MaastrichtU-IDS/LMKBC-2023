from transformers import pipeline
fill_mask = pipeline("fill-mask", model="bert-base-cased")

result=fill_mask("Paris is the [MASK] [MASK] [MASK] of France.")
# [{'score': 0.7, 'sequence': 'Paris is the capital of France.'},
# {'score': 0.2, 'sequence': 'Paris is the birthplace of France.'},
# {'score': 0.1, 'sequence': 'Paris is the heart of France.'}]
print(result)
