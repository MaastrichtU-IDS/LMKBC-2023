from transformers import pipeline, AutoTokenizer

from transformers import BertModel

bert = BertModel.from_pretrained()()
bert.forward()

tokenizer = AutoTokenizer.from_pretrained(
    'facebook/opt-350m', padding_side='left', max_length=128
)
generator = pipeline(
    'text-generation',
    model='bin/text-generation/facebook/opt-350m/best_ckpt',
    tokenizer=tokenizer,
    max_new_tokens=128,
)

while True:
    input_sentence = input()
    if input_sentence == "exit":
        break
    resp = generator(input_sentence)
    print(resp)
    '''
Which countries border Suriname?

Algeria is bordered to the northeast by Tunisia; to the east by Libya; to the southeast by Niger; to the southwest by Mali, Mauritania, and Western Sahara; to the west by Morocco; and to the north by the Mediterranean Sea. Which countries border Algeria?
    
    '''
