import json


line = '''
{
"exists": ["Karina Jelinek"], 
"sentence_tokens": [
"Since", "Leonardo", "Far", "##ia", "was", "the", "husband", "of", "television", "personality", "Karina Jelinek", ",", "the", "government", "tried", "to", "treat", "the", "programs", "revelation", "##s", "as", "a", "dispute", "between", "TV", "presenter", "##s", "on", "its", "controlled", "TV", "channels", "to", "disc", "##red", "##it", "it", ".", "l", "##t", ";", "re", "##f", "name", "##q", "##uo", "##t", ";", "Men", "##del", "##evich", "##15", "##8", "##q", "##uo", "##t", ";", "g", "##t", ";", "It", "hoped", "focusing", "Far", "##ia", "would", "def", "##lect", "attention", "from", "Ki", "##rch", "##ner", ".", "l", "##t", ";", "re", "##f", "##gt", ";"]}
{"exists": ["Joseph Ole Lenku"], "sentence_tokens": ["Joseph Ole Lenku", "assumed", "the", "office", "of", "the", "Governor", "on", "8", "August", "2017", ".", "Since", "taking", "office", "he", "has", "implemented", "a", "series", "of", "reforms", "in", "the", "County", "Government", "and", "has", "focused", "development", "efforts", "in", "the", "sectors", "of", "Health", ",", "Lands", "and", "Education", "."
]
}
'''

j = json.loads(line)

print(j)
