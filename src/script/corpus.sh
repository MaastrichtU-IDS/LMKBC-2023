#!/bin/usr/bash

cd  res/additional_corpus/

awk '!/<\//'  "enwiki-20230401-pages-articles-multistream.xml" | awk -F '{{' '{print $1}' | awk 'length>100'  | sed 's/[^a-zA-Z0-9 :,.?;]//g'  | tr -s ' ' > result.txt






awk '!/xml:/'  result.txt > cleanned.txt
| 
awk 'NR==FNR{a[/$0/];next} {for (i in a) if ($0 ~ i) print}' entity.txt cleanned.txt > contain_word.txt

cat res/additional_corpus/cleanned.txt | parallel --pipe -j 15 grep -Fw -f res/additional_corpus/entity.txt > res/additional_corpus/contain_words.txt
# sed 's/[^a-zA-Z0-9 :,.?;"']//g'  3.txt >
#  4.txt

# tr -s ' ' < 4.txt > 5.txt 