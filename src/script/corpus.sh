#!/bin/usr/bash

root_path="/home/yd/project/LMKBC-2023"
cd  $root_path/res/additional_corpus/   

awk '!/<\//'  "enwiki-20230401-pages-articles-multistream.xml" | awk '!/xml:/' | awk '!/http/'  |awk -F '&lt;' '{print $1}' |  awk -F '&quot;' '{print $1}' |  awk -F '{{' '{print $1}' | awk 'length>200'  > preprocess_1.txt  

# awk '{ gsub(/\[\[/, " ", $0); print }' preprocess_1.txt | awk '{ gsub(/&quot;|&lt;|ref&gt;|span&gt;|sup&gt|&gt;|\|\//, " ", $0); print }' |   tr -s ' ' > preprocess_2.txt

# awk 'length>200' preprocess_2.txt | awk '!/http/' | awk  '{ gsub(/&quot;|&lt;|ref&gt;|span&gt;|sup&gt|&gt;|\|\//, " ", $0); print }' | tr -s ' '  > preprocess_3.txt
# sed 's/[^a-zA-Z0-9 :,.?;]//g'  |
# awk '{ gsub(/&quot;|&lt;|ref&gt;/, " ", $0); print }' | tr -s ' '  > preprocess_3.txt

# awk 'NR==FNR{a[/$0/];next} {for (i in a) if ($0 ~ i) print}' entity.txt cleanned.txt > contain_word.txt

# cat  preprocess_2.txt  | parallel --pipe -j 15 grep -Fw -f entity.txt > preprocess_3.txt
# sed 's/[^a-zA-Z0-9 :,.?;"']//g'  3.txt >
#  4.txt

# tr -s ' ' < 4.txt > 5.txt 