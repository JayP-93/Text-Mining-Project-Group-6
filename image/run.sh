#!/usr/bin/env bash

python /code/monolingual_cv.py /input/IT > /output/datasets/monolingual_word_emb_it.txt
python /code/monolingual_cv.py /input/EN > /output/datasets/monolingual_word_emb_en.txt
python /code/monolingual_cv.py /input/DE > /output/datasets/monolingual_word_emb_de.txt
python /code/monolingual_cv.py /input/CZ > /output/datasets/monolingual_word_emb_cz.txt
python /code/multi_lingual_no_langfeat.py /input > /output/datasets/multilingual_word_emb_no_langfeat_en.txt
python /code/multi_lingual.py /input > /output/dataset/multilingual_word_emb_langfeat_en.txt
python /code/IdeaPOC.py > /output/datasets/mono_multi_cross.txt

exec python "$@"