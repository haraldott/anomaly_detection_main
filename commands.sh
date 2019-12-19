#!/bin/bash

python meta_script_bert.py -hiddenunits=768 -epochs=1
python meta_script.py -embeddingsize=100 -hiddenunits=100 -epochs=80
python meta_script.py -embeddingsize=100 -hiddenunits=100 -epochs=100
python meta_script.py -embeddingsize=100 -hiddenunits=100 -epochs=150

python meta_script.py -embeddingsize=200 -hiddenunits=200 -epochs=80
python meta_script.py -embeddingsize=200 -hiddenunits=200 -epochs=100
python meta_script.py -embeddingsize=200 -hiddenunits=200 -epochs=150

python meta_script.py -embeddingsize=250 -hiddenunits=250 -epochs=80
python meta_script.py -embeddingsize=250 -hiddenunits=250 -epochs=100
python meta_script.py -embeddingsize=250 -hiddenunits=250 -epochs=150

python meta_script_bert.py -hiddenunits=768 -epochs=50
python meta_script_bert.py -hiddenunits=768 -epochs=75
python meta_script_bert.py -hiddenunits=768 -epochs=100
python meta_script_bert.py -hiddenunits=768 -epochs=150