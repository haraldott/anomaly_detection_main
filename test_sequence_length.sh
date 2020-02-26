#!/bin/sh
for i in {1..6}
do
  python meta_script_bert.py -seq_len=$i
done