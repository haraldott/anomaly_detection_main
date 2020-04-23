#!/bin/sh
# Normal learning finetuning
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="insert_words" -anomaly_amount=1
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="insert_words" -anomaly_amount=2 -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="insert_words" -anomaly_amount=3 -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="insert_words" -anomaly_amount=4 -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="insert_words" -anomaly_amount=5 -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="insert_words" -anomaly_amount=6 -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="insert_words" -anomaly_amount=7 -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="insert_words" -anomaly_amount=8 -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="insert_words" -anomaly_amount=9 -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="remove_words" -anomaly_amount=1 -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="remove_words" -anomaly_amount=2 -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="remove_words" -anomaly_amount=3 -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="replace_words" -anomaly_amount=1 -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="replace_words" -anomaly_amount=2 -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="replace_words" -anomaly_amount=3 -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="duplicate_lines" -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="delete_lines" -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="random_lines" -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="shuffle" -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="no_anomaly" -anomaly_only
CUDA_VISIBLE_DEVICES=1 python normal_learning.py -finetune -anomaly_type="reverse_order" -anomaly_only