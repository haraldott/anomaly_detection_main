#!/bin/sh
#python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_deleted_lines" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/anomalies/18k_spr_deleted.pickle" -anomaly_description="deleted_lines"
#
#python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_duplicated_lines" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/anomalies/18k_spr_duplicated.pickle" -anomaly_description="duplicated_lines" -anomaly_only
#
#python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_2_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_2_words" -anomaly_only
#
#python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_3_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_3_words" -anomaly_only
#
#python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_4_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_4_words" -anomaly_only
#
#python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_5_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_5_words" -anomaly_only
#
#python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_6_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_6_words" -anomaly_only
#
#python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_removed_1_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="removed_1_words" -anomaly_only
#
#python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_removed_2_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="removed_2_words" -anomaly_only
#
#python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_removed_3_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="removed_3_words" -anomaly_only
#
#python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_removed_3_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="removed_3_words" -anomaly_only
#
#python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_random_lines" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/anomalies/18k_spr_random_lines.pickle" -anomaly_description="random_lines" -anomaly_only


###########################################################################
# WITH FINETUNE
###########################################################################
#CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_deleted_lines" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/anomalies/18k_spr_deleted.pickle" -anomaly_description="deleted_lines" -bert_model_finetune='wordembeddings/finetuning-models/137k_plus_18k_spr'
#
#CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_duplicated_lines" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/anomalies/18k_spr_duplicated.pickle" -anomaly_description="duplicated_lines"  -bert_model_finetune='wordembeddings/finetuning-models/137k_plus_18k_spr'
#
#CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_2_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_2_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k+18k_spr_injected_2_words'
#
#CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_3_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_3_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k+18k_spr_injected_3_words'
#
#CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_4_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_4_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k+18k_spr_injected_4_words'
#
#CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_5_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_5_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k+18k_spr_injected_5_words'
#
#CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_6_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_6_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k+18k_spr_injected_6_words'
#
#CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_removed_1_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="removed_1_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k_plus_18k_spr'
#
#CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_removed_2_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="removed_2_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k_plus_18k_spr'
#
#CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_removed_3_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="removed_3_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k_plus_18k_spr'
#
#CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_removed_3_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="removed_3_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k_plus_18k_spr'
#
#CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_random_lines" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/anomalies/18k_spr_random_lines.pickle" -anomaly_description="random_lines"  -bert_model_finetune='wordembeddings/finetuning-models/137k+18k_random_lines'


CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_deleted_lines" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/anomalies/18k_spr_deleted.pickle" -anomaly_description="deleted_lines" -bert_model_finetune='wordembeddings/finetuning-models/sasho_combined'

CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_duplicated_lines" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/anomalies/18k_spr_duplicated.pickle" -anomaly_description="duplicated_lines"  -bert_model_finetune='wordembeddings/finetuning-models/137k_plus_18k_spr'

CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_2_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_2_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k+18k_spr_injected_2_words'

CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_3_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_3_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k+18k_spr_injected_3_words'

CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_4_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_4_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k+18k_spr_injected_4_words'

CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_5_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_5_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k+18k_spr_injected_5_words'

CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_6_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="injected_6_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k+18k_spr_injected_6_words'

CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_removed_1_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="removed_1_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k_plus_18k_spr'

CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_removed_2_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="removed_2_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k_plus_18k_spr'

CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_removed_3_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="removed_3_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k_plus_18k_spr'

CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_spr_removed_3_words" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle" -anomaly_description="removed_3_words"  -bert_model_finetune='wordembeddings/finetuning-models/137k_plus_18k_spr'

CUDA_VISIBLE_DEVICES=1 python meta_script_bert.py -corpus_anomaly_inputfile="data/openstack/utah/parsed/anomalies_injected/18k_random_lines" -instance_information_file_anomalies="data/openstack/utah/raw/sorted_per_request_pickle/anomalies/18k_spr_random_lines.pickle" -anomaly_description="random_lines"  -bert_model_finetune='wordembeddings/finetuning-models/137k+18k_random_lines'