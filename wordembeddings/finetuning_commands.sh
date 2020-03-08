#!/bin/sh
python bert_finetuning.py -sentences='../data/openstack/utah/parsed/merged_templates/137k+18k_spr_injected_2_words' -output_dir='finetuning-models/137k+18k_spr_injected_2_words' -logfile_path='finetuning-models/137k+18k_spr_injected_2_words/log.txt'

python bert_finetuning.py -sentences='../data/openstack/utah/parsed/merged_templates/137k+18k_spr_injected_3_words' -output_dir='finetuning-models/137k+18k_spr_injected_3_words' -logfile_path='finetuning-models/137k+18k_spr_injected_3_words/log.txt'

python bert_finetuning.py -sentences='../data/openstack/utah/parsed/merged_templates/137k+18k_spr_injected_4_words' -output_dir='finetuning-models/137k+18k_spr_injected_4_words' -logfile_path='finetuning-models/137k+18k_spr_injected_4_words/log.txt'

python bert_finetuning.py -sentences='../data/openstack/utah/parsed/merged_templates/137k+18k_spr_injected_5_words' -output_dir='finetuning-models/137k+18k_spr_injected_5_words' -logfile_path='finetuning-models/137k+18k_spr_injected_5_words/log.txt'

python bert_finetuning.py -sentences='../data/openstack/utah/parsed/merged_templates/137k+18k_spr_injected_6_words' -output_dir='finetuning-models/137k+18k_spr_injected_6_words' -logfile_path='finetuning-models/137k+18k_spr_injected_6_words/log.txt'

python bert_finetuning.py -sentences='../data/openstack/utah/parsed/merged_templates/137k+18k_spr_removed_1_words' -output_dir='finetuning-models/137k+18k_spr_removed_1_words' -logfile_path='finetuning-models/137k+18k_spr_removed_1_words/log.txt'

python bert_finetuning.py -sentences='../data/openstack/utah/parsed/merged_templates/137k+18k_spr_removed_2_words' -output_dir='finetuning-models/137k+18k_spr_removed_2_words' -logfile_path='finetuning-models/137k+18k_spr_removed_2_words/log.txt'

python bert_finetuning.py -sentences='../data/openstack/utah/parsed/merged_templates/137k+18k_spr_removed_3_words' -output_dir='finetuning-models/137k+18k_spr_removed_3_words' -logfile_path='finetuning-models/137k+18k_spr_removed_3_words/log.txt'

python bert_finetuning.py -sentences='../data/openstack/utah/parsed/openstack_137k_plus_18k_sorted_per_request_templates' -output_dir='finetuning-models/137k_plus_18k_spr' -logfile_path='finetuning-models/137k_plus_18k_spr/log.txt'

python bert_finetuning.py -sentences='../data/openstack/sasho/parsed/logs_aggregated_normal_only_spr_templates' -output_dir='finetuning-models/sasho_normal_spr' -logfile_path='finetuning-models/sasho_normal_spr/log.txt'