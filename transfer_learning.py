import argparse

import matplotlib

matplotlib.use('Agg')
from settings import settings
from wordembeddings.transform_glove import merge_templates
import logparser.Drain.Drain_demo as drain
import wordembeddings.transform_bert as transform_bert
from loganaliser.main import AnomalyDetection
from wordembeddings.bert_finetuning import finetune
from shared_functions import calculate_precision_and_plot, determine_anomalies, \
    get_cosine_distance, inject_anomalies, get_labels_from_corpus, transfer_labels
import os
from wordembeddings.visualisation import write_to_tsv_files_bert_sentences
from shared_functions import get_embeddings

predicted_labels_of_file_containing_anomalies = "predicted_labels_of_file_containing_anomalies"

# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------INITIALISE PARAMETERS--------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
cwd = os.getcwd() + "/"
parser = argparse.ArgumentParser()
parser.add_argument('-option', type=str, default='UtahSashoTransfer')
parser.add_argument('-seq_len', type=int, default=7)
parser.add_argument('-n_layers', type=int, default=1)
parser.add_argument('-n_hidden_units', type=int, default=128)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-clip', type=int, default=1.22)
parser.add_argument('-epochs', type=int, default=120)
parser.add_argument('-anomaly_only', action='store_true')
parser.add_argument('-finetune', action='store_true')
parser.add_argument('-anomaly_type', type=str, default='random_lines')
parser.add_argument('-anomaly_amount', type=int, default=0)
parser.add_argument('-embeddings_model', type=str, default="bert")
parser.add_argument('-label_encoder', type=str, default=None)
parser.add_argument('-experiment', type=str, default='default')
args = parser.parse_args()

print("starting {} {}".format(args.anomaly_type, args.anomaly_amount))
option = args.option

if args.finetune:
    results_dir = settings[option]["dataset_2"]["results_dir"] + "_finetune/"
else:
    results_dir = settings[option]["dataset_2"]["results_dir"] + "/"

results_dir_experiment = "{}_epochs_{}_seq_len:_{}_anomaly_type:{}_{}/".format(
    results_dir + args.embeddings_model, args.epochs, args.seq_len, args.anomaly_type, args.anomaly_amount)

normal_1 = settings[option]["dataset_1"]["raw_normal"]
normal_2 = settings[option]["dataset_2"]["raw_normal"]
anomaly_2 = settings[option]["dataset_2"]["raw_anomaly"]

raw_dir_1 = settings[option]["dataset_1"]["raw_dir"]
raw_dir_2 = settings[option]["dataset_2"]["raw_dir"]

parsed_dir_1 = settings[option]["dataset_1"]["parsed_dir"]
parsed_dir_2 = settings[option]["dataset_2"]["parsed_dir"]

results_dir = settings[option]["dataset_2"]["results_dir"]

embeddings_dir_1 = settings[option]["dataset_1"]["embeddings_dir"]
embeddings_dir_2 = settings[option]["dataset_2"]["embeddings_dir"]

logtype_1 = settings[option]["dataset_1"]["logtype"]
logtype_2 = settings[option]["dataset_2"]["logtype"]

instance_information_file_normal_1 = settings[option]["dataset_1"]['instance_information_file_normal']
instance_information_file_normal_2 = settings[option]["dataset_2"]['instance_information_file_normal']
instance_information_file_anomalies_pre_inject_2 = settings[option]["dataset_2"][
    'instance_information_file_anomalies_pre_inject']
instance_information_file_anomalies_injected_2 = \
    settings[option]["dataset_2"][
        'instance_information_file_anomalies_injected'] + anomaly_2 + "_" + args.anomaly_type + "_" + str(
        args.anomaly_amount)

anomalies_injected_dir_2 = parsed_dir_2 + "anomalies_injected/"
anomaly_indeces_dir_2 = parsed_dir_2 + "anomalies_injected/anomaly_indeces/"

# corpus files produced by Drain
corpus_normal_1 = cwd + parsed_dir_1 + normal_1 + '_corpus'
corpus_normal_2 = cwd + parsed_dir_2 + normal_2 + '_corpus'
corpus_pre_anomaly_2 = cwd + parsed_dir_2 + anomaly_2 + '_corpus'

# bert vectors as pickle files
embeddings_normal_1 = cwd + embeddings_dir_1 + normal_1 + '.pickle'
embeddings_normal_2 = cwd + embeddings_dir_2 + normal_2 + '.pickle'
embeddings_anomalies_injected_2 = cwd + embeddings_dir_2 + anomaly_2 + '.pickle'

if args.finetune:
    finetuning_model_dir = "wordembeddings/finetuning-models/" + normal_1
    lstm_model_save_path = cwd + 'loganaliser/saved_models/' + normal_1 + '_with_finetune' + '_lstm.pth'
else:
    finetuning_model_dir = "bert-base-uncased"
    lstm_model_save_path = cwd + 'loganaliser/saved_models/' + normal_1 + '_lstm.pth'

# take corpus parsed by drain, inject anomalies in this file
anomaly_injected_corpus_2 = cwd + anomalies_injected_dir_2 + anomaly_2 + "_" + args.anomaly_type
# put here the information about which line is an anomaly from anomaly injection
anomaly_indeces_2 = cwd + results_dir_experiment + "true_anomaly_labels.txt"
# anomaly_indeces_2 = cwd + anomaly_indeces_dir_2 + anomaly_2 + "_" + args.anomaly_type + "_" + args.anomaly_amount  + '_anomaly_indeces.txt'

# create all directories, if they don't exist yet
os.makedirs(results_dir, exist_ok=True)
os.makedirs(results_dir_experiment, exist_ok=True)
os.makedirs(parsed_dir_1, exist_ok=True)
os.makedirs(parsed_dir_2, exist_ok=True)
os.makedirs(embeddings_dir_1, exist_ok=True)
os.makedirs(embeddings_dir_2, exist_ok=True)
os.makedirs(anomalies_injected_dir_2, exist_ok=True)
os.makedirs(anomaly_indeces_dir_2, exist_ok=True)

### DRAIN PARSING
if not os.path.exists(corpus_normal_1) or not os.path.exists(corpus_normal_2) or not os.path.exists(corpus_pre_anomaly_2):
    drain.execute(directory=raw_dir_1, file=normal_1, output=parsed_dir_1, logtype=logtype_1)
    drain.execute(directory=raw_dir_2, file=normal_2, output=parsed_dir_2, logtype=logtype_2)
    drain.execute(directory=raw_dir_2, file=anomaly_2, output=parsed_dir_2, logtype=logtype_2)

#pre_process_log_events(corpus_pre_anomaly_2, corpus_normal_1, corpus_normal_2)

### INJECT ANOMALIES in dataset 2
anomaly_lines, lines_before_alter, lines_after_alter = \
    inject_anomalies(anomaly_type=args.anomaly_type, corpus_input=corpus_pre_anomaly_2,
                     corpus_output=anomaly_injected_corpus_2, anomaly_indices_output_path=anomaly_indeces_2,
                     instance_information_in=instance_information_file_anomalies_pre_inject_2,
                     instance_information_out=instance_information_file_anomalies_injected_2,
                     anomaly_amount=args.anomaly_amount, results_dir=results_dir_experiment)

# produce templates out of the corpuses that we have from the anomaly file
templates_normal_1 = list(set(open(corpus_normal_1, 'r').readlines()))
templates_normal_2 = list(set(open(corpus_normal_2, 'r').readlines()))
# merge_templates(templates_normal_1, templates_normal_2, merged_template_path=parsed_dir_1 + "_merged_templates_normal")
templates_anomalies_injected = list(set(open(anomaly_injected_corpus_2, 'r').readlines()))
merged_templates = merge_templates(templates_normal_1, templates_normal_2, templates_anomalies_injected,
                                   merged_template_path=None)
merged_templates = list(merged_templates)

if args.finetune:
    if not os.path.exists(finetuning_model_dir):
        finetune(templates=templates_normal_1, output_dir=finetuning_model_dir)

word_embeddings = get_embeddings(args.embeddings_model, merged_templates)

write_to_tsv_files_bert_sentences(word_embeddings=word_embeddings,
                                  tsv_file_vectors=results_dir_experiment + "visualisation/vectors.tsv",
                                  tsv_file_sentences=results_dir_experiment + "visualisation/sentences.tsv")

if args.anomaly_type in ["insert_words", "remove_words", "replace_words"]:
    get_cosine_distance(lines_before_alter, lines_after_alter, results_dir_experiment, word_embeddings)

# transform output of bert into numpy word embedding vectors
transform_bert.transform(sentence_embeddings=word_embeddings, logfile=corpus_normal_1, outputfile=embeddings_normal_1)

transform_bert.transform(sentence_embeddings=word_embeddings, logfile=corpus_normal_2, outputfile=embeddings_normal_2)

transform_bert.transform(sentence_embeddings=word_embeddings, logfile=anomaly_injected_corpus_2,
                         outputfile=embeddings_anomalies_injected_2)

target_normal_labels, n_classes, normal_label_embeddings_map, normal_template_class_map =\
    get_labels_from_corpus(normal_corpus=open(corpus_normal_1, 'r').readlines(), encoder_path=args.label_encoder,
                           templates=templates_normal_1, embeddings=word_embeddings)

dataset_2_corpus_target_labels = transfer_labels(dataset1_templates=templates_normal_1,
                                                 dataset2_templates=templates_normal_2,
                                                 dataset2_corpus=corpus_normal_2, word_embeddings=word_embeddings,
                                                 template_class_mapping=normal_template_class_map,
                                                 results_dir=results_dir_experiment)

if not args.anomaly_only:
    # NORMAL TRAINING with dataset 1
    ad_normal = AnomalyDetection(loadvectors=embeddings_normal_1,
                                 savemodelpath=lstm_model_save_path,
                                 seq_length=args.seq_len,
                                 num_epochs=args.epochs,
                                 n_hidden_units=args.n_hidden_units,
                                 n_layers=args.n_layers,
                                 embeddings_model='bert',
                                 train_mode=True,
                                 instance_information_file=instance_information_file_normal_1,
                                 n_classes=n_classes,
                                 clip=args.clip,
                                 results_dir=results_dir_experiment,
                                 batch_size=args.batch_size,
                                 target_labels=target_normal_labels)

    ad_normal.start_training()
# FEW SHOT TRAINING with dataset 2
ad_normal_transfer = AnomalyDetection(loadvectors=embeddings_normal_2,
                                      savemodelpath=lstm_model_save_path,
                                      seq_length=args.seq_len,
                                      num_epochs=5,
                                      n_hidden_units=args.n_hidden_units,
                                      n_layers=args.n_layers,
                                      embeddings_model='bert',
                                      train_mode=True,
                                      instance_information_file=instance_information_file_normal_2,
                                      transfer_learning=True,
                                      n_classes=n_classes,
                                      clip=args.clip,
                                      results_dir=results_dir_experiment,
                                      batch_size=args.batch_size,
                                      target_labels=dataset_2_corpus_target_labels)
if not args.anomaly_only:
    ad_normal_transfer.start_training()

ad_anomaly = AnomalyDetection(loadvectors=embeddings_anomalies_injected_2,
                              savemodelpath=lstm_model_save_path,
                              seq_length=args.seq_len,
                              num_epochs=args.epochs,
                              n_hidden_units=args.n_hidden_units,
                              n_layers=args.n_layers,
                              embeddings_model='bert',
                              instance_information_file=instance_information_file_anomalies_injected_2,
                              anomalies_run=True,
                              results_dir=cwd + results_dir_experiment,
                              batch_size=args.batch_size,
                              clip=args.clip,
                              target_labels=dataset_2_corpus_target_labels, #TODO: eigentlich sollte für anomaly hier nichts geladen werden, da es eh nicht benutzt wird
                              n_classes=n_classes)

determine_anomalies(anomaly_lstm_model=ad_anomaly, results_dir=results_dir_experiment,
                    order_of_values_of_file_containing_anomalies=cwd + results_dir_experiment + 'anomaly_label_indices',
                    lines_that_have_anomalies=anomaly_lines, normal_label_embedding_mapping=normal_label_embeddings_map,
                    corpus_of_log_containing_anomalies=anomaly_injected_corpus_2,
                    set_embeddings_of_log_containing_anomalies=word_embeddings)

print("done.")
calculate_precision_and_plot(results_dir_experiment, args, cwd)
