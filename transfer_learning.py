import argparse
import os
import subprocess

import matplotlib

matplotlib.use('Agg')
from settings import settings
from wordembeddings.transform_glove import merge_templates
import numpy as np
from logparser.utah_log_parser import *
from scipy.spatial.distance import cosine
import torch
from os import path
from numpy import percentile
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import logparser.Drain.Drain_demo as drain
import wordembeddings.transform_bert as transform_bert
from loganaliser.main import AnomalyDetection
from tools import distribution_plots as distribution_plots, calc_precision_utah as calc_precision_utah
from wordembeddings.bert_finetuning import finetune

predicted_labels_of_file_containing_anomalies = "predicted_labels_of_file_containing_anomalies"


def get_cosine_distance(lines_before_altering, lines_after_altering, templates):
    lines_before_as_bert_vectors = []
    lines_after_as_bert_vectors = []

    for sentence in lines_before_altering:
        idx = templates.index(sentence)
        if idx is None:
            raise ValueError("{} not found in template file".format(sentence))
        lines_before_as_bert_vectors.append(bert_vectors[idx])

    for sentence in lines_after_altering:
        idx = templates.index(sentence)
        if idx is None:
            raise ValueError("{} not found in template file".format(sentence))
        lines_after_as_bert_vectors.append(bert_vectors[idx])

    cosine_distances = []
    for before, after in zip(lines_before_as_bert_vectors, lines_after_as_bert_vectors):
        cosine_distances.append(cosine(before, after))
    write_lines_to_file(results_dir_experiment + "lines_before_after_cosine_distances.txt", cosine_distances, new_line=True)


def write_lines_to_file(file_path, content, new_line = False):
    file = open(file_path, 'w+')
    if new_line:
        [file.write(str(line) + "\n") for line in content]
    else:
        [file.write(str(line)) for line in content]
    file.close()


def inject_anomalies(anomaly_type, corpus_input, corpus_output, anomaly_indices_output_path, instance_information_in, instance_information_out, anomaly_amount):
    if anomaly_type in ["insert_words", "remove_words"]:
        if anomaly_type == "insert_words":
            lines_before_alter, lines_after_alter, anomalies_true = insert_words(corpus_input, corpus_output, anomaly_indices_output_path,
                                                                                 instance_information_in, instance_information_out, anomaly_amount)
        elif anomaly_type == "remove_words":
            lines_before_alter, lines_after_alter, anomalies_true = remove_words(corpus_input, corpus_output, anomaly_indices_output_path,
                                                                                 instance_information_in, instance_information_out, anomaly_amount)

        write_lines_to_file(results_dir_experiment + "lines_before_altering.txt", lines_before_alter)
        write_lines_to_file(results_dir_experiment + "lines_after_altering.txt", lines_after_alter)
        return anomalies_true, lines_before_alter, lines_after_alter

    elif anomaly_type == "duplicate_lines":
        anomalies_true = delete_or_duplicate_events(corpus_input, corpus_output, anomaly_indices_output_path, instance_information_in, instance_information_out, mode="dup")
    elif anomaly_type == "delete_lines":
        anomalies_true = delete_or_duplicate_events(corpus_input, corpus_output, anomaly_indices_output_path, instance_information_in, instance_information_out, mode="del")
    elif anomaly_type == "random_lines":
        anomalies_true = delete_or_duplicate_events(corpus_input, corpus_output, anomaly_indices_output_path, instance_information_in, instance_information_out, mode="ins")
    elif anomaly_type == "shuffle":
        anomalies_true = shuffle(corpus_input, corpus_output, instance_information_in, instance_information_out, anomaly_indices_output_path)
    else:
        print("anomaly type does not exist")
        raise

    return anomalies_true, None, None


def calculate_precision_and_plot(this_results_dir_experiment):
    # precision = calc_precision_utah(log_file_containing_anomalies=log_file_containing_anomalies,
    #                                 outliers_file=cwd + this_results_dir_experiment + 'outliers_values')
    distribution_plots(this_results_dir_experiment, args.epochs, args.seq_len, 768, 0)
    # subprocess.call(['tar', 'cvf', cwd + this_results_dir_experiment + "{}_epochs_{}_seq_len_{}_description:_{}_{}"
    #                 .format('bert', args.epochs, args.seq_len, args.anomaly_description, args.anomaly_amount) + '.tar',
    #                  '--directory=' + cwd + this_results_dir_experiment,
    #                  'normal_loss_values',
    #                  'anomaly_loss_values',
    #                  'outliers_values',
    #                  'anomaly_loss_indices',
    #                  'plot.png'])


def learning(arg, embeddings_path, epochs, instance_information_file):
    ad_normal = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                                 loadvectors=embeddings_path,
                                 savemodelpath=lstm_model_save_path,
                                 seq_length=arg.seq_len,
                                 num_epochs=epochs,
                                 n_hidden_units=arg.hiddenunits,
                                 n_layers=arg.hiddenlayers,
                                 embeddings_model='bert',
                                 train_mode=True,
                                 instance_information_file=instance_information_file)

    ad_normal.start_training()
    return ad_normal


def calculate_normal_loss(normal_lstm_model, results_dir, values_type):
    normal_loss_values = normal_lstm_model.loss_values(normal=True)

    os.makedirs(results_dir, exist_ok=True)
    normal_values_file = open(cwd + results_dir + values_type, 'w+')
    for val in normal_loss_values:
        normal_values_file.write(str(val) + "\n")
    normal_values_file.close()
    return normal_loss_values


def calculate_anomaly_loss(anomaly_lstm_model, results_dir, normal_loss_values, anomaly_loss_order,
                           anomaly_true_labels):
    anomaly_loss_values = anomaly_lstm_model.loss_values(normal=False)
    anomaly_loss_order = open(anomaly_loss_order, 'rb').readlines()
    anomaly_loss_order = [int(x) for x in anomaly_loss_order]

    assert len(anomaly_loss_order) == len(anomaly_loss_values)
    anomaly_loss_values_correct_order = [0] * len(anomaly_loss_order)
    for index, loss_val in zip(anomaly_loss_order, anomaly_loss_values):
        anomaly_loss_values_correct_order[index] = loss_val

    write_lines_to_file(results_dir_experiment + 'anomaly_loss_values', anomaly_loss_values_correct_order, new_line=True)

    per = percentile(normal_loss_values, 96.0)

    pred_outliers_indeces = [i for i, val in enumerate(anomaly_loss_values_correct_order) if val > per]
    pred_outliers_values = [val for val in anomaly_loss_values_correct_order if val > per]

    write_lines_to_file(results_dir_experiment + "pred_outliers_indeces.txt", pred_outliers_indeces, new_line=True)
    write_lines_to_file(results_dir_experiment + "pred_outliers_values.txt", pred_outliers_values, new_line=True)

    # produce labels for f1 score, precision, etc.
    pred_labels = np.zeros(len(anomaly_loss_values_correct_order), dtype=int)
    for anomaly_index in pred_outliers_indeces:
        pred_labels[anomaly_index] = 1

    true_labels = np.zeros(len(anomaly_loss_values_correct_order), dtype=int)
    for anomaly_index in anomaly_true_labels:
        true_labels[anomaly_index] = 1

    scores_file = open(results_dir_experiment + "scores.txt", "w+")
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)

    scores_file.write("F1-Score: {}\n".format(str(f1)))
    scores_file.write("Precision-Score: {}\n".format(str(precision)))
    scores_file.write("Recall-Score: {}\n".format(str(recall)))
    scores_file.write("Accuracy-Score: {}\n".format(str(accuracy)))
    scores_file.close()


# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------INITIALISE PARAMETERS---------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
cwd = os.getcwd() + "/"
parser = argparse.ArgumentParser()
parser.add_argument('-option', type=str, default='UtahSashoTransfer')
parser.add_argument('-seq_len', type=int, default=7)
parser.add_argument('-reduced', action='store_true')
parser.add_argument('-epochs', type=int, default=1)
parser.add_argument('-hiddenunits', type=int, default=250)
parser.add_argument('-hiddenlayers', type=int, default=4)
parser.add_argument('-transferlearning', action='store_true')
parser.add_argument('-anomaly_only', action='store_true')
parser.add_argument('-anomaly_description', type=str, default='None')
parser.add_argument('-corpus_anomaly_inputfile', type=str)
parser.add_argument('-instance_information_file_anomalies', type=str)
parser.add_argument('-bert_model_finetune', type=str, default='bert-base-uncased')
parser.add_argument('-finetune', action='store_true')
parser.add_argument('-anomaly_type', type=str, default='insert_words')
parser.add_argument('-anomaly_amount', type=int, default=8)
args = parser.parse_args()

option = args.option
results_dir_experiment = "{}_epochs_{}_seq_len:_{}_anomaly_type:{}_{}/".format(
    settings[option]["dataset_2"]["results_dir"] + 'bert',
    args.epochs, args.seq_len, args.anomaly_type, args.anomaly_amount)

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
instance_information_file_anomalies_injected_2 = settings[option]["dataset_2"][
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
finetuning_model_dir = "wordembeddings/finetuning-models/" + normal_1

if args.finetune:
    lstm_model_save_path = cwd + 'loganaliser/saved_models/' + normal_1 + '_with_finetune' + '_lstm.pth'
else:
    lstm_model_save_path = cwd + 'loganaliser/saved_models/' + normal_1 + '_lstm.pth'
vae_model_save_path = cwd + 'loganaliser/saved_models/' + normal_1 + '_vae.pth'

# take corpus parsed by drain, inject anomalies in this file
anomaly_injected_corpus_2 = cwd + anomalies_injected_dir_2 + anomaly_2 + "_" + args.anomaly_type
# put here the information about which line is an anomaly from anomaly injection
anomaly_indeces_2 = cwd + results_dir_experiment + "true_anomaly_labels.txt"
# anomaly_indeces_2 = cwd + anomaly_indeces_dir_2 + anomaly_2 + "_" + args.anomaly_type + "_" + args.anomaly_amount  + '_anomaly_indeces.txt'

# create all directories, if they don't exist yet
os.makedirs(results_dir, exist_ok=True)
os.makedirs(results_dir_experiment, exist_ok=True)
os.makedirs(raw_dir_1, exist_ok=True)
os.makedirs(raw_dir_2, exist_ok=True)
os.makedirs(parsed_dir_1, exist_ok=True)
os.makedirs(parsed_dir_2, exist_ok=True)
os.makedirs(embeddings_dir_1, exist_ok=True)
os.makedirs(embeddings_dir_2, exist_ok=True)
os.makedirs(anomalies_injected_dir_2, exist_ok=True)
os.makedirs(anomaly_indeces_dir_2, exist_ok=True)

### DRAIN PARSING
drain.execute(directory=raw_dir_1, file=normal_1, output=parsed_dir_1, logtype=logtype_1)
drain.execute(directory=raw_dir_2, file=normal_2, output=parsed_dir_2, logtype=logtype_2)
drain.execute(directory=raw_dir_2, file=anomaly_2, output=parsed_dir_2, logtype=logtype_2)

### INJECT ANOMALIES in dataset 2
anomalies_true, lines_before_alter, lines_after_alter = inject_anomalies(anomaly_type=args.anomaly_type, corpus_input=corpus_pre_anomaly_2,
                                                                         corpus_output=anomaly_injected_corpus_2,
                                                                         anomaly_indices_output_path=anomaly_indeces_2,
                                                                         instance_information_in=instance_information_file_anomalies_pre_inject_2,
                                                                         instance_information_out=instance_information_file_anomalies_injected_2,
                                                                         anomaly_amount=args.anomaly_amount)

# produce templates out of the corpuses that we have from the anomaly file
templates_normal_1 = list(set(open(corpus_normal_1, 'r').readlines()))
templates_normal_2 = list(set(open(corpus_normal_2, 'r').readlines()))
# merge_templates(templates_normal_1, templates_normal_2, merged_template_path=parsed_dir_1 + "_merged_templates_normal")
templates_anomalies_injected = list(set(open(anomaly_injected_corpus_2, 'r').readlines()))
merged_templates = merge_templates(templates_normal_1, templates_normal_2, templates_anomalies_injected, merged_template_path=None)
merged_templates = list(merged_templates)

if args.finetune:
    if not path.exists(finetuning_model_dir):
        finetune(templates=templates_normal_1, output_dir=finetuning_model_dir)

bert_vectors, _, _, _ = transform_bert.get_bert_vectors(merged_templates, bert_model=finetuning_model_dir)


if args.anomaly_type in ["insert_words", "remove_words"]:
    get_cosine_distance(lines_before_alter, lines_after_alter, merged_templates)

# transform output of bert into numpy word embedding vectors
transform_bert.transform(sentence_embeddings=bert_vectors,
                         logfile=corpus_normal_1,
                         templates=merged_templates,
                         outputfile=embeddings_normal_1)

transform_bert.transform(sentence_embeddings=bert_vectors,
                         logfile=corpus_normal_2,
                         templates=merged_templates,
                         outputfile=embeddings_normal_2)

transform_bert.transform(sentence_embeddings=bert_vectors,
                         logfile=anomaly_injected_corpus_2,
                         templates=merged_templates,
                         outputfile=embeddings_anomalies_injected_2)

if not args.anomaly_only:
    # NORMAL TRAINING with dataset 1
    ad_normal = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                                 loadvectors=embeddings_normal_1,
                                 savemodelpath=lstm_model_save_path,
                                 seq_length=args.seq_len,
                                 num_epochs=args.epochs,
                                 n_hidden_units=args.hiddenunits,
                                 n_layers=args.hiddenlayers,
                                 embeddings_model='bert',
                                 train_mode=True,
                                 instance_information_file=instance_information_file_normal_1)

    ad_normal.start_training()
# FEW SHOT TRAINING with dataset 2
ad_normal_transfer = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                                      loadvectors=embeddings_normal_2,
                                      savemodelpath=lstm_model_save_path,
                                      seq_length=args.seq_len,
                                      num_epochs=1,
                                      n_hidden_units=args.hiddenunits,
                                      n_layers=args.hiddenlayers,
                                      embeddings_model='bert',
                                      train_mode=True,
                                      instance_information_file=instance_information_file_normal_2,
                                      transfer_learning=True)
if not args.anomaly_only:
    ad_normal_transfer.start_training()

normal_loss_values = calculate_normal_loss(normal_lstm_model=ad_normal_transfer,
                                           results_dir=results_dir_experiment,
                                           values_type='normal_loss_values')
ad_anomaly = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                              loadvectors=embeddings_anomalies_injected_2,
                              savemodelpath=lstm_model_save_path,
                              seq_length=args.seq_len,
                              num_epochs=args.epochs,
                              n_hidden_units=args.hiddenunits,
                              n_layers=args.hiddenlayers,
                              embeddings_model='bert',
                              instance_information_file=instance_information_file_anomalies_injected_2,
                              anomalies_run=True,
                              results_dir=cwd + results_dir_experiment + 'anomaly_loss_indices')
calculate_anomaly_loss(anomaly_lstm_model=ad_anomaly, results_dir=results_dir_experiment,
                       normal_loss_values=normal_loss_values,
                       anomaly_loss_order=cwd + results_dir_experiment + 'anomaly_loss_indices',
                       anomaly_true_labels=anomalies_true)
calculate_precision_and_plot(results_dir_experiment)
