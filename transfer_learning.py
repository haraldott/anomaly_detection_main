import argparse
import os
import subprocess

import matplotlib

matplotlib.use('Agg')
from settings import settings

import numpy as np
from logparser.utah_log_parser import *

import logparser.Drain.Drain_demo as drain
import wordembeddings.transform_bert as transform_bert
from loganaliser.main import AnomalyDetection
from tools import distribution_plots as distribution_plots, calc_precision_utah as calc_precision_utah
from wordembeddings.bert_finetuning import finetune

predicted_labels_of_file_containing_anomalies = "predicted_labels_of_file_containing_anomalies"


def inject_anomalies(anomaly_type, corpus_input, corpus_output,
                     anomaly_indices_output_path, instance_information_in, instance_information_out, anomaly_amount):
    if anomaly_type == "insert_words":
        insert_words(corpus_input, corpus_output, anomaly_indices_output_path,
                     instance_information_in, instance_information_out, anomaly_amount)
    elif anomaly_type == "remove_words":
        remove_words(corpus_input, corpus_output, anomaly_indices_output_path,
                     instance_information_in, instance_information_out, anomaly_amount)
    elif anomaly_type == "duplicate_lines":
        delete_or_duplicate_events(corpus_input, corpus_output, anomaly_indices_output_path,
                                   instance_information_in, instance_information_out, mode="dup")
    elif anomaly_type == "delete_lines":
        delete_or_duplicate_events(corpus_input, corpus_output, anomaly_indices_output_path,
                                   instance_information_in, instance_information_out,  mode="del")
    elif anomaly_type == "random_lines":
        delete_or_duplicate_events(corpus_input, corpus_output, anomaly_indices_output_path,
                                   instance_information_in, instance_information_out, mode="ins")
    elif anomaly_type == "shuffle":
        shuffle(corpus_input, corpus_output,
                instance_information_in, instance_information_out, anomaly_indices_output_path)
    else:
        print("anomaly type does not exist")
        raise

def calculate_precision_and_plot(this_results_dir_experiment):
    # precision = calc_precision_utah(log_file_containing_anomalies=log_file_containing_anomalies,
    #                                 outliers_file=cwd + this_results_dir_experiment + 'outliers_values')
    distribution_plots(this_results_dir_experiment, args.epochs, args.seq_len, 768, 0)
    subprocess.call(['tar', 'cvf', cwd + this_results_dir_experiment + "{}_epochs_{}_seq_len_{}_description:_{}_{}"
                                    .format('bert', args.epochs, args.seq_len, args.anomaly_description, args.anomaly_amount) + '.tar',
                     '--directory=' + cwd + this_results_dir_experiment,
                     'normal_loss_values',
                     'anomaly_loss_values',
                     'outliers_values',
                     'anomaly_loss_indices',
                     'plot.png'])


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

    mean = np.mean(normal_loss_values)
    std = np.std(normal_loss_values)
    cut_off = std * 3
    lower, upper = mean - cut_off, mean + cut_off
    os.makedirs(results_dir, exist_ok=True)
    normal_values_file = open(cwd + results_dir + values_type, 'w+')
    for val in normal_loss_values:
        normal_values_file.write(str(val) + "\n")
    normal_values_file.close()
    return lower, upper


def calculate_anomaly_loss(anomaly_lstm_model, results_dir, lo, up):
    anomaly_loss_values = anomaly_lstm_model.loss_values(normal=False)

    anomaly_values_file = open(cwd + results_dir + 'anomaly_loss_values', 'w+')
    for val in anomaly_loss_values:
        anomaly_values_file.write(str(val) + "\n")
    anomaly_values_file.close()
    outliers = []
    for i, x in enumerate(anomaly_loss_values):
        if x < lo or x > up:
            outliers.append(str(i + args.seq_len) + "," + str(x))

    outliers_values_file = open(cwd + results_dir + 'outliers_values', 'w+')
    for val in outliers:
        outliers_values_file.write(str(val) + "\n")
    outliers_values_file.close()


#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------INITIALISE PARAMETERS---------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
cwd = os.getcwd() + "/"
parser = argparse.ArgumentParser()
parser.add_argument('-option', type=str, default='UtahSashoTransfer')
parser.add_argument('-seq_len', type=int, default=7)
parser.add_argument('-reduced', action='store_true')
parser.add_argument('-epochs', type=int, default=100)
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
parser.add_argument('-anomaly_amount', type=int, default=1)
args = parser.parse_args()

option = args.option
results_dir_experiment = "{}_epochs_{}_seq_len:_{}_anomaly_type:{}_{}/".format(settings[option]["dataset_2"]["results_dir"] + 'bert',
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
instance_information_file_anomalies_pre_inject_2 = settings[option]["dataset_2"]['instance_information_file_anomalies_pre_inject']
instance_information_file_anomalies_injected_2 = settings[option]["dataset_2"]['instance_information_file_anomalies_injected'] + anomaly_2 + "_" + args.anomaly_type + "_" + str(args.anomaly_amount)

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
    lstm_model_save_path = cwd + 'loganaliser/saved_models/' + normal_1 + '_with_finetune' + '_lstm.pth'
else:
     lstm_model_save_path = cwd + 'loganaliser/saved_models/' + normal_1 + '_lstm.pth'
vae_model_save_path = cwd + 'loganaliser/saved_models/' + normal_1 + '_vae.pth'

# take corpus parsed by drain, inject anomalies in this file
anomaly_injected_corpus_2 = cwd + anomalies_injected_dir_2 + anomaly_2 + "_" + args.anomaly_type
# put here the information about which line is an anomaly from anomaly injection
anomaly_indeces_2 = cwd + results_dir_experiment + anomaly_2 + "_" + args.anomaly_type + "_" + str(args.anomaly_amount)  + '_anomaly_indeces.txt'
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
inject_anomalies(anomaly_type=args.anomaly_type, corpus_input=corpus_pre_anomaly_2, corpus_output=anomaly_injected_corpus_2,
                 anomaly_indices_output_path=anomaly_indeces_2, instance_information_in=instance_information_file_anomalies_pre_inject_2,
                 instance_information_out=instance_information_file_anomalies_injected_2, anomaly_amount=args.anomaly_amount)

# produce templates out of the corpuses that we have from the anomaly file
templates_normal_1 = list(set(open(corpus_normal_1, 'r').readlines()))
templates_normal_2 = list(set(open(corpus_normal_2, 'r').readlines()))
templates_anomalies_injected = list(set(open(anomaly_injected_corpus_2, 'r').readlines()))

if args.finetune:
    finetune(templates=templates_normal_1, output_dir="wordembeddings/finetuning-models/" + normal_1)

bert_vectors_normal_1, _, _, _ = transform_bert.get_bert_vectors(templates_normal_1, bert_model="wordembeddings/finetuning-models/" + normal_1)
bert_vectors_normal_2, _, _, _ = transform_bert.get_bert_vectors(templates_normal_2, bert_model="wordembeddings/finetuning-models/" + normal_1)
bert_vectors_anomalies_injected, _, _, _ = transform_bert.get_bert_vectors(templates_anomalies_injected, bert_model="wordembeddings/finetuning-models/" + normal_1)

# transform output of bert into numpy word embedding vectors
transform_bert.transform(sentence_embeddings=bert_vectors_normal_1,
                         logfile=corpus_normal_1,
                         templates=templates_normal_1,
                         outputfile=embeddings_normal_1)

transform_bert.transform(sentence_embeddings=bert_vectors_normal_2,
                         logfile=corpus_normal_2,
                         templates=templates_normal_2,
                         outputfile=embeddings_normal_2)


transform_bert.transform(sentence_embeddings=bert_vectors_anomalies_injected,
                         logfile=anomaly_injected_corpus_2,
                         templates=templates_anomalies_injected,
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
                             num_epochs=5,
                             n_hidden_units=args.hiddenunits,
                             n_layers=args.hiddenlayers,
                             embeddings_model='bert',
                             train_mode=True,
                             instance_information_file=instance_information_file_normal_2,
                             transfer_learning=True)
if not args.anomaly_only:
    ad_normal_transfer.start_training()

lower, upper = calculate_normal_loss( normal_lstm_model=ad_normal_transfer,
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
calculate_anomaly_loss(anomaly_lstm_model=ad_anomaly, results_dir=results_dir_experiment, lo=lower, up=upper)
calculate_precision_and_plot(results_dir_experiment)
