import argparse
import os
import subprocess

import matplotlib

matplotlib.use('Agg')
import settings

import numpy as np

import logparser.Drain.Drain_demo as drain
import wordembeddings.transform_bert as transform_bert
import wordembeddings.transform_glove as transform_glove
from loganaliser.main import AnomalyDetection
from tools import distribution_plots as distribution_plots, calc_precision_utah as calc_precision_utah
from wordembeddings.bert_finetuning import finetune

predicted_labels_of_file_containing_anomalies = "predicted_labels_of_file_containing_anomalies"


def calculate_precision_and_plot(this_results_dir_experiment, log_file_containing_anomalies):
    # precision = calc_precision_utah(log_file_containing_anomalies=log_file_containing_anomalies,
    #                                 outliers_file=cwd + this_results_dir_experiment + 'outliers_values')
    distribution_plots(this_results_dir_experiment, args.epochs, args.seq_len, 768, 0)
    subprocess.call(['tar', 'cvf', cwd + this_results_dir_experiment + "{}_epochs_{}_seq_len_{}_description:_{}"
                                    .format('bert', args.epochs, args.seq_len, args.anomaly_description) + '.tar',
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
parser.add_argument('-option', type=str, default='UtahSorted137')
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
args = parser.parse_args()

option = args.option
results_dir_experiment = "{}_epochs_{}_seq_len:_{}_anomaly_type:{}/".format(settings.settings[option]["resultsdir"] + 'bert',
                                                                            args.epochs, args.seq_len, args.anomaly_description)
combinedinputfile = settings.settings[option]["combinedinputfile"]
anomalyinputfile = settings.settings[option]["anomalyinputfile"]
normalinputfile = settings.settings[option]["normalinputfile"]
inputdir = settings.settings[option]["inputdir"]
parseddir = settings.settings[option]["parseddir"]
resultsdir = settings.settings[option]["resultsdir"]
embeddingspickledir = settings.settings[option]["embeddingspickledir"]
embeddingsdir = settings.settings[option]["embeddingsdir"]
logtype = settings.settings[option]["logtype"]
instance_information_file_normal = settings.settings[option]['instance_information_file_normal']
instance_information_file_anomalies = settings.settings[option]['instance_information_file_anomalies']
parseddir_untouched = parseddir + "untouched/"
anomalies_injected_dir = parseddir + "anomalies_injected/"
anomaly_indeces_dir = parseddir + "anomalies_injected/anomaly_indeces/"

# create all directories, if they don't exist yet
os.makedirs(resultsdir, exist_ok=True)
os.makedirs(results_dir_experiment, exist_ok=True)
os.makedirs(inputdir, exist_ok=True)
os.makedirs(parseddir, exist_ok=True)
os.makedirs(embeddingsdir, exist_ok=True)
os.makedirs(parseddir_untouched, exist_ok=True)
os.makedirs(anomalies_injected_dir, exist_ok=True)
os.makedirs(anomaly_indeces_dir, exist_ok=True)

templates_normal = cwd + parseddir + normalinputfile + '_templates'
templates_anomaly = cwd + parseddir + anomalyinputfile + '_templates'
templates_added = cwd + parseddir + combinedinputfile + '_templates'
templates_merged = cwd + parseddir + combinedinputfile + '_merged_templates'

corpus_normal_inputfile = cwd + parseddir + normalinputfile + '_corpus'
corpus_anomaly_inputfile = cwd + parseddir + anomalyinputfile + '_corpus'
corpus_combined_file = cwd + parseddir + combinedinputfile + '_corpus'

embeddingsfile_for_transformer = cwd + embeddingsdir + combinedinputfile + '_vectors.txt'
embeddingsfile_for_glove = '../' + embeddingsdir + combinedinputfile + '_vectors'

embeddings_normal = cwd + embeddingspickledir + normalinputfile + '.pickle'
embeddings_anomalies = cwd + embeddingspickledir + anomalyinputfile + '.pickle'
vae_model_save_path = cwd + 'loganaliser/saved_models/' + normalinputfile + '_vae.pth'
lstm_model_save_path = cwd + 'loganaliser/saved_models/' + normalinputfile + '_lstm.pth'

corpus_anomaly_outputfile = cwd + parseddir + anomalies_injected_dir + anomalyinputfile + '_anomalies_injected_corpus'
anomalies_injected_indeces = cwd + anomaly_indeces_dir + anomalyinputfile + '_anomaly_indeces.txt'

# if parameters come from args, use them, otherwise leave them as they are
if args.corpus_anomaly_inputfile: corpus_anomaly_inputfile = args.corpus_anomaly_inputfile
if args.instance_information_file_anomalies: instance_information_file_anomalies = args.instance_information_file_anomalies


# produce templates out of the corpuses that we have from the anomaly file
templates_anomaly = list(set(open(corpus_anomaly_inputfile, 'r').readlines()))
transform_glove.merge_templates(templates_normal, templates_anomaly,
                                merged_template_path=templates_merged)

if args.finetune:
    finetune(templates=templates_merged, output_dir=args.bert_model_finetune)

bert_vectors, _, _, _ = transform_bert.get_bert_vectors(templates_merged, bert_model=args.bert_model_finetune)

# transform output of bert into numpy word embedding vectors
if not args.anomaly_only:
    transform_bert.transform(sentence_embeddings=bert_vectors,
                             logfile=corpus_normal_inputfile,
                             templatefile=templates_merged,
                             outputfile=embeddings_normal)

transform_bert.transform(sentence_embeddings=bert_vectors,
                         logfile=corpus_anomaly_inputfile,
                         templatefile=templates_merged,
                         outputfile=embeddings_anomalies)


# -------------------------------------------------------------------------------------------------------
# --------------------------------NORMAL LEARNING--------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
if not args.transferlearning:

    # learning on normal data
    ad_normal = AnomalyDetection(loadautoencodermodel=vae_model_save_path, loadvectors=embeddings_normal, savemodelpath=lstm_model_save_path,
                                 seq_length=args.seq_len, num_epochs=args.epochs, n_hidden_units=args.hiddenunits,
                                 n_layers=args.hiddenlayers, embeddings_model='bert', train_mode=True,
                                 instance_information_file=instance_information_file_normal)
    if not args.anomaly_only:
        ad_normal.start_training()
        # calculate loss on normal data and log
    lower, upper = calculate_normal_loss(normal_lstm_model=ad_normal,
                          results_dir=results_dir_experiment,
                          values_type='normal_loss_values')
    # predict on data containing anomaly and log
    ad_anomaly = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                                  loadvectors=embeddings_anomalies,
                                  savemodelpath=lstm_model_save_path,
                                  seq_length=args.seq_len,
                                  num_epochs=args.epochs,
                                  n_hidden_units=args.hiddenunits,
                                  n_layers=args.hiddenlayers,
                                  embeddings_model='bert',
                                  instance_information_file=instance_information_file_anomalies,
                                  anomalies_run=True,
                                  results_dir=cwd + results_dir_experiment + 'anomaly_loss_indices')
    calculate_anomaly_loss(anomaly_lstm_model=ad_anomaly, results_dir=results_dir_experiment, lo=lower, up=upper)
    calculate_precision_and_plot(this_results_dir_experiment=results_dir_experiment,
                                 log_file_containing_anomalies=inputdir + anomalyinputfile)
# -------------------------------------------------------------------------------------------------------
# ---------------------------------TRANSFER LEARNING-----------------------------------------------------
# -------------------------------------------------------------------------------------------------------


else:
    # initialise paths
    normalinputfile_transfer = settings.settings[option]["normalinputfile_transfer"]
    embeddingspickledir_transfer = settings.settings[option]["embeddingspickledir_transfer"]
    embeddings_normal_transfer = cwd + embeddingspickledir_transfer + normalinputfile_transfer + '.pickle'
    embeddings_anomalies_transfer = cwd + embeddingspickledir + anomalyinputfile + '.pickle'
    results_dir_experiment_transfer = "{}_epochs_{}_seq_len_{}/".format(
        settings.settings[option]["resultsdir_transfer"] + 'bert', args.epochs, args.seq_len)
    anomalyfile_transfer = settings.settings[option]["inputdir_transfer"] + settings.settings[option][
        "anomalyinputfile_transfer"]
    instance_information_file_normal_transfer = settings.settings[option]["instance_information_file_normal_transfer"]
    instance_information_file_anomalies_transfer = settings.settings[option][
        "instance_information_file_anomalies_transfer"]

    transfer_input_file = "data/openstack/sasho/parsed/logs_aggregated_normal_only_spr_corpus"
    trasfer_input_file_template = "data/openstack/sasho/parsed/logs_aggregated_normal_only_spr_templates"
    transfer_instance_information = "data/openstack/sasho/raw/sorted_per_request_pickle/logs_aggregated_normal_only_spr.pickle"
    # get bert vectors for dataset 2
    bert_vectors, _, _, _ = transform_bert.get_bert_vectors(trasfer_input_file_template, bert_model=args.bert_model_finetune)
    transform_bert.transform(sentence_embeddings=bert_vectors,
                             logfile=transfer_input_file,
                             templatefile=trasfer_input_file_template,
                             outputfile=embeddings_normal_transfer)

    # NORMAL TRAINING with dataset 1
    ad_normal = learning(args, embeddings_normal, args.epochs, instance_information_file_normal)
    # FEW SHOT TRAINING with dataset 2
    ad_normal_transfer = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                                 loadvectors=embeddings_normal_transfer,
                                 savemodelpath=lstm_model_save_path,
                                 seq_length=args.seq_len,
                                 num_epochs=5,
                                 n_hidden_units=args.hiddenunits,
                                 n_layers=args.hiddenlayers,
                                 embeddings_model='bert',
                                 train_mode=True,
                                 instance_information_file=transfer_instance_information,
                                 transfer_learning=True)

    ad_normal_transfer.start_training()
    lower, upper = calculate_normal_loss( normal_lstm_model=ad_normal_transfer,
                                          results_dir=results_dir_experiment_transfer,
                                          values_type='normal_loss_values')
    ad_anomaly = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                                  loadvectors=embeddings_anomalies,
                                  savemodelpath=lstm_model_save_path,
                                  seq_length=args.seq_len,
                                  num_epochs=args.epochs,
                                  n_hidden_units=args.hiddenunits,
                                  n_layers=args.hiddenlayers,
                                  embeddings_model='bert',
                                  instance_information_file=instance_information_file_anomalies_transfer)
    calculate_anomaly_loss(anomaly_lstm_model=ad_anomaly, results_dir=results_dir_experiment_transfer, lo=lower, up=upper)
    calculate_precision_and_plot(results_dir_experiment_transfer, anomalyfile_transfer)
