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
from logparser.utah_log_parser import shuffle_log_sequences, parse_and_sort

predicted_labels_of_file_containing_anomalies = "predicted_labels_of_file_containing_anomalies"
bert_model_finetune = "wordembeddings/finetuning-models/utah_137k_plus_change_word_anomalies"


def calculate_precision_and_plot(this_results_dir_experiment, log_file_containing_anomalies):
    precision = calc_precision_utah(log_file_containing_anomalies=log_file_containing_anomalies,
                                    outliers_file=cwd + this_results_dir_experiment + 'outliers_values')
    distribution_plots(this_results_dir_experiment, args.epochs, args.seq_len, 768, precision)
    subprocess.call(['tar', 'cvf', cwd + this_results_dir_experiment + "{}_epochs_{}_seq_len_{}"
                                    .format('bert', args.epochs, args.seq_len) + '.tar',
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
    cut_off = std * 3  # TODO: is this ok?
    lower, upper = mean - cut_off, mean + cut_off
    os.makedirs(results_dir, exist_ok=True)
    normal_values_file = open(cwd + results_dir + values_type, 'w+')
    for val in normal_loss_values:
        normal_values_file.write(str(val) + "\n")
    normal_values_file.close()


def calculate_anomaly_loss(anomaly_lstm_model, results_dir):
    anomaly_loss_values = anomaly_lstm_model.loss_values(normal=False)

    anomaly_values_file = open(cwd + results_dir + 'anomaly_loss_values', 'w+')
    for val in anomaly_loss_values:
        anomaly_values_file.write(str(val) + "\n")
    anomaly_values_file.close()


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
args = parser.parse_args()

# if we do anomaly only, we implicitly also want reduced
if args.anomaly_only: args.reduced = True

option = args.option
results_dir_experiment = "{}_epochs_{}_seq_len_{}/" \
    .format(settings.settings[option]["resultsdir"] + 'bert', args.epochs, args.seq_len)
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
templates_merged_glove = '../' + parseddir + combinedinputfile + '_merged_templates'

corpus_normal_inputfile = cwd + parseddir + normalinputfile + '_corpus'
corpus_anomaly_inputfile = cwd + parseddir + anomalyinputfile + '_corpus'
corpus_combined_file = cwd + parseddir + combinedinputfile + '_corpus'

embeddingsfile_for_transformer = cwd + embeddingsdir + combinedinputfile + '_vectors.txt'
embeddingsfile_for_glove = '../' + embeddingsdir + combinedinputfile + '_vectors'

embeddings_normal = cwd + embeddingspickledir + normalinputfile + '.pickle'
embeddings_anomalies = cwd + embeddingspickledir + anomalyinputfile + '.pickle'
vae_model_save_path = cwd + 'loganaliser/saved_models/' + normalinputfile + '_vae.pth'
lstm_model_save_path = cwd + 'loganaliser/saved_models/' + normalinputfile + '_lstm.pth'

corpus_anomaly_outputfile = cwd + parseddir + anomalyinputfile + 'anomalies_injected_corpus'
anomalies_injected_indeces = cwd + anomaly_indeces_dir + anomalyinputfile + 'anomaly_indeces.txt'




if not args.reduced:
    # parse_and_sort(logfile_path=inputdir + anomalyinputfile,
    #                output_path=parseddir + corpus_anomaly_outputfile,
    #                instance_information_path=instance_information_file_anomalies)
    # start Drain parser
    drain.execute(directory=inputdir, file=combinedinputfile, output=parseddir, logtype=logtype)
    drain.execute(directory=inputdir, file=anomalyinputfile, output=parseddir, logtype=logtype)
    drain.execute(directory=inputdir, file=normalinputfile, output=parseddir, logtype=logtype)

    shuffle_log_sequences(corpus_input_file_path=corpus_anomaly_inputfile,
                          output_file_path=corpus_anomaly_outputfile,
                          instance_information_path=instance_information_file_anomalies,
                          anomaly_indices_output_path=anomalies_injected_indeces)

    templates_anomaly = list(set(open(corpus_anomaly_outputfile, 'r').readlines()))
    transform_glove.merge_templates(templates_normal, templates_anomaly, templates_added,
                                    merged_template_path=templates_merged)

    # do_finetuning(corpus=templates_merged,
    #               output_dir=bert_model_finetune)

    bert_vectors, _, _, _ = transform_bert.get_bert_vectors(templates_merged,
                                                            bert_model=bert_model_finetune)

    # transform output of bert into numpy word embedding vectors
    transform_bert.transform(sentence_embeddings=bert_vectors,
                             logfile=corpus_normal_inputfile,
                             templatefile=templates_merged,
                             outputfile=embeddings_normal)

    transform_bert.transform(sentence_embeddings=bert_vectors,
                             logfile=corpus_anomaly_outputfile,
                             templatefile=templates_merged,
                             outputfile=embeddings_anomalies)
# -------------------------------------------------------------------------------------------------------
# --------------------------------NORMAL LEARNING--------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
if not args.transferlearning:
    if not args.anomaly_only:
        # learning on normal data
        ad_normal = learning(arg=args, embeddings_path=embeddings_normal, epochs=args.epochs,
                             instance_information_file=instance_information_file_normal)
        # calculate loss on normal data and log
        calculate_normal_loss(normal_lstm_model=ad_normal,
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
    calculate_anomaly_loss(anomaly_lstm_model=ad_anomaly, results_dir=results_dir_experiment)
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

    # NORMAL TRAINING with dataset 1
    ad_normal = learning(args, embeddings_normal, args.epochs, instance_information_file_normal)
    # FEW SHOT TRAINING with dataset 2
    ad_normal_transfer = learning(arg=args, embeddings_path=embeddings_normal_transfer, epochs=0,
                                  instance_information_file=instance_information_file_normal_transfer)
    calculate_normal_loss(normal_lstm_model=ad_normal,
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
    calculate_anomaly_loss(anomaly_lstm_model=ad_anomaly, results_dir=results_dir_experiment_transfer)
    calculate_precision_and_plot(results_dir_experiment_transfer, anomalyfile_transfer)
