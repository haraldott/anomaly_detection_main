import argparse
import os
import subprocess
import matplotlib
import pathlib
matplotlib.use('Agg')
import settings

import numpy as np

import logparser.Drain.Drain_demo as drain
import wordembeddings.transform_bert as transform_bert
import wordembeddings.transform_glove as transform_glove
from loganaliser.main import AnomalyDetection
from tools import distribution_plots as distribution_plots, calc_precision_utah as calc_precision_utah

cwd = os.getcwd() + "/"
parser = argparse.ArgumentParser()
parser.add_argument('-option', type=str, default='UtahSorted137')
parser.add_argument('-seq_len', type=int, default=7)
parser.add_argument('-full', type=str, default="True")
parser.add_argument('-epochs', type=int, default=1)
parser.add_argument('-hiddenunits', type=int, default=250)
parser.add_argument('-hiddenlayers', type=int, default=4)
args = parser.parse_args()

option = args.option
results_dir_experiment = "{}_epochs_{}_hiddenunits_{}/".format(settings.settings[option]["resultsdir"]+'bert/', args.epochs, args.hiddenunits)
combinedinputfile = settings.settings[option]["combinedinputfile"]
anomalyinputfile = settings.settings[option]["anomalyinputfile"]
normalinputfile = settings.settings[option]["normalinputfile"]
inputdir = settings.settings[option]["inputdir"]
parseddir = settings.settings[option]["parseddir"]
resultsdir = settings.settings[option]["resultsdir"]
embeddingspickledir = settings.settings[option]["embeddingspickledir"]
embeddingsdir = settings.settings[option]["embeddingsdir"]
logtype = settings.settings[option]["logtype"]
instance_information_file = settings.settings[option]['instance_information_file']

# create all directories, if they don't exist yet
pathlib.Path(resultsdir).mkdir(parents=True, exist_ok=True)
pathlib.Path(results_dir_experiment).mkdir(parents=True, exist_ok=True)
pathlib.Path(inputdir).mkdir(parents=True, exist_ok=True)
pathlib.Path(parseddir).mkdir(parents=True, exist_ok=True)
pathlib.Path(embeddingsdir).mkdir(parents=True, exist_ok=True)

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

if args.full == "True":
    # start Drain parser
    drain.execute(directory=inputdir, file=combinedinputfile, output=parseddir, logtype=logtype)
    drain.execute(directory=inputdir, file=anomalyinputfile, output=parseddir, logtype=logtype)
    drain.execute(directory=inputdir, file=normalinputfile, output=parseddir, logtype=logtype)

    transform_glove.merge_templates(templates_normal, templates_anomaly, templates_added,
                                    merged_template_path=templates_merged)

    bert_vectors, _, _, _ = transform_bert.get_bert_vectors(templates_merged,
                                                            bert_model='wordembeddings/finetuning-models')

    # transform output of bert into numpy word embedding vectors
    transform_bert.transform(sentence_embeddings=bert_vectors,
                             logfile=corpus_normal_inputfile,
                             templatefile=templates_merged,
                             outputfile=embeddings_normal)

    transform_bert.transform(sentence_embeddings=bert_vectors,
                             logfile=corpus_anomaly_inputfile,
                             templatefile=templates_merged,
                             outputfile=embeddings_anomalies)

# start LSTM
ad_normal = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                             loadvectors=embeddings_normal,
                             savemodelpath=lstm_model_save_path,
                             seq_length=args.seq_len,
                             num_epochs=args.epochs,
                             n_hidden_units=args.hiddenunits,
                             n_layers=args.hiddenlayers,
                             embeddings_model='bert',
                             train_mode=True,
                             instance_information_file=instance_information_file)
ad_normal.start_training()

# run normal values once through LSTM to obtain loss values, model will be loaded again in this function call,
# train_mode will be set to False
normal_loss_values = ad_normal.loss_values(normal=True)

mean = np.mean(normal_loss_values)
std = np.std(normal_loss_values)
cut_off = std * 3  # TODO: is this ok?
lower, upper = mean - cut_off, mean + cut_off
normal_values_file = open(cwd + results_dir_experiment + 'normal_loss_values', 'w+')
for val in normal_loss_values:
    normal_values_file.write(str(val) + "\n")
normal_values_file.close()

ad_anomaly = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                              loadvectors=embeddings_anomalies,
                              savemodelpath=lstm_model_save_path,
                              seq_length=args.seq_len,
                              num_epochs=args.epochs,
                              n_hidden_units=args.hiddenunits,
                              n_layers=args.hiddenlayers,
                              embeddings_model='bert')
anomaly_loss_values = ad_anomaly.loss_values(normal=False)

anomaly_values_file = open(cwd + results_dir_experiment + 'anomaly_loss_values', 'w+')
for val in anomaly_loss_values:
    anomaly_values_file.write(str(val) + "\n")
anomaly_values_file.close()

outliers = []
for i, x in enumerate(anomaly_loss_values):
    if x < lower or x > upper:
        outliers.append(str(i + args.seq_len) + "," + str(x))

outliers_values_file = open(cwd + results_dir_experiment + 'outliers_values', 'w+')
for val in outliers:
    outliers_values_file.write(str(val) + "\n")
outliers_values_file.close()

precision = calc_precision_utah(cwd + results_dir_experiment + 'anomaly_loss_values', cwd + results_dir_experiment + 'outliers_values')
distribution_plots(results_dir_experiment, args.epochs, args.hiddenunits, 768, precision)

subprocess.call(['tar', 'cvf', cwd + results_dir_experiment + 'results.tar',
                 '--directory=' + cwd + results_dir_experiment,
                 'normal_loss_values',
                 'anomaly_loss_values',
                 'outliers_values',
                 'plot.png'])
