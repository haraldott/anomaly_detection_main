import argparse
import os
import subprocess

import numpy as np

import logparser.Drain.Drain_demo as drain
import wordembeddings.transform_glove as transform_glove
from loganaliser.main import AnomalyDetection
from loganaliser.vanilla_autoencoder import VanillaAutoEncoder
from tools import distribution_plots as distribution_plots, calc_precision_utah as calc_precision_utah

cwd = os.getcwd() + "/"
parser = argparse.ArgumentParser()
parser.add_argument('-combinedinputfile', type=str, default='openstack_18k_plus_52k')
parser.add_argument('-anomalyinputfile', type=str, default='openstack_18k_anomalies')
parser.add_argument('-normalinputfile', type=str, default='openstack_52k_normal')
parser.add_argument('-inputdir', type=str, default='data/openstack/utah/raw/')
parser.add_argument('-parseddir', type=str, default='data/openstack/utah/parsed/')
parser.add_argument('-resultsdir', type=str, default='data/openstack/utah/results/glove/')
parser.add_argument('-embeddingspickledir', type=str, default='data/openstack/utah/padded_embeddings_pickle/')
parser.add_argument('-embeddingsdir', type=str, default='data/openstack/utah/embeddings/')
parser.add_argument('-logtype', default='OpenStack', type=str)
parser.add_argument('-seq_len', type=int, default=7)
parser.add_argument('-full', type=str, default="True")
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-hiddenunits', type=int, default=250)
parser.add_argument('-embeddingsize', type=int, default=100)
args = parser.parse_args()

results_dir = "{}_epochs_{}_hiddenunits_{}_embeddingsize_{}/"\
    .format(args.resultsdir, args.epochs, args.hiddenunits, args.embeddingsize)
templates_normal = cwd + args.parseddir + args.normalinputfile + '_templates'
templates_anomaly = cwd + args.parseddir + args.anomalyinputfile + '_templates'
templates_added = cwd + args.parseddir + args.combinedinputfile + '_templates'
templates_merged = cwd + args.parseddir + args.combinedinputfile + '_merged_templates'
templates_merged_glove = '../' + args.parseddir + args.combinedinputfile + '_merged_templates'

corpus_normal_inputfile = cwd + args.parseddir + args.normalinputfile + '_corpus'
corpus_anomaly_inputfile = cwd + args.parseddir + args.anomalyinputfile + '_corpus'
corpus_combined_file = cwd + args.parseddir + args.combinedinputfile + '_corpus'

embeddingsfile_for_transformer = cwd + args.embeddingsdir + args.combinedinputfile + '_vectors.txt'
embeddingsfile_for_glove = '../' + args.embeddingsdir + args.combinedinputfile + '_vectors'

padded_embeddings_normal = cwd + args.embeddingspickledir + args.normalinputfile + '.pickle'
padded_embeddings_anomalies = cwd + args.embeddingspickledir + args.anomalyinputfile + '.pickle'
padded_embeddings_combined = cwd + args.embeddingspickledir + args.combinedinputfile + '.pickle'
vae_model_save_path = cwd + 'loganaliser/saved_models/' + args.normalinputfile + '_vae.pth'
lstm_model_save_path = cwd + 'loganaliser/saved_models/' + args.normalinputfile

if args.full == "True":
    # start Drain parser
    drain.execute(dir=args.inputdir, file=args.combinedinputfile, output=args.parseddir)
    drain.execute(dir=args.inputdir, file=args.anomalyinputfile, output=args.parseddir)
    drain.execute(dir=args.inputdir, file=args.normalinputfile, output=args.parseddir)

    transform_glove.merge_templates(templates_normal, templates_anomaly, templates_added,
                                    merged_template_path=templates_merged)

    # start glove-c
    subprocess.call(['glove-c/word_embeddings.sh',
                     '-c', templates_merged_glove,
                     '-s', embeddingsfile_for_glove,
                     '-v', args.embeddingsize])

    # transform output of glove into numpy word embedding vectors
    transform_glove.transform(logfile=corpus_normal_inputfile,
                              vectorsfile=embeddingsfile_for_transformer,
                              outputfile=padded_embeddings_normal)

    transform_glove.transform(logfile=corpus_anomaly_inputfile,
                              vectorsfile=embeddingsfile_for_transformer,
                              outputfile=padded_embeddings_anomalies)

    transform_glove.transform(logfile=corpus_combined_file,
                              vectorsfile=embeddingsfile_for_transformer,
                              outputfile=padded_embeddings_combined)

    vae = VanillaAutoEncoder(load_vectors=padded_embeddings_combined, model_save_path=vae_model_save_path)
    vae.start()

ad_normal = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                             loadvectors=padded_embeddings_normal,
                             savemodelpath=lstm_model_save_path,
                             seq_length=args.seq_len,
                             num_epochs=args.epochs,
                             n_hidden_units=args.hiddenunits)
ad_normal.start_training()

# run normal values once through LSTM to obtain loss values
normal_loss_values = ad_normal.loss_values(normal=True)

mean = np.mean(normal_loss_values)
std = np.std(normal_loss_values)
cut_off = std * 3  # TODO: is this ok?
lower, upper = mean - cut_off, mean + cut_off
normal_values_file = open(cwd + results_dir + 'normal_loss_values', 'w+')
for val in normal_loss_values:
    normal_values_file.write(str(val) + "\n")
normal_values_file.close()

ad_anomaly = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                              loadvectors=padded_embeddings_anomalies,
                              savemodelpath=lstm_model_save_path,
                              seq_length=args.seq_len,
                              num_epochs=args.epochs,
                              n_hidden_units=args.hiddenunits)
anomaly_loss_values = ad_anomaly.loss_values(normal=False)

anomaly_values_file = open(cwd + results_dir + 'anomaly_loss_values', 'w+')
for val in anomaly_loss_values:
    anomaly_values_file.write(str(val) + "\n")
anomaly_values_file.close()

outliers = []
for i, x in enumerate(anomaly_loss_values):
    if x < lower or x > upper:
        outliers.append(str(i + args.seq_len) + "," + str(x))

outliers_values_file = open(cwd + results_dir + 'outliers_values', 'w+')
for val in outliers:
    outliers_values_file.write(str(val) + "\n")
outliers_values_file.close()

precision = calc_precision_utah(cwd + results_dir + 'anomaly_loss_values', cwd + results_dir + 'outliers_values')
distribution_plots(results_dir, args.epochs, args.hiddenunits, args.embeddingsize, precision)

subprocess.call(['tar', 'cvf', cwd + results_dir + 'results.tar',
                 '--directory=' + cwd + results_dir,
                 'normal_loss_values',
                 'anomaly_loss_values',
                 'outliers_values',
                 'plot.png'])
