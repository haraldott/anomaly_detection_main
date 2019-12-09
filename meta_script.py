import argparse
import subprocess

import numpy as np

import logparser.Drain.Drain_demo as drain
import wordembeddings.transform_glove as transform_glove
from loganaliser.main import AnomalyDetection
from loganaliser.vanilla_autoencoder import VanillaAutoEncoder

parser = argparse.ArgumentParser()
parser.add_argument('-combinedinputfile', type=str, default='openstack_18k_plus_52k')
parser.add_argument('-anomalyinputfile', type=str, default='openstack_18k_anomalies')
parser.add_argument('-normalinputfile', type=str, default='openstack_52k_normal')
parser.add_argument('-inputdir', type=str, default='data/openstack/utah/raw/')
parser.add_argument('-parseddir', type=str, default='data/openstack/utah/parsed/')
parser.add_argument('-embeddingspickledir', type=str, default='data/openstack/utah/padded_embeddings_pickle/')
parser.add_argument('-embeddingsdir', type=str, default='data/openstack/utah/embeddings/')
parser.add_argument('-logtype', default='OpenStack', type=str)
parser.add_argument('-seq_len', type=int, default=7)
args = parser.parse_args()

templates_inputfile_full_path = '../' + args.parseddir + args.combinedinputfile + '_templates'

corpus_normal_inputfile_full_path = '../' + args.parseddir + args.normalinputfile + '_corpus'
corpus_anomaly_inputfile_full_path = '../' + args.parseddir + args.anomalyinputfile + '_corpus'
corpus_combined_file_full_path = '../' + args.parseddir + args.combinedinputfile + '_corpus'

embeddingsfile_full_path_for_transformer = '../' + args.embeddingsdir + args.inputfile + '_vectors.txt'
embeddingsfile_full_path_for_glove = '../' + args.embeddingsdir + args.inputfile + '_vectors'

padded_embeddings_normal_file_full_path = '../' + args.embeddingspickledir + args.normalinputfile + '.pickle'
padded_embeddings_anomalies_file_full_path = '../' + args.embeddingspickledir + args.anomalyinputfile + '.pickle'
padded_embeddings_combined_file_full_path = '../' + args.embeddingspickledir + args.combinedinputfile + '.pickle'
vae_model_save_path = 'saved_models/' + args.inputfile + '_vae.pth'
lstm_model_save_path = 'saved_models/' + args.inputfile + '_lstm.pth'

# start Drain parser
drain.execute(dir=args.inputdir, file=args.combinedinputfile, output=args.parseddir)
drain.execute(dir=args.inputdir, file=args.anomalyinputfile, output=args.parseddir)
drain.execute(dir=args.inputdir, file=args.normalinputfile, output=args.parseddir)

# start glove-c
subprocess.call(['glove-c/word_embeddings.sh',
                 '-c', templates_inputfile_full_path,
                 '-s', embeddingsfile_full_path_for_glove])

# transform output of glove into numpy word embedding vectors
transform_glove.transform(logfile=corpus_normal_inputfile_full_path,
                          vectorsfile=embeddingsfile_full_path_for_transformer,
                          outputfile=padded_embeddings_normal_file_full_path)

transform_glove.transform(logfile=corpus_anomaly_inputfile_full_path,
                          vectorsfile=embeddingsfile_full_path_for_transformer,
                          outputfile=padded_embeddings_anomalies_file_full_path)

transform_glove.transform(logfile=corpus_combined_file_full_path,
                          vectorsfile=embeddingsfile_full_path_for_transformer,
                          outputfile=padded_embeddings_combined_file_full_path)

vae = VanillaAutoEncoder(load_vectors=padded_embeddings_combined_file_full_path, model_save_path=vae_model_save_path)
vae.start()

ad_normal = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                             loadvectors=padded_embeddings_normal_file_full_path,
                             savemodelpath=lstm_model_save_path,
                             seq_length=args.seq_len)
ad_normal.start_training()
normal_loss_values = ad_normal.loss_values(normal=True)

mean = np.mean(normal_loss_values)
std = np.std(normal_loss_values)
cut_off = std * 3  # TODO: is this ok?
lower, upper = mean - cut_off, mean + cut_off
normal_values_file = open('normal_loss_values', 'w+')
for val in normal_loss_values:
    normal_values_file.write(str(val) + "\n")
normal_values_file.close()

ad_anomaly = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                              loadvectors=padded_embeddings_anomalies_file_full_path,
                              savemodelpath=lstm_model_save_path,
                              seq_length=args.seq_len)
anomaly_loss_values = ad_anomaly.loss_values(normal=False)

anomaly_values_file = open('anomaly_loss_values', 'w+')
for val in anomaly_loss_values:
    anomaly_values_file.write(str(val) + "\n")
anomaly_values_file.close()

outliers = []
for i, x in enumerate(anomaly_loss_values):
    if x < lower or x > upper:
        outliers.append((i + args.seq_len, x))

outliers_values_file = open('outliers_values', 'w+')
for val in outliers:
    outliers_values_file.write(str(val) + "\n")
outliers_values_file.close()
