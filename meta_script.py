import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument('-inputdir', type=str, default='data/openstack/utah/raw/')
parser.add_argument('-inputfile', type=str, default='openstack_52k_normal')
parser.add_argument('-parseddir', type=str, default='data/openstack/utah/parsed/')
parser.add_argument('-embeddingspickledir', type=str, default='data/openstack/utah/')
parser.add_argument('-embeddingsdir', type=str, default='data/openstack/utah/embeddings/')
parser.add_argument('-logtype', default='OpenStack', type=str)
args = parser.parse_args()

# start Drain parser
subprocess.call(['python', 'logparser/Drain/Drain_demo.py',
                 '-dir', args.inputdir,
                 '-file', args.inputfile,
                 '-output', args.parseddir])

inputfile_full_path = '../' + args.parseddir + args.inputfile + '_corpus'
embddingsfile_full_path = '../' + args.embeddingsdir + args.inputfile + '_vectors'
padded_embeddings_file_full_path = '../' + args.embeddingspickledir + args.inputfile + '.pickle'

# start glove-c
subprocess.call(['glove-c/word_embeddings.sh',
                 '-c', inputfile_full_path,
                 '-s', embddingsfile_full_path])

subprocess.call(['wordembeddings/transform_glove.py',
                 '-logfile', inputfile_full_path,
                 '-vectorsfile', embddingsfile_full_path,
                 '-outputfile', padded_embeddings_file_full_path])