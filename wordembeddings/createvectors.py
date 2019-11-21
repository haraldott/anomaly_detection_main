from glove import Corpus, Glove
import numpy as np
import argparse
import pandas as pd
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-file', type=str,
                    default='../data/openstack/utah/parsed/openstack_18k_anomalies_structured.csv')
parser.add_argument('-saveglove', type=str,
                    default='../data/openstack/utah/embeddings/glove_18k_anomalies_no_norm_and_padding.model')
parser.add_argument('-savevectors', type=str,
                    default='../data/openstack/utah/embeddings/glove_18k_anomalies_no_norm_and_padding.pickle')
args = parser.parse_args()
log_file_path = args.file
glove_save_path = args.saveglove
vectors_save_path = args.savevectors

p = pd.read_csv(log_file_path)
corpus = Corpus()
new_lines = [line.split(' ') for line in p["EventTemplate"]]
corpus.fit(new_lines)
glove = Glove(no_components=60, learning_rate=0.01)
glove.fit(corpus.matrix, epochs=80, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

# transform words into embeddings
new_lines_as_vectors = []
for sublist in new_lines:
    new_sublist = []
    for word in sublist:
        new_sublist.append(glove.word_vectors[glove.dictionary[word]])
    new_lines_as_vectors.append(np.array(new_sublist))
embeddings = np.array(new_lines_as_vectors)



#  save everything
pickle.dump(embeddings, open(vectors_save_path, "wb"))
pickle.dump(glove, open(glove_save_path, "wb"))
