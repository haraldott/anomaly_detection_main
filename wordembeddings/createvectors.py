from glove import Corpus, Glove
import numpy as np
import argparse
import pandas as pd
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-file', type=str, default='../data/openstack/parsed_drain_st_0.2_depth_2/openstack_18k_anomalies_structured.csv')
parser.add_argument('-saveglove', type=str, default='../data/openstack/utah/embeddings/glove.model')
parser.add_argument('-savevectors', type=str, default='../data/openstack/utah/embeddings/vectors.pickle')
args = parser.parse_args()
log_file_path = args.file
glove_save_path = args.saveglove
vectors_save_path = args.savevectors

p = pd.read_csv(log_file_path)
corpus = Corpus()
new_lines = [line.split(' ') for line in p["EventTemplate"]]
corpus.fit(new_lines)
glove = Glove(no_components=30, learning_rate=0.2)
glove.fit(corpus.matrix, epochs=100, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

# transform words into embeddings
new_lines_as_vectors = []
for sublist in new_lines:
    new_sublist = []
    for word in sublist:
        new_sublist.append(glove.word_vectors[glove.dictionary[word]])
    new_lines_as_vectors.append(np.array(new_sublist))
new_lines_as_vectors_np = np.array(new_lines_as_vectors)

#  save everything
#vectors_file = open(vectors_save_path, "w")
pickle.dump(new_lines_as_vectors_np, open(vectors_save_path, "wb"))
#glove_file = open(glove_save_path, "w")
pickle.dump(glove, open(glove_save_path, "wb"))

