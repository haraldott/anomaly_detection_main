from glove import Corpus, Glove
import numpy as np
import argparse
import pandas as pd
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-file', type=str,
                    default='../data/openstack/parsed_drain_st_0.2_depth_2/openstack_18k_anomalies_structured.csv')
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
glove = Glove(no_components=80, learning_rate=0.01)
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


# padding
embeddings_dim = embeddings[0][0].shape[0]  # dimension of each of the word embeddings vectors
sentence_lens = [len(sentence) for sentence in embeddings]  # how many words a log line consists of, without padding
longest_sent = max(sentence_lens)
total_batch_size = len(embeddings)
pad_vector = np.zeros(embeddings_dim)
padded_emb = np.ones((total_batch_size, longest_sent, embeddings_dim)) * pad_vector

for i, x_len in enumerate(sentence_lens):
    sequence = embeddings[i]
    padded_emb[i, 0:x_len] = sequence[:x_len]

# normalise between 0 and 1
p_max, p_min = padded_emb.max(), padded_emb.min()
padded_embeddings = (padded_emb - p_min)/(p_max - p_min)

#  save everything
pickle.dump(padded_embeddings, open(vectors_save_path, "wb"))
pickle.dump(glove, open(glove_save_path, "wb"))
