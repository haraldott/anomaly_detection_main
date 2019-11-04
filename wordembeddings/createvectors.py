from glove import Corpus, Glove
import numpy as np
import argparse
import pandas as pd
import torch


def create_word_vectors():
    global glove
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type=str,
                        default='/Users/haraldott/Development/thesis/anomaly_detection_main/data/openstack/parsed_drain_st_0.2_depth_2/openstack_18k_anomalies_structured.csv')
    parser.add_argument('-save', type=str, default='outfile.npy')
    args = parser.parse_args()
    file_path = args.file
    save_path = args.save
    p = pd.read_csv(file_path)
    corpus = Corpus()
    new_lines = [line.split(' ') for line in p["EventTemplate"]]
    corpus.fit(new_lines)
    glove = Glove(no_components=200, learning_rate=0.5)
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    new_lines_as_vectors = []
    for sublist in new_lines:
        new_sublist = []
        for word in sublist:
            new_sublist.append(glove.word_vectors[glove.dictionary[word]])
        new_lines_as_vectors.append(np.array(new_sublist))
    new_lines_as_vectors_np = np.array(new_lines_as_vectors)
    return new_lines_as_vectors_np, glove
