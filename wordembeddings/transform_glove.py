import numpy as np
import pickle
import pandas as pd


def transform(logfile='../data/openstack/utah/parsed/openstack_18k_anomalies_corpus',
              vectorsfile='../data/openstack/utah/embeddings/openstack_18k_anomalies_embeddings.txt',
              outputfile='../data/openstack/utah/padded_embeddings_pickle/openstack_18k_anomalies_embeddings.pickle'):
    """

    :param logfile:
    :param vectorsfile:
    :param outputfile:
    :return:
    """
    file = open(logfile)
    lines = file.read().splitlines()
    log_lines_per_word = [line.split(' ') for line in lines]

    # transform glove output
    vectors_file = open(vectorsfile, 'r')
    vectors_lookup = vectors_file.readlines()

    words = {}
    vectors = []
    for i, line in enumerate(vectors_lookup):
        line_split = line.split(' ')
        words[(line_split[0])] = i
        vector = [line_split[i] for i in range(1, len(line_split))]
        vector_as_array = np.array(vector)
        vectors.append(vector_as_array)
    vectors = np.array(vectors)

    new_lines_as_vectors = []
    for sublist in log_lines_per_word:
        new_sublist = []
        for word in sublist:
            word_index = words.get(word)
            if word_index is None:
                raise ValueError(word + " does not have an index, it returned None")
            new_sublist.append(vectors[word_index])
        new_lines_as_vectors.append(np.array(new_sublist))
    embeddings = np.array(new_lines_as_vectors)

    # padding
    embeddings_dim = vectors.shape[1]  # dimension of each of the word embeddings vectors
    sentence_lens = [len(sentence) for sentence in
                     log_lines_per_word]  # how many words a log line consists of, without padding
    longest_sent = max(sentence_lens)
    total_batch_size = len(embeddings)
    pad_vector = np.zeros(embeddings_dim)
    padded_emb = np.ones((total_batch_size, longest_sent, embeddings_dim)) * pad_vector

    for i, x_len in enumerate(sentence_lens):
        sequence = embeddings[i]
        padded_emb[i, 0:x_len] = sequence[:x_len]

    # normalise between 0 and 1
    p_max, p_min = padded_emb.max(), padded_emb.min()
    padded_embeddings = (padded_emb - p_min) / (p_max - p_min)

    pickle.dump(padded_embeddings, open(outputfile, 'wb'))


def merge_templates(*template_files, merged_template_path):
    """

    :param merged_template_path: 
    :param template_files:
    :return:
    """
    log_lines = []
    for file in template_files:
        try:
            this_file = open(file, "r")
            for line in this_file.readlines():
                log_lines.append(line)
        except FileNotFoundError:
            print("Could not open file {}".format(file))
    log_lines = set(log_lines)
    merged_template_file = open(merged_template_path, 'w+')
    for t in log_lines:
        merged_template_file.write(t)


def extract_event_templates():
    p = pd.read_csv("../data/openstack/utah/parsed/openstack_52k_normal_structured.csv")
    et = p["EventTemplate"]

    parsed_file = open("52k_normal_event_templates.txt", "w+")
    for sentence in et:
        parsed_file.write(sentence + "\n")
