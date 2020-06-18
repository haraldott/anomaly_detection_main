import numpy as np
import pickle
import pandas as pd


def _load_word_vectors(templates,
                       vectorsfile='../data/openstack/utah/embeddings/openstack_18k_plus_52k_vectors.txt'):
    # transform glove output
    vectors_file = open(vectorsfile, 'r')
    vectors_lookup = vectors_file.readlines()

    words_vectors = {}
    for i, line in enumerate(vectors_lookup):
        line_split = line.split(' ')
        vector = [float(line_split[i]) for i in range(1, len(line_split))]
        words_vectors[(line_split[0])] = np.array(vector)

    sentence_embeddings = {}
    for line in templates:
        line_stripped = line.rstrip()
        split_line = line_stripped.split(' ')

        sent_vec = [words_vectors.get(word) for word in split_line]
        sentence_embeddings.update({line: sent_vec}) # TODO doch np array ?!

    return sentence_embeddings


def dump_word_vectors(templates,
                      vectorsfile='../data/openstack/utah/embeddings/openstack_18k_plus_52k_vectors.txt',
                      words_vectors_file='vectors_for_cosine_distance/sasho_glove_vectors_for_cosine.pickle'):
    sentence_embeddings = _load_word_vectors(templates, vectorsfile=vectorsfile)
    pickle.dump(sentence_embeddings, open(words_vectors_file, 'wb'))


def transform(templates,
              logfile='../data/openstack/utah/parsed/openstack_18k_plus_52k_corpus',
              vectorsfile='../data/openstack/utah/embeddings/openstack_18k_plus_52k_vectors.txt',
              outputfile='../data/openstack/utah/padded_embeddings_pickle/openstack_18k_anomalies_embeddings.pickle'):
    sentence_embeddings = _load_word_vectors(templates, vectorsfile=vectorsfile)

    # padding
    if type(logfile) == str:
        with open(logfile, 'r') as f:
            corpus = f.readlines()
    elif type(logfile) == list:
        corpus = logfile

    corpus_as_embeddings = [sentence_embeddings.get(sent) for sent in corpus]

    embeddings_dim = len(list(sentence_embeddings.values())[0][0])  # dimension of each of the word embeddings vectors
    sentence_lens = [len(sentence.split(' ')) if len(sentence) > 0 else print(i) for i, sentence in
                     enumerate(sentence_embeddings.keys())]  # how many words a log line consists of, without padding
    longest_sent = max(sentence_lens)
    total_batch_size = len(corpus_as_embeddings)
    pad_vector = np.zeros(embeddings_dim)
    padded_emb = np.ones((total_batch_size, longest_sent, embeddings_dim)) * pad_vector

    for i, sequence in enumerate(corpus_as_embeddings):
        sent_len = len(sequence)
        padded_emb[i, :sent_len] = sequence[:sent_len]

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
        if type(file) == str:
            try:
                this_file = open(file, "r")
                for line in this_file.readlines():
                    log_lines.append(line)
            except FileNotFoundError:
                print("Could not open file {}".format(file))
        elif type(file) == list:
            for line in file:
                log_lines.append(line)
    log_lines = set(log_lines)
    if merged_template_path:
        merged_template_file = open(merged_template_path, 'w+')
        for t in log_lines:
            merged_template_file.write(t)
    return log_lines


def extract_event_templates():
    p = pd.read_csv("../data/openstack/utah/parsed/openstack_52k_normal_structured.csv")
    et = p["EventTemplate"]

    parsed_file = open("52k_normal_event_templates.txt", "w+")
    for sentence in et:
        parsed_file.write(sentence + "\n")

