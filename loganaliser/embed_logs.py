from typing import List

import pandas as pd
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings


def load_dataset(log_file: str):
    struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
    return struct_log["EventTemplate"]


def embed_dataset() -> List:
    # init standard GloVe embedding
    glove_embedding = WordEmbeddings('glove')
    flair_embedding_forward = FlairEmbeddings('news-forward')

    # create a StackedEmbedding object that combines glove and forward/backward flair embeddings
    stacked_embeddings = StackedEmbeddings([
        glove_embedding,
        flair_embedding_forward,
    ])
    sentence_dataset = load_dataset(
        '/Users/haraldott/Development/thesis/anomaly_detection_main/data/openstack/parsed_drain_st_0.2_depth_2/first_1000.csv')

    embedded_sentences = []
    count = 0.0
    for s in sentence_dataset:
        sentence = Sentence(s)
        glove_embedding.embed(sentence)
        embedded_sentences.append(sentence)
        if count % 50 == 0 or count == len(sentence_dataset):
            print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(sentence_dataset)))
        count += 1
    words = []
    for sentence in embedded_sentences:
        for word in sentence:
            words.append(word.embedding)  #  TODO: is this correct? return all
    return words
