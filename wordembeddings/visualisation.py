from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
from wordembeddings import transform_bert as transform_bert
import os


def tsne_plot_2d(filename, label, emb, words=[], a=1):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, 1))
    x = emb[:, 0]
    y = emb[:, 1]
    plt.scatter(x, y, c=colors, alpha=a, label=label)
    for i, word in enumerate(words):
        plt.annotate(word, alpha=0.3, xy=(x[i], y[i]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom', size=10)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig(filename, format='png', dpi=500, bbox_inches='tight')
    plt.show()


def plot_glove(vectorsfile='../data/openstack/utah/embeddings/openstack_18k_plus_52k_vectors.txt'):
    vectors_file = open(vectorsfile, 'r')
    vectors_lookup = vectors_file.readlines()

    words = []
    vectors = []
    for i, line in enumerate(vectors_lookup):
        line_split = line.split(' ')
        words.append((line_split[0]))
        vector = [float(line_split[i]) for i in range(1, len(line_split))]
        vector_as_array = np.array(vector)
        vectors.append(vector_as_array)
    embeddings = np.array(vectors)

    tsne_ak_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_ak_2d = tsne_ak_2d.fit_transform(embeddings)

    tsne_plot_2d('glove.png', 'Word embeddings', embeddings_ak_2d, words, a=0.9)
    return words, vectors


def write_to_tsv_files_glove():
    tokenized_text, token_vecs = plot_glove()
    with open('vectors_glove.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for word in token_vecs:
            tsv_writer.writerow(word)

    with open('words_glove.tsv', 'wt') as out_file:
        for word in tokenized_text:
            out_file.write(word + "\n")


def write_to_tsv_files_bert():
    _, _, token_vecs_sum, tokenized_text = transform_bert.get_bert_vectors()
    with open('vectors_bert.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for word in token_vecs_sum:
            tsv_writer.writerow(word)

    with open('words_bert.tsv', 'wt') as out_file:
        for sentence in tokenized_text:
            for word in sentence:
                out_file.write(word + "\n")


def write_to_tsv_files_bert_sentences(vectors,
                                      sentences,
                                      tsv_file_vectors='vectors_bert_sentences_before_altering.tsv',
                                      tsv_file_sentences='bert_sentences_before_altering.tsv'):
    # sentences, _, _, _ = transform_bert.get_bert_vectors(
    #     templates_location="/Users/haraldott/Downloads/results/no finetune/bert_epochs_100_seq_len:_7_anomaly_type:insert_words_9/lines_before_altering.txt")
    os.makedirs(os.path.dirname(tsv_file_vectors), exist_ok=True)

    with open(tsv_file_vectors, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for vector in vectors:
            tsv_writer.writerow(vector)

    with open(tsv_file_sentences, 'wt') as out_file:
        for sentence in sentences:
            out_file.write(sentence)
