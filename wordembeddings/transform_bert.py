import pickle

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.manifold import TSNE

from wordembeddings.visualisation import tsne_plot_2d


# ref: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

def transform(sentence_embeddings,
              logfile='../data/openstack/utah/parsed/openstack_18k_anomalies_corpus',
              outputfile='../data/openstack/utah/padded_embeddings_pickle/openstack_18k_anomalies_embeddings.pickle',
              templatefile='../data/openstack/utah/parsed/openstack_18k_plus_52k_merged_templates'):
    file = open(logfile)
    logfilelines = file.readlines()

    file = open(templatefile)
    templatefilelines = file.readlines()

    sentences_as_vectors = []
    for sentence in logfilelines:
        idx = templatefilelines.index(sentence)
        if idx is None:
            raise ValueError("{} not found in template file".format(sentence))
        sentences_as_vectors.append(sentence_embeddings[idx])
    sentences_as_vectors = torch.stack(sentences_as_vectors)

    # normalise between 0 and 1
    # TODO: normalisation for bert yes or no?
    # p_max, p_min = sentences_as_vectors.max(), sentences_as_vectors.min()
    # sentences_as_vectors = (sentences_as_vectors - p_min) / (p_max - p_min)

    pickle.dump(sentences_as_vectors, open(outputfile, 'wb'))


def get_bert_vectors(templates_location='../data/openstack/utah/parsed/openstack_18k_plus_52k_merged_templates'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    lines = open(templates_location, 'r')

    marked_sentences = []
    for line in lines:
        marked_sentences.append("[CLS] " + line + " [SEP]")

    # Split the sentences into tokens.
    tokenized_text = [tokenizer.tokenize(sentence) for sentence in marked_sentences]

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_text]

    # Mark each of the 22 tokens as belonging to sentence "1".
    # TODO: use 0 and 1 to distinguish between sentences (2.3 in ref)
    segments_ids = [[1] * len(t) for t in tokenized_text]

    # Convert inputs to PyTorch tensors
    tokens_tensors = [torch.tensor([idx_tokens]) for idx_tokens in indexed_tokens]
    seg_tensors = [torch.tensor([seg_ids]) for seg_ids in segments_ids]

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    with torch.no_grad():
        encoded_layers = [model(t, s) for t, s in zip(tokens_tensors, seg_tensors)]

    token_embeddings_stacked = [torch.squeeze(torch.stack(layers[0]), dim=1) for layers in encoded_layers]

    token_embeddings = [emb.permute(1, 0, 2) for emb in token_embeddings_stacked]

    token_vecs_cat = __concatenate_layers(token_embeddings)
    token_vecs_sum = __sum_layers(token_embeddings)

    sentence_embeddings = []
    for t in encoded_layers:
        sentence_embeddings.append(torch.mean(t[0][10][0], dim=0))

    return sentence_embeddings, token_vecs_cat, token_vecs_sum, tokenized_text


def __concatenate_layers(token_embeddings):
    token_vectors_cat = []
    for sentence in token_embeddings:
        for token in sentence:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            token_vectors_cat.append(cat_vec.numpy())
    return np.asarray(token_vectors_cat)


def __sum_layers(token_embeddings):
    token_vectors_sum = []
    for sentence in token_embeddings:
        for token in sentence:
            cat_vec = torch.sum(token[-4:], dim=0)
            token_vectors_sum.append(cat_vec.numpy())
    return np.asarray(token_vectors_sum)


# _, token_vecs_cat, token_vecs_sum, tokenized_text = get_bert_vectors()
# plot_bert(token_vecs_cat, tokenized_text)
def plot_bert(token_vecs, tokenized_text):
    flat_tokenized_text = []
    for sentence in tokenized_text:
        for word in sentence:
            flat_tokenized_text.append(word)

    tsne_ak_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_ak_2d = tsne_ak_2d.fit_transform(token_vecs)
    tsne_plot_2d('bert.png', 'Word embeddings', embeddings_ak_2d, flat_tokenized_text, a=0.9)