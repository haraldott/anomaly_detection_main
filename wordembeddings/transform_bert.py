import argparse
import os
import pickle

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel


def transform(sentence_embeddings,
              logfile='../data/openstack/utah/parsed/openstack_18k_anomalies_corpus',
              outputfile='../data/openstack/utah/padded_embeddings_pickle/openstack_18k_anomalies_embeddings.pickle',
              templatefile='../data/openstack/utah/parsed/openstack_18k_plus_52k_merged_templates'):
    """

    :param templatefile:
    :param sentence_embeddings:
    :param logfile:
    :param outputfile:
    :return:
    """
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
    p_max, p_min = sentences_as_vectors.max(), sentences_as_vectors.min()
    sentences_as_vectors = (sentences_as_vectors - p_min) / (p_max - p_min)

    pickle.dump(sentences_as_vectors, open(outputfile, 'wb'))


def get_bert_vectors(templates_location='../data/openstack/utah/parsed/openstack_18k_plus_52k_merged_templates'):
    cwd = os.getcwd() + "/"
    parser = argparse.ArgumentParser()
    parser.add_argument('-parsedinputfile', type=str, default='data/openstack/utah/parsed/')

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

    # token_embeddings_stacked = [torch.squeeze(torch.stack(layers[0]), dim=1) for layers in encoded_layers]
    #
    # token_embeddings = [emb.permute(1, 0, 2) for emb in token_embeddings_stacked]

    sentence_embeddings = []
    for t in encoded_layers:
        sentence_embeddings.append(torch.mean(t[0][11][0], dim=0))

    return sentence_embeddings
