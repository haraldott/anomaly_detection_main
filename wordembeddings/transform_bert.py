import pickle

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer as pretrained_BertTokenizer
from pytorch_pretrained_bert import BertModel as pretrained_BertModel
from transformers import BertTokenizer as transformers_BertTokenizer
from transformers import BertModel as transformers_BertModel


def get_bert_embeddings(templates_location, model):
    token_vecs_cat, token_vecs_sum, tokenized_text, encoded_layers = _prepare_bert_vectors(
        templates_location, bert_model=model)
    sentence_embeddings = []
    for t in encoded_layers:
        sentence_embeddings.append(torch.mean(t[0][10][0], dim=0))

    return sentence_embeddings


def transform(sentence_embeddings,
              logfile='../data/openstack/utah/parsed/openstack_18k_anomalies_corpus',
              outputfile='../data/openstack/utah/padded_embeddings_pickle/openstack_18k_anomalies_embeddings.pickle',
              templates='../data/openstack/utah/parsed/openstack_18k_plus_52k_merged_templates'):
    file = open(logfile)
    logfilelines = file.readlines()

    if type(templates) == str:
        sentences = open(templates, 'r').readlines()
    elif type(templates) == list:
        sentences = templates
    else:
        print("templates must be either a list of templates or a path str")
        raise

    sentences_as_vectors = []
    for sentence in logfilelines:
        idx = sentences.index(sentence)
        if idx is None:
            raise ValueError("{} not found in template file".format(sentence))
        sentences_as_vectors.append(sentence_embeddings[idx])
    sentences_as_vectors = torch.stack(sentences_as_vectors)

    # normalise between 0 and 1
    # TODO: normalisation for bert yes or no?
    # p_max, p_min = sentences_as_vectors.max(), sentences_as_vectors.min()
    # sentences_as_vectors = (sentences_as_vectors - p_min) / (p_max - p_min)

    pickle.dump(sentences_as_vectors, open(outputfile, 'wb'))

    return sentences_as_vectors


def _prepare_bert_vectors(templates='../data/openstack/sasho/parsed/logs_aggregated_full.csv_templates',
                          bert_model='bert-base-uncased'):
    # checks, if pre-trained model or directory containing pre-trained model exists, is being done here
    tokenizer = pretrained_BertTokenizer.from_pretrained(bert_model)
    model = pretrained_BertModel.from_pretrained(bert_model)

    if type(templates) == str:
        lines = open(templates, 'r')
    elif type(templates) == list:
        lines = templates
    else:
        print("templates must be either a list of templates or a path str")
        raise

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

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    with torch.no_grad():
        encoded_layers = [model(t, s) for t, s in zip(tokens_tensors, seg_tensors)]

    token_embeddings_stacked = [torch.squeeze(torch.stack(layers[0]), dim=1) for layers in encoded_layers]

    token_embeddings = [emb.permute(1, 0, 2) for emb in token_embeddings_stacked]

    token_vectors_cat = []
    token_vectors_sum = []

    for sentence in token_embeddings:
        for token in sentence:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vectors_cat.append(cat_vec.numpy())
            token_vectors_sum.append(sum_vec.numpy())

    token_vecs_cat = np.asarray(token_vectors_cat)
    token_vecs_sum = np.asarray(token_vectors_sum)

    return token_vecs_cat, token_vecs_sum, tokenized_text, encoded_layers


def dump_word_vectors(templates_location='../data/openstack/sasho/parsed/logs_aggregated_full.csv_templates',
                      word_embeddings_location='vectors_for_cosine_distance/sasho_bert_vectors_for_cosine.pickle',
                      bert_model='bert-base-uncased'):
    token_vecs_cat, _, tokenized_text, _ = _prepare_bert_vectors(templates_location, bert_model=bert_model)
    # dump words with according vectors in file
    words = []
    for sent in tokenized_text:
        for word in sent:
            words.append(word)

    word_embeddings = [tuple((word, vec)) for word, vec in zip(words, token_vecs_cat)]
    pickle.dump(word_embeddings, open(word_embeddings_location, 'wb'))


def get_sentence_vectors(templates_location='../data/openstack/sasho/parsed/logs_aggregated_full.csv_templates',
                         bert_model='bert-base-uncased'):
    token_vecs_cat, token_vecs_sum, tokenized_text, encoded_layers = _prepare_bert_vectors(
        templates_location, bert_model=bert_model)
    sentence_embeddings = []
    for t in encoded_layers:
        sentence_embeddings.append(torch.mean(t[0][10][0], dim=0))
    return sentence_embeddings


def get_bert_vectors(templates_location='../data/openstack/sasho/parsed/logs_aggregated_full.csv_templates',
                     bert_model='bert-base-uncased'):
    token_vecs_cat, token_vecs_sum, tokenized_text, encoded_layers = _prepare_bert_vectors(
        templates_location, bert_model=bert_model)
    sentence_embeddings = []
    for t in encoded_layers:
        sentence_embeddings.append(torch.mean(t[0][10][0], dim=0))

    return sentence_embeddings, token_vecs_cat, token_vecs_sum, tokenized_text


def get_bert_vectors_from_corpus(
        outputfile="..data/openstack/utah/padded_embeddings_pickle/openstack_18k_anomalies_changed_words.pickle",
        corpus_location="../data/openstack/utah/parsed/anomalies.txt",
        bert_model="bert-base-uncased"):
    lines = open(corpus_location, 'r').readlines()
    tokenizer = transformers_BertTokenizer.from_pretrained(bert_model)
    model = transformers_BertModel.from_pretrained(bert_model)

    encoded_text = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in lines]

    encoded_text_torch = [torch.tensor(sent) for sent in encoded_text]

    word_embeddings = []
    with torch.no_grad():
        [word_embeddings.append(model(encoded_text.unsqueeze(0))[1]) for encoded_text in encoded_text_torch]

    if outputfile:
        pickle.dump(word_embeddings, open(outputfile, 'wb'))

    return word_embeddings


# _, token_vecs_cat, token_vecs_sum, tokenized_text = get_bert_vectors()
# plot_bert(token_vecs_cat, tokenized_text)
# def plot_bert(token_vecs, tokenized_text):
#     flat_tokenized_text = []
#     for sentence in tokenized_text:
#         for word in sentence:
#             flat_tokenized_text.append(word)
#
#     tsne_ak_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=32)
#     embeddings_ak_2d = tsne_ak_2d.fit_transform(token_vecs)
#     tsne_plot_2d('bert.png', 'Word embeddings', embeddings_ak_2d, flat_tokenized_text, a=0.9)

if __name__ == '__main__':
    get_bert_vectors_from_corpus()
