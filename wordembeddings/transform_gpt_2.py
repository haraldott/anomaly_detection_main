import torch
from transformers import GPT2Model, GPT2Tokenizer
import pickle


def get_gpt2_embeddings(templates, model):
    tokenized_text, encoded_layers = _prepare_gpt2_vectors(templates=templates,
                                                           gpt2_model=model)
    # dump words with according vectors in file
    words = []
    for sent in tokenized_text:
        for word in sent:
            words.append(word)

    word_embeddings = [tuple((word, vec)) for word, vec in zip(words, encoded_layers)]
    sentence_embeddings_mean = []
    for sentence in word_embeddings:
        sentence_embeddings_mean.append(torch.squeeze(sentence[1].mean(dim=1)))
    return sentence_embeddings_mean


def _prepare_gpt2_vectors(templates='../data/openstack/sasho/parsed/logs_aggregated_full.csv_templates',
                          gpt2_model='gpt2'):
    if type(templates) == str:
        lines = open(templates, 'r')
    elif type(templates) == list:
        lines = templates
    else:
        print("templates must be either a list of templates or a path str")
        raise
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
    model = GPT2Model.from_pretrained(gpt2_model)

    tokenized_text = [tokenizer.tokenize(sentence) for sentence in lines]

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized_text]
    tokens_tensors = [torch.tensor([idx_tokens]) for idx_tokens in input_ids]
    with torch.no_grad():
        encoded_layers = [model(t)[0] for t in tokens_tensors]

    return tokenized_text, encoded_layers


def dump_word_vectors(templates_location='../data/openstack/utah/parsed/18k_spr_templates',
                      word_embeddings_location='vectors_for_cosine_distance/sasho_gpt2_vectors_for_cosine.pickle',
                      bert_model='gpt2'):
    tokenized_text, encoded_layers = _prepare_gpt2_vectors(templates=templates_location, gpt2_model=bert_model)
    # dump words with according vectors in file
    words = []
    for sent in tokenized_text:
        for word in sent:
            words.append(word)

    word_embeddings = [tuple((word, vec)) for word, vec in zip(words, encoded_layers)]
    sentence_embeddings_mean = []
    for sentence in word_embeddings:
        sentence_embeddings_mean.append(sentence[1].mean(dim=1))
    pickle.dump(sentence_embeddings_mean, open(word_embeddings_location, 'wb'))


if __name__ == '__main__':
    dump_word_vectors()
