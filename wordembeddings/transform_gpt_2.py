import torch
import pickle


def get_word_embeddings(templates, pretrained_weights, tokenizer_class, model_class):
    tokenized_text, encoded_layers = _prepare_vectors(templates=templates,
                                                      pretrained_weights=pretrained_weights,
                                                      tokenizer_class=tokenizer_class,
                                                      model_class=model_class)
    # dump words with according vectors in file
    words = []
    for sent in tokenized_text:
        for word in sent:
            words.append(word)

    word_embeddings = [tuple((word, vec)) for word, vec in zip(words, encoded_layers)]
    sentence_embeddings_dict = {}
    for template, sentence_embedding in zip(templates, word_embeddings):
        sentence_embeddings_dict.update({template: torch.squeeze(sentence_embedding[1].mean(dim=1))})
    return sentence_embeddings_dict


def _prepare_vectors(templates, pretrained_weights, tokenizer_class, model_class):
    if type(templates) == str:
        lines = open(templates, 'r')
    elif type(templates) == list:
        lines = templates
    else:
        print("templates must be either a list of templates or a path str")
        raise

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    tokenized_text = [tokenizer.tokenize(sentence) for sentence in lines]

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized_text]
    tokens_tensors = [torch.tensor([idx_tokens]) for idx_tokens in input_ids]
    with torch.no_grad():
        encoded_layers = [model(t)[0] for t in tokens_tensors]

    return tokenized_text, encoded_layers


def get_sentence_vectors(templates, pretrained_weights, tokenizer_class, model_class):
    tokenized_text, encoded_layers = _prepare_vectors(templates=templates,
                                                      pretrained_weights=pretrained_weights,
                                                      tokenizer_class=tokenizer_class,
                                                      model_class=model_class)
    # dump words with according vectors in file
    words = []
    for sent in tokenized_text:
        for word in sent:
            words.append(word)

    word_embeddings = [tuple((word, vec)) for word, vec in zip(words, encoded_layers)]
    sentence_embeddings = []
    for template, sentence_embedding in zip(templates, word_embeddings):
        sentence_embeddings.append(torch.squeeze(sentence_embedding[1].mean(dim=1)))
    return sentence_embeddings


def dump_word_vectors(templates_location='../data/openstack/utah/parsed/18k_spr_templates',
                      word_embeddings_location='vectors_for_cosine_distance/sasho_gpt2_vectors_for_cosine.pickle',
                      bert_model='gpt2'):
    tokenized_text, encoded_layers = _prepare_vectors(templates=templates_location, pretrained_weights=bert_model)
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
