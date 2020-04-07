import matplotlib

matplotlib.use('Agg')
import numpy as np
from logparser.anomaly_injector import insert_words, remove_words, delete_or_duplicate_events, shuffle, no_anomaly, replace_words, reverse_order
from scipy.spatial.distance import cosine
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import os
import tarfile
from wordembeddings.transform_gpt_2 import get_word_embeddings
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer

class TemplatesDataset(Dataset):
    def __init__(self, corpus):
        self.le = LabelEncoder
        self.le.fit(corpus)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, sentence):
        return self.le.transform([sentence])[0]

def get_cosine_distance(lines_before_altering, lines_after_altering, templates, results_dir_exp, vectors):
    lines_before_as_bert_vectors = []
    lines_after_as_bert_vectors = []

    for sentence in lines_before_altering:
        idx = templates.index(sentence)
        if idx is None:
            raise ValueError("{} not found in template file".format(sentence))
        lines_before_as_bert_vectors.append(vectors[idx])

    for sentence in lines_after_altering:
        idx = templates.index(sentence)
        if idx is None:
            raise ValueError("{} not found in template file".format(sentence))
        lines_after_as_bert_vectors.append(vectors[idx])

    cosine_distances = []
    for before, after in zip(lines_before_as_bert_vectors, lines_after_as_bert_vectors):
        cosine_distances.append(cosine(before, after))
    write_lines_to_file(results_dir_exp + "lines_before_after_cosine_distances.txt", cosine_distances, new_line=True)


def write_lines_to_file(file_path, content, new_line=False):
    file = open(file_path, 'w+')
    if new_line:
        [file.write(str(line) + "\n") for line in content]
    else:
        [file.write(str(line)) for line in content]
    file.close()


def inject_anomalies(anomaly_type, corpus_input, corpus_output, anomaly_indices_output_path, instance_information_in,
                     instance_information_out, anomaly_amount, results_dir):
    if anomaly_type in ["insert_words", "remove_words", "replace_words"]:
        if anomaly_type == "insert_words":
            lines_before_alter, lines_after_alter, anomalies_true_label = insert_words(corpus_input, corpus_output,
                                                                                       anomaly_indices_output_path,
                                                                                       instance_information_in,
                                                                                       instance_information_out,
                                                                                       anomaly_amount)
        elif anomaly_type == "remove_words":
            lines_before_alter, lines_after_alter, anomalies_true_label = remove_words(corpus_input, corpus_output,
                                                                                       anomaly_indices_output_path,
                                                                                       instance_information_in,
                                                                                       instance_information_out,
                                                                                       anomaly_amount)
        elif anomaly_type == "replace_words":
            lines_before_alter, lines_after_alter, anomalies_true_label = replace_words(corpus_input, corpus_output,
                                                                                       anomaly_indices_output_path,
                                                                                       instance_information_in,
                                                                                       instance_information_out,
                                                                                       anomaly_amount)

        write_lines_to_file(results_dir + "lines_before_altering.txt", lines_before_alter)
        write_lines_to_file(results_dir + "lines_after_altering.txt", lines_after_alter)
        return anomalies_true_label, lines_before_alter, lines_after_alter

    elif anomaly_type == "duplicate_lines":
        anomalies_true_label = delete_or_duplicate_events(corpus_input, corpus_output, anomaly_indices_output_path,
                                                          instance_information_in, instance_information_out, mode="dup")
    elif anomaly_type == "delete_lines":
        anomalies_true_label = delete_or_duplicate_events(corpus_input, corpus_output, anomaly_indices_output_path,
                                                          instance_information_in, instance_information_out, mode="del")
    elif anomaly_type == "random_lines":
        anomalies_true_label = delete_or_duplicate_events(corpus_input, corpus_output, anomaly_indices_output_path,
                                                          instance_information_in, instance_information_out, mode="ins")
    elif anomaly_type == "shuffle":
        anomalies_true_label = shuffle(corpus_input, corpus_output, instance_information_in,
                                       instance_information_out,
                                       anomaly_indices_output_path)
    elif anomaly_type == "no_anomaly":
        anomalies_true_label = no_anomaly(corpus_input, corpus_output, instance_information_in,
                                          instance_information_out,
                                          anomaly_indices_output_path)
    elif anomaly_type == "reverse_order":
        anomalies_true_label = reverse_order(corpus_input, corpus_output, instance_information_in,
                                             instance_information_out,
                                             anomaly_indices_output_path)
    else:
        print("anomaly type does not exist")
        raise

    return anomalies_true_label, None, None


def calculate_precision_and_plot(this_results_dir_experiment, arg, cwd):
    # distribution_plots(this_results_dir_experiment, arg.epochs, arg.seq_len, 768, 0)

    archive_name = this_results_dir_experiment + "{}_epochs_{}_seq_len_{}_description:_{}_{}".format(arg.embeddings_model, arg.epochs,
                                                                                                     arg.seq_len,
                                                                                                     arg.anomaly_type,
                                                                                                     arg.anomaly_amount) + '.tar'

    with tarfile.open(name=archive_name, mode="w:gz") as tar:
        tar.add(name=cwd + this_results_dir_experiment, arcname=os.path.basename(cwd + this_results_dir_experiment))


def determine_anomalies(anomaly_lstm_model, results_dir, order_of_values_of_file_containing_anomalies, lines_that_have_anomalies,
                        normal_label_embedding_mapping, embeddings_of_log_containing_anomalies):
    # hyperparameters
    thresh = 0.1
    predicted_labels_of_file_containing_anomalies = anomaly_lstm_model.calc_labels()
    order_of_values_of_file_containing_anomalies = open(order_of_values_of_file_containing_anomalies, 'rb').readlines()
    order_of_values_of_file_containing_anomalies = [int(x) for x in order_of_values_of_file_containing_anomalies]
    embeddings_of_log_containing_anomalies = pickle.load(open(embeddings_of_log_containing_anomalies, "rb"))
    assert len(order_of_values_of_file_containing_anomalies) == len(predicted_labels_of_file_containing_anomalies)
    predicted_labels_of_file_containing_anomalies_correct_order = [0] * len(order_of_values_of_file_containing_anomalies)
    for index, label in zip(order_of_values_of_file_containing_anomalies, predicted_labels_of_file_containing_anomalies):
        predicted_labels_of_file_containing_anomalies_correct_order[index] = label

    write_lines_to_file(results_dir + 'anomaly_labels', predicted_labels_of_file_containing_anomalies_correct_order, new_line=True)

    label_to_normal_embeddings_to_anomaly_embeddings_dict = {}
    for label, emb in normal_label_embedding_mapping.items():
        cosine_distances = [cosine(emb, a_emb) for a_emb in embeddings_of_log_containing_anomalies]
        label_to_normal_embeddings_to_anomaly_embeddings_dict.update({label: cosine_distances})


    # see if there are embeddings with distance <= thresh, if none -> anomaly, else: no anomaly
    pred_anomaly_labels = []
    pred_outliers_indeces = []
    for i, label in enumerate(predicted_labels_of_file_containing_anomalies_correct_order):
        cos_distances = label_to_normal_embeddings_to_anomaly_embeddings_dict.get(label)
        if any(x <= thresh for x in cos_distances):
            pred_anomaly_labels.append(0)
        else:
            pred_outliers_indeces.append(i)
            pred_anomaly_labels.append(1)

    write_lines_to_file(results_dir + "pred_outliers_indeces.txt", pred_outliers_indeces, new_line=True)

    # produce labels for f1 score, precision, etc.
    true_labels = np.zeros(len(predicted_labels_of_file_containing_anomalies_correct_order), dtype=int)
    for anomaly_index in lines_that_have_anomalies:
        true_labels[anomaly_index] = 1

    scores_file = open(results_dir + "scores.txt", "w+")
    f1 = f1_score(true_labels, pred_anomaly_labels)
    precision = precision_score(true_labels, pred_anomaly_labels)
    recall = recall_score(true_labels, pred_anomaly_labels)
    accuracy = accuracy_score(true_labels, pred_anomaly_labels)

    scores_file.write("F1-Score: {}\n".format(str(f1)))
    scores_file.write("Precision-Score: {}\n".format(str(precision)))
    scores_file.write("Recall-Score: {}\n".format(str(recall)))
    scores_file.write("Accuracy-Score: {}\n".format(str(accuracy)))
    conf = confusion_matrix(true_labels, pred_anomaly_labels)
    scores_file.write("confusion matrix:\n")
    scores_file.write('\n'.join('\t'.join('%0.3f' % x for x in y) for y in conf))
    scores_file.close()


def get_embeddings(type, templates_location):
    if type == "bert":
        word_embeddings = get_word_embeddings(templates_location, pretrained_weights='bert-base-uncased',
                                              tokenizer_class=BertTokenizer, model_class=BertModel)
    elif type == "gpt2":
        word_embeddings = get_word_embeddings(templates_location, pretrained_weights='gpt2',
                                              tokenizer_class=GPT2Tokenizer, model_class=GPT2Model)
    else:
        raise Exception("unknown embeddings model selected")
    return word_embeddings


# encode corpus into labels
def get_labels_from_corpus(normal_corpus, encoder_path, templates, embeddings):
    if not encoder_path:
        encoder = LabelEncoder()
        encoder.fit(normal_corpus)
        pickle.dump(encoder, open("encoder_normal.pickle", 'wb'))
    else:
        encoder = pickle.load(open(encoder_path, 'rb'))
    target_normal_labels = encoder.transform(normal_corpus)
    normal_label_embeddings_map = {}
    for sent in templates:
        normal_label_embeddings_map.update({encoder.transform([sent])[0]: embeddings.get(sent)})
    n_classes = len(encoder.classes_)
    return target_normal_labels, n_classes, normal_label_embeddings_map


def distribution_plots(dir, vals1, vals2):
    vals1 = [float(x) for x in vals1]
    vals2 = [float(x) for x in vals2]

    sns.distplot(vals1, rug=True, kde=False,
                 kde_kws={'linewidth': 3},
                 label='normal')

    sns.distplot(vals2, rug=True, kde=False,
                 kde_kws={'linewidth': 3},
                 label='anomaly')

    plt.legend(prop={'size': 16}, title='n and a')
    plt.xlabel('Label')
    plt.ylabel('Density')
    plt.savefig(dir + 'plot')
    plt.clf()

def pre_process_log_events(*file):
    for f in file:
        text = open(f, "r").readlines()
        new_text = open(f, "w")
        for line in text:
            line = line.replace(".", "")
            line = line.replace("<*>", "")
            line = line.replace("(", "")
            line = line.replace(")", "")
            line = line.replace(",", "")
            line = line.replace(":", "")
            line = line.replace("[", "")
            line = line.replace("]", "")
            new_text.write(line)
        new_text.close()