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
import heapq

def transfer_labels(dataset1_templates, dataset2_templates, dataset2_corpus, word_embeddings, template_class_mapping,
                    results_dir):
    dataset2_corpus = open(dataset2_corpus, 'r').readlines()
    dataset_2_class_mapping = {}
    dataset_2_sentence_sentence_mapping = {}
    dataset_2_sentence_class_mapping = {}
    for template_2 in dataset2_templates:
        template_2_embedding = word_embeddings.get(template_2)
        smallest_cos_distance = 1
        for template_1 in dataset1_templates:
            template_1_embedding = word_embeddings.get(template_1)
            distance = cosine(template_2_embedding, template_1_embedding)
            if distance < smallest_cos_distance:
                smallest_cos_distance = distance
                smallest_cos_template = template_1
        corresponding_class = template_class_mapping.get(smallest_cos_template)
        dataset_2_class_mapping.update({corresponding_class: template_2_embedding})
        dataset_2_sentence_sentence_mapping.update({template_2: smallest_cos_template})
        dataset_2_sentence_class_mapping.update({template_2: corresponding_class})

    # write sentence to sentence mapping to file
    with open(results_dir + "sentence_to_sentence_mapping.txt", 'w') as f:
        for k, v in dataset_2_sentence_sentence_mapping.items():
            f.write(k + ": " + v + "\n")
    # do sentence to class mapping
    dataset_2_corpus_target_labels = [dataset_2_sentence_class_mapping.get(sentence) for sentence in dataset2_corpus]
    return dataset_2_corpus_target_labels


class TemplatesDataset(Dataset):
    def __init__(self, corpus):
        self.le = LabelEncoder
        self.le.fit(corpus)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, sentence):
        return self.le.transform([sentence])[0]

def get_cosine_distance(lines_before_altering, lines_after_altering, results_dir_exp, vectors):
    lines_before_as_bert_vectors = []
    lines_after_as_bert_vectors = []

    for sentence_b in lines_before_altering:
        emb = vectors.get(sentence_b)
        if emb is not None:
            lines_before_as_bert_vectors.append(emb)
        else:
            raise ValueError("{} not found in template file".format(sentence_b))

    for sentence_a in lines_after_altering:
        emb = vectors.get(sentence_a)
        if emb is not None:
            lines_after_as_bert_vectors.append(emb)
        else:
            raise ValueError("{} not found in template file".format(sentence_a))

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


def calculate_precision_and_plot(this_results_dir_experiment, embeddings_model, epochs, seq_len,
                                 anomaly_type, anomaly_amount, cwd):
    archive_name = this_results_dir_experiment + "{}_epochs_{}_seq_len_{}_description:_{}_{}".format(embeddings_model, epochs,
                                                                                                     seq_len,
                                                                                                     anomaly_type,
                                                                                                     anomaly_amount) + '.tar'

    with tarfile.open(name=archive_name, mode="w:gz") as tar:
        tar.add(name=cwd + this_results_dir_experiment, arcname=os.path.basename(cwd + this_results_dir_experiment))


def get_top_k_embedding_label_mapping(set_embeddings_of_log_containing_anomalies, normal_label_embedding_mapping):
    top_k = 3
    thresh = 0.25
    top_k_anomaly_embedding_label_mapping = {}
    for sentence, anom_emb in set_embeddings_of_log_containing_anomalies.items():
        cos_distances = {}
        for label, norm_emb in normal_label_embedding_mapping.items():
            cos_distances.update({label: cosine(anom_emb, norm_emb)})
        largest_labels_indeces = heapq.nsmallest(top_k, cos_distances, key=cos_distances.get)
        largest_labels = [i for i in largest_labels_indeces if cos_distances.get(i) < thresh]
        top_k_anomaly_embedding_label_mapping.update({sentence: largest_labels})
    return top_k_anomaly_embedding_label_mapping

class PredictionResult():
    def __init__(self, f1, precision, recall, accuracy, confusion_matrix, predicted_outliers,
                 predicted_labels_of_file_containing_anomalies_correct_order):
        self.f1 = f1
        self.precision = precision
        self.recall = recall
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix
        self.predicted_outliers = predicted_outliers
        self.predicted_labels_of_file_containing_anomalies_correct_order = predicted_labels_of_file_containing_anomalies_correct_order


class DetermineAnomalies():
    def __init__(self, lines_that_have_anomalies, corpus_of_log_containing_anomalies,
                 top_k_anomaly_embedding_label_mapping, order_of_values_of_file_containing_anomalies):
        self.lines_that_have_anomalies = lines_that_have_anomalies
        self.corpus_of_log_containing_anomalies = open(corpus_of_log_containing_anomalies, 'r').readlines()
        self.top_k_anomaly_embedding_label_mapping = top_k_anomaly_embedding_label_mapping
        self.order_of_values_of_file_containing_anomalies = order_of_values_of_file_containing_anomalies

    def determine(self, predicted_labels_of_file_containing_anomalies):
        # see if there are embeddings with distance <= thresh, if none -> anomaly, else: no anomaly
        assert len(self.order_of_values_of_file_containing_anomalies) == len(predicted_labels_of_file_containing_anomalies)
        predicted_labels_of_file_containing_anomalies_correct_order = [0] * len(
            self.order_of_values_of_file_containing_anomalies)
        for index, label in zip(self.order_of_values_of_file_containing_anomalies,
                                predicted_labels_of_file_containing_anomalies):
            predicted_labels_of_file_containing_anomalies_correct_order[index] = label

        pred_anomaly_labels = []
        pred_outliers_indeces = []
        for i, (label, sentence) in enumerate(
                zip(predicted_labels_of_file_containing_anomalies_correct_order, self.corpus_of_log_containing_anomalies)):
            if label in self.top_k_anomaly_embedding_label_mapping.get(sentence):
                pred_anomaly_labels.append(0)
            else:
                pred_outliers_indeces.append(i)
                pred_anomaly_labels.append(1)

        # produce labels for f1 score, precision, etc.
        true_labels = np.zeros(len(predicted_labels_of_file_containing_anomalies_correct_order), dtype=int)
        for anomaly_index in self.lines_that_have_anomalies:
            true_labels[anomaly_index] = 1

        f1 = f1_score(true_labels, pred_anomaly_labels)
        precision = precision_score(true_labels, pred_anomaly_labels)
        recall = recall_score(true_labels, pred_anomaly_labels)
        accuracy = accuracy_score(true_labels, pred_anomaly_labels)
        conf = confusion_matrix(true_labels, pred_anomaly_labels)
        result = PredictionResult(f1, precision, recall, accuracy, conf, pred_outliers_indeces,
                                  predicted_labels_of_file_containing_anomalies_correct_order)

        return result


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
    normal_template_class_map = {}
    for sent in templates:
        normal_label_embeddings_map.update({encoder.transform([sent])[0]: embeddings.get(sent)})
        normal_template_class_map.update({sent: encoder.transform([sent])[0]})
    n_classes = len(encoder.classes_)
    return target_normal_labels, n_classes, normal_label_embeddings_map, normal_template_class_map


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