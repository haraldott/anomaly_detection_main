import matplotlib

matplotlib.use('Agg')
import numpy as np
from logparser.anomaly_injector import insert_words, remove_words, delete_or_duplicate_events, shuffle, no_anomaly, replace_words, reverse_order
from scipy.spatial.distance import cosine
from numpy import percentile
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from tools import distribution_plots as distribution_plots
import os
import tarfile
from wordembeddings.transform_gpt_2 import get_gpt2_embeddings
from wordembeddings.transform_bert import get_bert_embeddings


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


def calculate_precision_and_plot(this_results_dir_experiment, epochs, seq_len, embeddings_model, anomaly_type, anomaly_amount, cwd):
    distribution_plots(this_results_dir_experiment, epochs, seq_len, 768, 0)

    archive_name = this_results_dir_experiment + "{}_epochs_{}_seq_len_{}_description:_{}_{}".format(embeddings_model, epochs,
                                                                                                     seq_len,
                                                                                                     anomaly_type,
                                                                                                     anomaly_amount) + '.tar'

    with tarfile.open(name=archive_name, mode="w:gz") as tar:
        tar.add(name=cwd + this_results_dir_experiment, arcname=os.path.basename(cwd + this_results_dir_experiment))


def calculate_normal_loss(normal_lstm_model, results_dir, values_type, cwd):
    normal_loss_values = normal_lstm_model.loss_values(normal=True)
    write_lines_to_file(cwd + results_dir + values_type, normal_loss_values, True)
    return normal_loss_values


def calculate_anomaly_loss(anomaly_loss_values, normal_loss_values, anomaly_loss_order, anomaly_true_labels, no_anomaly):
    # anomaly_loss_order = open(anomaly_loss_order, 'rb').readlines()
    # anomaly_loss_order = [int(x) for x in anomaly_loss_order]

    assert len(anomaly_loss_order) == len(anomaly_loss_values)
    anomaly_loss_values_correct_order = [0] * len(anomaly_loss_order)
    for index, loss_val in zip(anomaly_loss_order, anomaly_loss_values):
        anomaly_loss_values_correct_order[index] = loss_val

    per = percentile(normal_loss_values, 100)

    pred_outliers_indeces = [i for i, val in enumerate(anomaly_loss_values_correct_order) if val > per]
    pred_outliers_values = [val for val in anomaly_loss_values_correct_order if val > per]

    # produce labels for f1 score, precision, etc.
    pred_labels = np.zeros(len(anomaly_loss_values_correct_order), dtype=int)
    for anomaly_index in pred_outliers_indeces:
        pred_labels[anomaly_index] = 1

    true_labels = np.zeros(len(anomaly_loss_values_correct_order), dtype=int)
    for anomaly_index in anomaly_true_labels:
        true_labels[anomaly_index] = 1

    # this is a run without anomalies, we have to invert the 0 and 1, otherwise no metric works
    if no_anomaly:
        true_labels = 1 - true_labels
        pred_labels = 1 - np.asarray(pred_labels)

    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    conf = confusion_matrix(true_labels, pred_labels)

    result = PredictionResult(f1, precision, recall, accuracy, conf, pred_outliers_indeces, pred_outliers_values,
                              anomaly_loss_values_correct_order, None)
    return result

class PredictionResult():
    def __init__(self, f1, precision, recall, accuracy, confusion_matrix, predicted_outliers, pred_outliers_values,
                 anomaly_loss_values_correct_order, train_loss_values):
        self.f1 = f1
        self.precision = precision
        self.recall = recall
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix
        self.predicted_outliers = predicted_outliers
        self.pred_outliers_values = pred_outliers_values
        self.anomaly_loss_values_correct_order = anomaly_loss_values_correct_order
        self.train_loss_values = train_loss_values


def get_embeddings(type, templates_location, finetuning_model_dir):
    if type == "bert":
        word_embeddings = get_bert_embeddings(templates_location, model=finetuning_model_dir)
    elif type == "gpt2":
        word_embeddings = get_gpt2_embeddings(templates_location, model='gpt2')
    else:
        raise Exception("unknown embeddings model selected")
    return word_embeddings