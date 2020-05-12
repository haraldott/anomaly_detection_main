import matplotlib

matplotlib.use('Agg')
import numpy as np
from logparser.anomaly_injector import insert_words, remove_words, delete_or_duplicate_events, shuffle, no_anomaly, replace_words, reverse_order
from scipy.spatial.distance import cosine
from numpy import percentile
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import os
import tarfile
from wordembeddings.transform_gpt_2 import get_word_embeddings
from sklearn.preprocessing import LabelEncoder
import pickle
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


def calculate_precision_and_plot(this_results_dir_experiment, cwd, embeddings_model, epochs, seq_len, anomaly_type,
                                 anomaly_amount, n_hidden_units, n_layers, clip, experiment, mode):
    archive_name = this_results_dir_experiment + "{}_{}_epochs_{}_seq_len_{}_anomaly_type_{}_{}_hidden_{}_layers_{}_clip_{}_experiment_{}".format(
        mode, embeddings_model, epochs, seq_len, anomaly_type, anomaly_amount, n_hidden_units, n_layers, clip,
        experiment) + '.tar'

    with tarfile.open(name=archive_name, mode="w:gz") as tar:
        tar.add(name=cwd + this_results_dir_experiment, arcname=os.path.basename(cwd + this_results_dir_experiment))


def calculate_normal_loss(normal_lstm_model, results_dir, values_type, cwd):
    normal_loss_values = normal_lstm_model.loss_values(normal=True)
    write_lines_to_file(cwd + results_dir + values_type, normal_loss_values, True)
    return normal_loss_values



###############################################################
# REGRESSION OUTLIERS
###############################################################

def calculate_anomaly_loss(anomaly_loss_values, normal_loss_values, anomaly_loss_order, anomaly_true_labels, no_anomaly):
    # anomaly_loss_order = open(anomaly_loss_order, 'rb').readlines()
    # anomaly_loss_order = [int(x) for x in anomaly_loss_order]

    assert len(anomaly_loss_order) == len(anomaly_loss_values)
    anomaly_loss_values_correct_order = [0] * len(anomaly_loss_order)
    for index, loss_val in zip(anomaly_loss_order, anomaly_loss_values):
        anomaly_loss_values_correct_order[index] = loss_val

    per = percentile(normal_loss_values, 99.2)

    pred_outliers_indeces = [i for i, val in enumerate(anomaly_loss_values_correct_order) if val > per]
    pred_outliers_loss_values = [val for val in anomaly_loss_values_correct_order if val > per]

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

    result = RegressionResult(f1, precision, recall, accuracy, conf, pred_outliers_indeces, pred_outliers_loss_values,
                              anomaly_loss_values_correct_order, None)
    return result

class ClassificationResult():
    def __init__(self, f1, precision, recall, accuracy, confusion_matrix, predicted_outliers, predicted_labels=None):
        self.f1 = f1
        self.precision = precision
        self.recall = recall
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix
        self.predicted_outliers = predicted_outliers
        self.predicted_labels = predicted_labels # classification

class RegressionResult():
    def __init__(self, f1, precision, recall, accuracy, confusion_matrix, predicted_outliers,
                 pred_outliers_loss_values=None, anomaly_loss_values=None, train_loss_values=None):
        self.f1 = f1
        self.precision = precision
        self.recall = recall
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix
        self.predicted_outliers = predicted_outliers
        self.pred_outliers_loss_values = pred_outliers_loss_values # regression
        self.anomaly_loss_values = anomaly_loss_values # regression
        self.train_loss_values = train_loss_values # regression

def get_top_k_embedding_label_mapping(set_embeddings_of_log_containing_anomalies, normal_label_embedding_mapping):
    top_k = 1
    thresh = 0.35
    top_k_anomaly_embedding_label_mapping = {}
    for sentence, anom_emb in set_embeddings_of_log_containing_anomalies.items():
        cos_distances = {}
        for label, norm_emb in normal_label_embedding_mapping.items():
            cos_distances.update({label: cosine(anom_emb, norm_emb)})
        largest_labels_indeces = heapq.nsmallest(top_k, cos_distances, key=cos_distances.get)
        largest_labels = [i for i in largest_labels_indeces if cos_distances.get(i) < thresh]
        top_k_anomaly_embedding_label_mapping.update({sentence: largest_labels})
    return top_k_anomaly_embedding_label_mapping




###################################################################
# MULTI CLASSIFICATION
###################################################################


class DetermineAnomalies():
    def __init__(self, lines_that_have_anomalies, corpus_of_log_containing_anomalies,
                 top_k_anomaly_embedding_label_mapping, order_of_values_of_file_containing_anomalies):
        self.lines_that_have_anomalies = lines_that_have_anomalies
        self.corpus_of_log_containing_anomalies = open(corpus_of_log_containing_anomalies, 'r').readlines()
        self.top_k_anomaly_embedding_label_mapping = top_k_anomaly_embedding_label_mapping
        self.order_of_values_of_file_containing_anomalies = order_of_values_of_file_containing_anomalies

    def determine(self, predicted_labels_of_file_containing_anomalies, no_anomaly):
        # see if there are embeddings with distance <= thresh, if none -> anomaly, else: no anomaly
        assert len(self.order_of_values_of_file_containing_anomalies) == len(predicted_labels_of_file_containing_anomalies)
        predicted_labels = [0] * len(self.order_of_values_of_file_containing_anomalies)
        for index, l in zip(self.order_of_values_of_file_containing_anomalies, predicted_labels_of_file_containing_anomalies):
            predicted_labels[index] = l

        # produce labels for f1 score, precision, etc.
        true_labels = np.zeros(len(predicted_labels), dtype=int)
        for anomaly_index in self.lines_that_have_anomalies:
            true_labels[anomaly_index] = 1

        wrong_assigned_indeces_min_distance_mapping = {}
        pred_anomaly_labels = []
        pred_outliers_indeces = []
        for i, (top_k_labels_pred, sentence) in enumerate(zip(predicted_labels, self.corpus_of_log_containing_anomalies)):
            most_probable_real_class = self.top_k_anomaly_embedding_label_mapping.get(sentence)
            if bool(set(most_probable_real_class) & set(top_k_labels_pred)):
                # check if we missed
                if (true_labels[i] != 0):
                    distances = []
                    for pred_label in predicted_labels[i]:
                        distances.append(cosine(pred_label, most_probable_real_class))
                    wrong_assigned_indeces_min_distance_mapping[i] = min(distances)
                pred_anomaly_labels.append(0)
            else:
                if (true_labels[i] != 1):
                    distances = []
                    for pred_label in predicted_labels[i]:
                        distances.append(cosine(pred_label, most_probable_real_class))
                    wrong_assigned_indeces_min_distance_mapping[i] = min(distances)
                pred_outliers_indeces.append(i)
                pred_anomaly_labels.append(1)

        # grid search for threshold for which best f1
        temp_pred_anom_labels = pred_anomaly_labels.copy()
        best_f1 = None
        best_thresh = None
        for thresh in np.arange(0,1,0.01):
            for i, dist in wrong_assigned_indeces_min_distance_mapping.items():
                if dist < thresh:
                    temp_pred_anom_labels[i] = 0
                else:
                    temp_pred_anom_labels[i] = 1
            f1 = f1_score(true_labels, temp_pred_anom_labels)
            if best_f1 is None or f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        print("best f1: {}\nbest thresh: {}".format(best_f1, best_thresh))
        # this is a run without anomalies, we have to invert the 0 and 1, otherwise no metric works
        if no_anomaly:
            true_labels = 1 - true_labels
            pred_anomaly_labels = 1 - np.asarray(pred_anomaly_labels)

        f1 = f1_score(true_labels, pred_anomaly_labels)
        precision = precision_score(true_labels, pred_anomaly_labels)
        recall = recall_score(true_labels, pred_anomaly_labels)
        accuracy = accuracy_score(true_labels, pred_anomaly_labels)
        conf = confusion_matrix(true_labels, pred_anomaly_labels)
        result = ClassificationResult(f1=f1, precision=precision, recall=recall, accuracy=accuracy, confusion_matrix=conf,
                                  predicted_outliers=pred_outliers_indeces, predicted_labels=predicted_labels)

        return result

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



###################################################################
# BINARY
###################################################################


def determine_binary_anomalies(predicted_labels_of_file_containing_anomalies,
                               order_of_values_of_file_containing_anomalies, lines_that_have_anomalies, no_anomaly):

    assert len(order_of_values_of_file_containing_anomalies) == len(predicted_labels_of_file_containing_anomalies)
    predicted_labels = [0] * len(order_of_values_of_file_containing_anomalies)
    for index, label in zip(order_of_values_of_file_containing_anomalies, predicted_labels_of_file_containing_anomalies):
        predicted_labels[index] = int(label)

    # logging of indices of the outliers, every "1" in predicted_labels is an outlier, so log its index
    pred_outliers_indeces = []
    for i, val in enumerate(predicted_labels):
        if val == 1:
            pred_outliers_indeces.append(i)

    # produce labels for f1 score, precision, etc.
    true_labels = np.zeros(len(predicted_labels), dtype=int)
    for anomaly_index in lines_that_have_anomalies:
        true_labels[anomaly_index] = 1

    # this is a run without anomalies, we have to invert the 0 and 1, otherwise no metric works
    if no_anomaly:
        true_labels = 1 - true_labels
        predicted_labels = 1 - np.asarray(predicted_labels)

    f1 = f1_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    conf = confusion_matrix(true_labels, predicted_labels)

    result = ClassificationResult(f1=f1, precision=precision, recall=recall, accuracy=accuracy, confusion_matrix=conf,
                                  predicted_outliers=pred_outliers_indeces, predicted_labels=predicted_labels)

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