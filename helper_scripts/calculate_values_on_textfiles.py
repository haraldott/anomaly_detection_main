from sklearn.metrics import f1_score, precision_score
import numpy as np
from numpy import percentile


def calc_mean():
    with open('/Users/haraldott/Downloads/results/no finetune/bert_epochs_0_seq_len_7_description__reverse_order_0/anomaly_loss_values') as f:
        reverse_order_loss = [float(x) for x in f.readlines()]
    with open('/Users/haraldott/Downloads/results/no finetune/bert_epochs_100_seq_len_7_description__no_anomaly_1/anomaly_loss_values') as f:
        no_anomaly_loss = [float(x) for x in f.readlines()]

    print("reverse_order_loss: {}, std: {}".format(np.mean(reverse_order_loss), np.std(reverse_order_loss)))
    print("no_anomaly_loss: {}, std: {}".format(np.mean(no_anomaly_loss), np.std(no_anomaly_loss)))



def calc_percentile_outliers(normal_loss_values_path="/Users/haraldott/Downloads/results/regression/bert_epochs_130_seq_len_7_description__random_lines_1/normal_loss_values",
                             anomaly_loss_values_path="/Users/haraldott/Downloads/results/regression/bert_epochs_130_seq_len_7_description__random_lines_1/anomaly_loss_values",
                             perc=97.0):
    with open(normal_loss_values_path) as f:
        normal_loss_values = [float(y) for y in f.readlines()]
    with open(anomaly_loss_values_path) as f:
        anomaly_loss_values = [float(y) for y in f.readlines()]

    per = percentile(normal_loss_values, perc)

    pred_outliers_indeces = [i for i, val in enumerate(anomaly_loss_values) if val > per]
    print(len(pred_outliers_indeces))
    return pred_outliers_indeces



def calc_f1_based_on_percentile(normal_loss_values_path="/Users/haraldott/Downloads/regression - strange behaviour, spikes during training/normal_loss_values",
                                anomaly_loss_values_path="/Users/haraldott/Downloads/regression - strange behaviour, spikes during training/anomaly_loss_values",
                                ground_truth_path="/Users/haraldott/Downloads/regression - strange behaviour, spikes during training/true_anomaly_labels.txt"):
    with open(ground_truth_path) as f:
        ground_truth = [int(y) for y in f.readlines()]
    pred_indeces = calc_percentile_outliers(normal_loss_values_path, anomaly_loss_values_path, 99.2)

    with open(anomaly_loss_values_path) as f:
        anomaly_loss_values = f.readlines()

    pred_labels = np.zeros(len(anomaly_loss_values), dtype=int)
    for index_pred in pred_indeces:
        pred_labels[index_pred] = 1

    true_labels = np.zeros(len(anomaly_loss_values), dtype=int)
    for index_true in ground_truth:
        true_labels[index_true] = 1

    print("f1 score: {}".format(f1_score(true_labels, pred_labels)))
    print("precision: {}".format(precision_score(true_labels, pred_labels)))


if __name__ == '__main__':
    calc_f1_based_on_percentile()