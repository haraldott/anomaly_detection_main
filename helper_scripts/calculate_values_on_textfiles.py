from sklearn.metrics import f1_score, precision_score
import numpy as np
from numpy import percentile


def calc_mean():
    with open('/Users/haraldott/Downloads/results/with finetuning/bert_epochs_100_seq_len:_7_anomaly_type:insert_words_9/lines_before_after_cosine_distances.txt') as f:
        with_finetune_9 = [float(x) for x in f.readlines()]
    with open('/Users/haraldott/Downloads/results/no finetune/bert_epochs_100_seq_len:_7_anomaly_type:insert_words_9/lines_before_after_cosine_distances.txt') as f:
        without_finetune_9 = [float(x) for x in f.readlines()]

    print("with_finetune_9: {}".format(np.mean(with_finetune_9)))
    print("without_finetune_9: {}".format(np.mean(without_finetune_9)))



def calc_percentile_outliers(normal_loss_values_path="/Users/haraldott/Downloads/results/Transfer Sasho Utah/bert_epochs_100_seq_len:_7_anomaly_type:insert_words_1/normal_loss_values",
                             anomaly_loss_values_path="/Users/haraldott/Downloads/results/Transfer Sasho Utah/bert_epochs_100_seq_len:_7_anomaly_type:insert_words_1/anomaly_loss_values",
                             perc=96.1):
    with open(normal_loss_values_path) as f:
        normal_loss_values = [float(y) for y in f.readlines()]
    with open(anomaly_loss_values_path) as f:
        anomaly_loss_values = [float(y) for y in f.readlines()]

    per = percentile(normal_loss_values, perc)

    pred_outliers_indeces = [i for i, val in enumerate(anomaly_loss_values) if val > per]
    print(len(pred_outliers_indeces))
    return pred_outliers_indeces



def calc_f1_based_on_percentile(normal_loss_values_path="/Users/haraldott/Downloads/results/Transfer Sasho Utah/bert_epochs_100_seq_len:_7_anomaly_type:insert_words_1/normal_loss_values",
                                anomaly_loss_values_path="/Users/haraldott/Downloads/results/Transfer Sasho Utah/bert_epochs_100_seq_len:_7_anomaly_type:insert_words_1/anomaly_loss_values",
                                ground_truth_path="/Users/haraldott/Downloads/results/Transfer Sasho Utah/bert_epochs_100_seq_len:_7_anomaly_type:insert_words_1/18k_spr_insert_words_1_anomaly_indeces.txt"):
    with open(ground_truth_path) as f:
        ground_truth = [int(y) for y in f.readlines()]
    pred_indeces = calc_percentile_outliers(normal_loss_values_path, anomaly_loss_values_path, 97)

    with open("/Users/haraldott/Downloads/results/Transfer Sasho Utah/bert_epochs_100_seq_len:_7_anomaly_type:insert_words_1/anomaly_loss_values") as f:
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
    calc_mean()