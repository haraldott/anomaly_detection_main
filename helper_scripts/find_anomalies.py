import numpy as np
from sklearn.metrics import f1_score, precision_score

anomaly_indices = open('/Users/haraldott/Downloads/transfer: finetune on sasho, 5 shot on utah, utah anomaly/bert_epochs_100_seq_len:_7_anomaly_type:insert_words_6/18k_spr_insert_words_6_anomaly_indeces.txt', 'r').readlines()
loss_values_normal = open('/Users/haraldott/Downloads/transfer: finetune on sasho, 5 shot on utah, utah anomaly/bert_epochs_100_seq_len:_7_anomaly_type:insert_words_6/normal_loss_values', 'r').readlines()
loss_values_containing_anomalies = open('/Users/haraldott/Downloads/transfer: finetune on sasho, 5 shot on utah, utah anomaly/bert_epochs_100_seq_len:_7_anomaly_type:insert_words_6/anomaly_loss_values', 'r').readlines()
anomaly_file_indices_information = open('/Users/haraldott/Downloads/transfer: finetune on sasho, 5 shot on utah, utah anomaly/bert_epochs_100_seq_len:_7_anomaly_type:insert_words_6/anomaly_loss_indices', 'r').readlines()

anomaly_indices = [int(y) for y in anomaly_indices]
loss_values_normal = [float(y) for y in loss_values_normal]
loss_values_containing_anomalies = [float(y) for y in loss_values_containing_anomalies]
anomaly_file_indices_information = [int(y) for y in anomaly_file_indices_information]

true_labels = np.zeros(len(loss_values_containing_anomalies), dtype=int)
for anomaly_index in anomaly_indices:
    true_labels[anomaly_index] = 1


assert all(val > 0 for val in loss_values_containing_anomalies), "there are loss values < 0, this part of the code only works for > 0 as of now."
max_val_loss_values_containing_anomalies = max(line for line in loss_values_containing_anomalies)
steps_threshold = np.arange(0.0, max_val_loss_values_containing_anomalies, 1e-4)
f1_scores = []
best_f1_score = None
best_score_threshold = None
best_precision = None

for current_threshold in steps_threshold:
    prediction_labels = np.zeros(len(loss_values_containing_anomalies), dtype=int)
    indices_predicted_as_anomalies = []
    for index, anomaly_loss_value in zip(anomaly_file_indices_information, loss_values_containing_anomalies):
        if anomaly_loss_value > current_threshold:
            prediction_labels[index] = 1
            indices_predicted_as_anomalies.append(index)
    f1 = f1_score(y_true=true_labels, y_pred=prediction_labels)
    precision = precision_score(y_true=true_labels, y_pred=prediction_labels)
    if not best_f1_score or f1 > best_f1_score:
        best_f1_score = f1
        best_score_threshold = current_threshold
    if not best_precision or precision > best_precision:
        best_precision = precision
        best_precision_threshold = current_threshold