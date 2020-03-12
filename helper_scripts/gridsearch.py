from sklearn.metrics import f1_score
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, plot_confusion_matrix

anomaly_file = open('/Users/haraldott/Development/thesis/detector/data/openstack/utah/raw/'
                    'sorted_per_request/openstack_18k_anomalies_sorted_per_request', 'r')
log_lines_containing_anomalies = anomaly_file.readlines()

instances_containing_anomalies = [
    "544fd51c-4edc-4780-baae-ba1d80a0acfc",
    "ae651dff-c7ad-43d6-ac96-bbcd820ccca8",
    "a445709b-6ad0-40ec-8860-bec60b6ca0c2",
    "1643649d-2f42-4303-bfcd-7798baec19f9"
]
anomaly_idx_groups = {}
indices_from_log_containing_anomalies = open('/Users/haraldott/Google Drive/Masterarbeit/Meetings/2020-02-24/bert_epochs_100_hiddenunits_250_no_finetune_no_ring/anomaly_loss_indices', 'r')
indices_from_log_containing_anomalies = [int(y) for y in indices_from_log_containing_anomalies.readlines()]

anomaly_lines_filtered = []
for index in indices_from_log_containing_anomalies:
    anomaly_lines_filtered.append(log_lines_containing_anomalies[index])

for i, line in enumerate(anomaly_lines_filtered):
    found_instance_id = None
    for instance_id in instances_containing_anomalies:
        if instance_id in line:
            found_instance_id = instance_id
    if found_instance_id:
        anomaly_idx_groups.setdefault(found_instance_id, []).append(i)

normal = open('/Users/haraldott/Google Drive/Masterarbeit/Meetings/2020-02-24/bert_epochs_100_hiddenunits_250_no_finetune_no_ring/normal_loss_values', 'r')
anomaly = open('/Users/haraldott/Google Drive/Masterarbeit/Meetings/2020-02-24/bert_epochs_100_hiddenunits_250_no_finetune_no_ring/anomaly_loss_values', 'r')


normal_values = [float(y) for y in normal.readlines()]
log_containing_anomalies = [float(y) for y in anomaly.readlines()]


true_labels = np.zeros(len(anomaly_lines_filtered), dtype=int)
for instance_id in anomaly_idx_groups:
    for val in anomaly_idx_groups[instance_id]:
        true_labels[val] = 1

mean = np.mean(normal_values)
std = np.std(normal_values)

steps_threshold = np.arange(0.0, 0.8, 1e-4)
steps_percentage = np.arange(0.1, 0.5, 1e-2)
f1_scores = []
best_score_val = None
best_score_threshold = None
best_score_percentage = None
best_score_true_labels = None

number_of_correctly_identified_anomalies_per_instance_id = defaultdict(int)
for current_threshold in steps_threshold:
    prediction_labels = np.zeros(len(indices_from_log_containing_anomalies), dtype=int)
    indices_predicted_as_anomalies = []
    for index, anomaly_loss_value in enumerate(log_containing_anomalies):
        if anomaly_loss_value > current_threshold:
            prediction_labels[index] = 1
            indices_predicted_as_anomalies.append(index)
    # go through all indices that have been predicted as anomalies, and compare them block-wise with the actual
    # indices containing anomalies. Try to find a ratio
    for predicted_index in indices_predicted_as_anomalies:
        for instance_id in anomaly_idx_groups:
            for line_number in anomaly_idx_groups[instance_id]:
                if predicted_index == line_number:
                    number_of_correctly_identified_anomalies_per_instance_id[instance_id] += 1
    percentage_of_detected_anomaly_lines_per_instance_id = defaultdict(float)
    # increase threshold for which we will mark all anomalies in a block as identified as anomalies
    for current_percentage_threshold in steps_percentage:
        temp_true_labels = true_labels.copy()
        for inst_id in number_of_correctly_identified_anomalies_per_instance_id:
            # for this inst_id, this is the percentage of correctly identified anomalies
            percentage_identified = len(anomaly_idx_groups[inst_id]) / number_of_correctly_identified_anomalies_per_instance_id[inst_id]
            # if we have identfied >= current_percentage_threshold % of the anomalies of a block, we will mark all of
            # the log lines of the block as anomalies
            if percentage_identified >= current_percentage_threshold:
                print("great success")
                for log_line in anomaly_idx_groups[inst_id]:
                    temp_true_labels[log_line] = 1
    score = f1_score(temp_true_labels, prediction_labels)
    print("score:{}, threshold: {}".format(score, current_threshold))
    if not best_score_val or score > best_score_val:
        best_score_val = score
        best_score_threshold = current_threshold
        best_score_percentage = current_percentage_threshold
        best_score_true_labels = temp_true_labels
        best_prediction_labels = prediction_labels

cm = confusion_matrix(y_true=best_score_true_labels, y_pred=best_prediction_labels, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
print("best_score_threshold: {}".format(best_score_threshold))

# NOTE: Fill all variables here with default values of the plot_confusion_matrix
disp = disp.plot(include_values=True)
plt.show()
