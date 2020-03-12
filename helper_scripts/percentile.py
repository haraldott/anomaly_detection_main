from scipy.stats import iqr
from numpy import percentile

#anomaly_indices = open('/Users/haraldott/Downloads/bert_epochs_100_seq_len_7_description__None_10/18k_spr_insert_words_10_anomaly_indeces.txt', 'r').readlines()
normal_loss_values = open('/Users/haraldott/Downloads/bert_epochs_100_seq_len_7_description__no_anomaly_1/normal_loss_values', 'r').readlines()
anomaly_loss_values = open('/Users/haraldott/Downloads/bert_epochs_100_seq_len_7_description__no_anomaly_1/anomaly_loss_values', 'r').readlines()
anomaly_loss_indices = open('/Users/haraldott/Downloads/bert_epochs_100_seq_len_7_description__no_anomaly_1/anomaly_loss_indices', 'r').readlines()

#anomaly_indices = [int(y) for y in anomaly_indices]
normal_loss_values = [float(y) for y in normal_loss_values]
anomaly_loss_values = [float(y) for y in anomaly_loss_values]
anomaly_loss_indices = [int(y) for y in anomaly_loss_indices]

per = percentile(normal_loss_values, 96.0)

loss_values_higher_than_per = []
for index, val in zip(anomaly_loss_indices, anomaly_loss_values):
    if val > per:
        loss_values_higher_than_per.append(index)



# iqr_val = iqr(loss_values_normal)
# len(set(loss_values_higher_than_per) & set(anomaly_indices))