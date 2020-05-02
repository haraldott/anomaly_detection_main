import os

# input_dir = "/Users/haraldott/Downloads/with_finetuning: for every different anomaly injection/"
# subdirs = [x[0] for x in os.walk(input_dir)]
#
# for dir in subdirs[1:]:
dir = "/Users/haraldott/Downloads/transfer: finetune on sasho, 5 shot on utah, utah anomaly/bert_epochs_100_seq_len:_7_anomaly_type:insert_words_6"
normal_loss_values = open(dir + "/normal_loss_values").readlines()
anomaly_loss_values = open(dir + "/anomaly_loss_values").readlines()
anomaly_loss_indeces = open(dir + "/anomaly_indices_order").readlines()

max_normal_loss_value = max(normal_loss_values)

anomaly_values_bigger = [i for i, val in zip(anomaly_loss_indeces, anomaly_loss_values) if val > max_normal_loss_value]

anomaly_loss_values_bigger_file = open(dir + "/anomaly_loss_values_bigger.txt", "w")
for val in anomaly_values_bigger:
    anomaly_loss_values_bigger_file.write(val)
anomaly_loss_values_bigger_file.close()