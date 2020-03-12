from sklearn.metrics import f1_score, precision_score

ground_truth = open("/Users/haraldott/Downloads/transfer: finetune on sasho, 5 shot on utah, utah anomaly/bert_epochs_100_seq_len:_7_anomaly_type:random_lines_1/18k_spr_random_lines_1_anomaly_indeces.txt", 'r').readlines()
pred_indeces = open("/Users/haraldott/Downloads/transfer: finetune on sasho, 5 shot on utah, utah anomaly/bert_epochs_100_seq_len:_7_anomaly_type:random_lines_1/anomaly_loss_values_bigger.txt", 'r').readlines()

ground_truth = [int(y) for y in ground_truth]
pred_indeces = [int(y) for y in pred_indeces]

print("f1 score: {}".format(f1_score(ground_truth, pred_indeces)))
print("precision: {}".format(precision_score(ground_truth, pred_indeces)))
