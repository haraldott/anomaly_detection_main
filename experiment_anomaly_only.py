from os import makedirs

from normal_learning import experiment

results_folder = 'results_other/'
makedirs(results_folder, exist_ok=True)

# BERT

anomaly_ratio = 0.02
with open(results_folder + 'bert_multiclass_anomaly_only.txt', 'a') as results:
    while anomaly_ratio <= 0.20:
        f1, precision, recall = experiment(anomaly_type='random_lines', anomaly_amount=1, mode="multiclass",
                                   prediction_only=True, anomaly_ratio=anomaly_ratio, alteration_ratio=0.00,
                                   embeddings_model='bert', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(anomaly_ratio, f1, precision, recall))
        results.flush()
        anomaly_ratio = round(anomaly_ratio + 0.02, 2)

anomaly_ratio = 0.02
with open(results_folder + 'bert_regression_anomaly_only.txt', 'a') as results:
    while anomaly_ratio <= 0.20:
        f1, precision, recall = experiment(anomaly_type='random_lines', anomaly_amount=1, mode="regression",
                                   prediction_only=True, anomaly_ratio=anomaly_ratio, alteration_ratio=0.00,
                                   embeddings_model='bert', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(anomaly_ratio, f1, precision, recall))
        results.flush()
        anomaly_ratio = round(anomaly_ratio + 0.02, 2)


# GPT

anomaly_ratio = 0.02
with open(results_folder + 'gpt2_multiclass_anomaly_only.txt', 'a') as results:
    while anomaly_ratio <= 0.20:
        f1, precision, recall = experiment(anomaly_type='random_lines', anomaly_amount=1, mode="multiclass",
                                           prediction_only=True, anomaly_ratio=anomaly_ratio, alteration_ratio=0.00,
                                           embeddings_model='gpt2', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(anomaly_ratio, f1, precision, recall))
        results.flush()
        anomaly_ratio = round(anomaly_ratio + 0.02, 2)

anomaly_ratio = 0.02
with open(results_folder + 'gpt2_regression_anomaly_only.txt', 'a') as results:
    while anomaly_ratio <= 0.20:
        f1, precision, recall = experiment(anomaly_type='random_lines', anomaly_amount=1, mode="regression",
                                           prediction_only=True, anomaly_ratio=anomaly_ratio, alteration_ratio=0.00,
                                           embeddings_model='gpt2', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(anomaly_ratio, f1, precision, recall))
        results.flush()
        anomaly_ratio = round(anomaly_ratio + 0.02, 2)



# XL

anomaly_ratio = 0.02
with open(results_folder + 'xl_multiclass_anomaly_only.txt', 'a') as results:
    while anomaly_ratio <= 0.20:
        f1, precision, recall = experiment(anomaly_type='random_lines', anomaly_amount=1, mode="multiclass",
                                           prediction_only=True, anomaly_ratio=anomaly_ratio, alteration_ratio=0.00,
                                           embeddings_model='xl', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(anomaly_ratio, f1, precision, recall))
        results.flush()
        anomaly_ratio = round(anomaly_ratio + 0.02, 2)

anomaly_ratio = 0.02
with open(results_folder + 'xl_regression_anomaly_only.txt', 'a') as results:
    while anomaly_ratio <= 0.20:
        f1, precision, recall = experiment(anomaly_type='random_lines', anomaly_amount=1, mode="regression",
                                           prediction_only=True, anomaly_ratio=anomaly_ratio, alteration_ratio=0.00,
                                           embeddings_model='xl', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(anomaly_ratio, f1, precision, recall))
        results.flush()
        anomaly_ratio = round(anomaly_ratio + 0.02, 2)