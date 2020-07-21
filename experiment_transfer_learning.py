from os import makedirs

from transfer_learning import experiment
results_folder = 'results_transfer/'
makedirs(results_folder, exist_ok=True)

# Bert multiclass
alteration_ratio = 0.1

# Bert regression
with open(results_folder + 'bert_classification_reverse.txt', 'w') as results:
    while alteration_ratio <= 0.1:
        f1, precision, recall = experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="multiclass",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='bert', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)
alteration_ratio = 0.1
with open(results_folder + 'bert_regression_transfer_reverse.txt', 'w') as results:
    while alteration_ratio <= 0.1:
        f1, precision, recall = experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="regression",
                                           prediction_only=False, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                           embeddings_model='bert', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)

# # GPT multiclass
alteration_ratio = 0.1
with open(results_folder + 'gpt2_classification_transfer_reverse.txt', 'w') as results:
    while alteration_ratio <= 0.1:
        f1, precision, recall = experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="multiclass",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='gpt2', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)

# GPT regression
alteration_ratio = 0.1
with open(results_folder + 'gpt2_regression_transfer_reverse.txt', 'w') as results:
    while alteration_ratio <= 0.1:
        f1, precision, recall = experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="regression",
                                           prediction_only=False, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                           embeddings_model='gpt2', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)



# XL regression
alteration_ratio = 0.1
with open(results_folder + 'xl_regression_transfer_reverse.txt', 'w') as results:
    while alteration_ratio <= 0.1:
        f1, precision, recall = experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="regression",
                                           prediction_only=False, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                           embeddings_model='xl', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)

# XL multiclass
alteration_ratio = 0.1
with open(results_folder + 'xl_multiclass_transfer_reverse.txt', 'w') as results:
    while alteration_ratio <= 0.1:
        f1, precision, recall = experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="multiclass",
                                           prediction_only=False, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                           embeddings_model='xl', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)