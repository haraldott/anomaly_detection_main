from os import makedirs

from transfer_learning import experiment

results_folder = 'results_replace_half_transfer/'
makedirs(results_folder, exist_ok=True)


# XL regression
alteration_ratio = 0.05
with open(results_folder + 'xl_regression_transfer_replace_half.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='replace_half', anomaly_amount=1, mode="regression",
                                           prediction_only=False, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                           embeddings_model='xl', epochs=60, experiment='transfer_replace_half')
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)


alteration_ratio = 0.05
with open(results_folder + 'bert_regression_transfer_replace_half.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='replace_half', anomaly_amount=1, mode="regression",
                                           prediction_only=False, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                           embeddings_model='bert', epochs=60, experiment='transfer_replace_half')
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)

# GPT regression
alteration_ratio = 0.05
with open(results_folder + 'gpt2_regression_transfer_replace_half.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='replace_half', anomaly_amount=1, mode="regression",
                                           prediction_only=False, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                           embeddings_model='gpt2', epochs=60, experiment='transfer_replace_half')
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)


# Bert multiclass
alteration_ratio = 0.05

with open(results_folder + 'bert_classification_replace_half.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='replace_half', anomaly_amount=1, mode="multiclass",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='bert', epochs=60, experiment='transfer_replace_half')
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)


# # GPT multiclass
alteration_ratio = 0.05
with open(results_folder + 'gpt2_classification_transfer_replace_half.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='replace_half', anomaly_amount=1, mode="multiclass",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='gpt2', epochs=60, experiment='transfer_replace_half')
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)

# XL multiclass
alteration_ratio = 0.05
with open(results_folder + 'xl_multiclass_transfer_replace_half.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='replace_half', anomaly_amount=1, mode="multiclass",
                                           prediction_only=False, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                           embeddings_model='xl', epochs=60, experiment='transfer_replace_half')
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)