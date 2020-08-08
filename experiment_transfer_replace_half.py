from os import makedirs

from transfer_learning import experiment

results_folder = 'results_replace_half_transfer/'
makedirs(results_folder, exist_ok=True)

# BERT

with open(results_folder + 'bert_multiclass_replace_half.txt', 'w') as results:
    f1, precision, recall = experiment(anomaly_type='replace_half', anomaly_amount=1, mode="multiclass",
                                       prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.00,
                                       embeddings_model='bert', epochs=60, experiment='transfer_replace_half')
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0, f1, precision, recall))
    results.flush()


with open(results_folder + 'bert_regression_replace_half.txt', 'w') as results:
    f1, precision, recall = experiment(anomaly_type='replace_half', anomaly_amount=1, mode="regression",
                                       prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.00,
                                       embeddings_model='bert', epochs=60, experiment='transfer_replace_half')
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0, f1, precision, recall))
    results.flush()

# GPT

with open(results_folder + 'gpt2_multiclass_replace_half.txt', 'w') as results:
    f1, precision, recall = experiment(anomaly_type='replace_half', anomaly_amount=1, mode="multiclass",
                                       prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.00,
                                       embeddings_model='gpt2', epochs=60, experiment='transfer_replace_half')
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0, f1, precision, recall))
    results.flush()


with open(results_folder + 'gpt2_regression_replace_half.txt', 'w') as results:
    f1, precision, recall = experiment(anomaly_type='replace_half', anomaly_amount=1, mode="regression",
                                       prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.00,
                                       embeddings_model='gpt2', epochs=60, experiment='transfer_replace_half')
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0, f1, precision, recall))
    results.flush()


# XL

with open(results_folder + 'xl_multiclass_replace_half.txt', 'w') as results:
    f1, precision, recall = experiment(anomaly_type='replace_half', anomaly_amount=1, mode="multiclass",
                                       prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.00,
                                       embeddings_model='xl', epochs=60, experiment='transfer_replace_half')
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0, f1, precision, recall))
    results.flush()


with open(results_folder + 'xl_regression_replace_half.txt', 'w') as results:
    f1, precision, recall = experiment(anomaly_type='replace_half', anomaly_amount=1, mode="regression",
                                       prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.00,
                                       embeddings_model='xl', epochs=60, experiment='transfer_replace_half')
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0, f1, precision, recall))
    results.flush()