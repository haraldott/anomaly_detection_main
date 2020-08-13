from os import makedirs

from normal_learning import experiment

results_folder = 'results_replace_half/'
makedirs(results_folder, exist_ok=True)

# BERT

with open(results_folder + 'bert_multiclass_anomaly_only.txt', 'a') as results:
    f1, precision, recall = experiment(anomaly_type='random_lines', anomaly_amount=1, mode="multiclass",
                               prediction_only=True, anomaly_ratio=0.05, alteration_ratio=0.05,
                               embeddings_model='bert', epochs=60)
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0.05, f1, precision, recall))
    results.flush()


with open(results_folder + 'bert_regression_anomaly_only.txt', 'a') as results:
    f1, precision, recall = experiment(anomaly_type='random_lines', anomaly_amount=1, mode="regression",
                               prediction_only=True, anomaly_ratio=0.05, alteration_ratio=0.05,
                               embeddings_model='bert', epochs=60)
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0.05, f1, precision, recall))
    results.flush()


# GPT

with open(results_folder + 'gpt2_multiclass_anomaly_only.txt', 'a') as results:
    f1, precision, recall = experiment(anomaly_type='random_lines', anomaly_amount=1, mode="multiclass",
                                       prediction_only=True, anomaly_ratio=0.05, alteration_ratio=0.05,
                                       embeddings_model='gpt2', epochs=60)
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0.05, f1, precision, recall))
    results.flush()

with open(results_folder + 'gpt2_regression_anomaly_only.txt', 'a') as results:
    f1, precision, recall = experiment(anomaly_type='random_lines', anomaly_amount=1, mode="regression",
                                       prediction_only=True, anomaly_ratio=0.05, alteration_ratio=0.05,
                                       embeddings_model='gpt2', epochs=60)
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0.05, f1, precision, recall))
    results.flush()



# XL

with open(results_folder + 'xl_multiclass_anomaly_only.txt', 'a') as results:
    f1, precision, recall = experiment(anomaly_type='random_lines', anomaly_amount=1, mode="multiclass",
                                       prediction_only=True, anomaly_ratio=0.05, alteration_ratio=0.05,
                                       embeddings_model='xl', epochs=60)
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0.05, f1, precision, recall))
    results.flush()

with open(results_folder + 'xl_regression_anomaly_only.txt', 'a') as results:
    f1, precision, recall = experiment(anomaly_type='random_lines', anomaly_amount=1, mode="regression",
                                       prediction_only=True, anomaly_ratio=0.05, alteration_ratio=0.05,
                                       embeddings_model='xl', epochs=60)
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0.05, f1, precision, recall))
    results.flush()