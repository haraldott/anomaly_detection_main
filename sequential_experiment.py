from os import makedirs

from normal_learning import experiment
results_folder = 'results_sequential/'
makedirs(results_folder, exist_ok=True)

alteration_ratio = 0.05
with open(results_folder + 'bert_multiclass_duplicate_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.2:
        f1, precision = experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="multiclass",
                                   prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='bert')
        results.write("{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision))
        results.flush()
        alteration_ratio += 0.05

alteration_ratio = 0.05
with open(results_folder + 'bert_multiclass_delete_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.2:
        f1, precision = experiment(anomaly_type='delete_lines', anomaly_amount=1, mode="multiclass",
                                   prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='bert')
        results.write("{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision))
        results.flush()
        alteration_ratio += 0.05

alteration_ratio = 0.05
with open(results_folder + 'bert_multiclass_shuffle_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.2:
        f1, precision = experiment(anomaly_type='shuffle', anomaly_amount=1, mode="multiclass",
                                   prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='bert')
        results.write("{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision))
        results.flush()
        alteration_ratio += 0.05


alteration_ratio = 0.05
with open(results_folder + 'bert_regression_duplicate_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.2:
        f1, precision = experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="regression",
                                   prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='bert')
        results.write("{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision))
        results.flush()
        alteration_ratio += 0.05

alteration_ratio = 0.05
with open(results_folder + 'bert_regression_delete_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.2:
        f1, precision = experiment(anomaly_type='delete_lines', anomaly_amount=1, mode="regression",
                                   prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='bert')
        results.write("{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision))
        results.flush()
        alteration_ratio += 0.05

alteration_ratio = 0.05
with open(results_folder + 'bert_regression_shuffle_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.2:
        f1, precision = experiment(anomaly_type='shuffle', anomaly_amount=1, mode="regression",
                                   prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='bert')
        results.write("{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision))
        results.flush()
        alteration_ratio += 0.05