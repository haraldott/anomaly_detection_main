from normal_learning import experiment

alteration_ratio = 0.05
with open('duplicate_lines_results_anomaly_ratio_0.04.txt', 'w') as results:
    while alteration_ratio <= 0.3:
        f1, precision = experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="multiclass", prediction_only=True, anomaly_ratio=0.04, alteration_ratio=alteration_ratio)
        results.write("{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision))
        alteration_ratio += 0.05

alteration_ratio = 0.05
with open('delete_lines_results_anomaly_ratio_0.04.txt', 'w') as results:
    while alteration_ratio <= 0.3:
        f1, precision = experiment(anomaly_type='delete_lines', anomaly_amount=1, mode="multiclass", prediction_only=True, anomaly_ratio=0.04, alteration_ratio=alteration_ratio)
        results.write("{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision))
        alteration_ratio += 0.05

alteration_ratio = 0.05
with open('shuffle_results_anomaly_ratio_0.04.txt', 'w') as results:
    while alteration_ratio <= 0.3:
        f1, precision = experiment(anomaly_type='shuffle', anomaly_amount=1, mode="multiclass", prediction_only=True, anomaly_ratio=0.04, alteration_ratio=alteration_ratio)
        results.write("{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision))
        alteration_ratio += 0.05