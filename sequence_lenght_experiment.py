from os import makedirs

from normal_learning import experiment

results_folder = 'results_seq_len/'
makedirs(results_folder, exist_ok=True)


with open(results_folder + 'bert_regression_insert_words_results_anomaly_ratio_0.05.txt', 'a') as results:
    sequence_lenght = 8
    while sequence_lenght <= 8:
        f1, precision, recall = experiment(anomaly_type='insert_words', anomaly_amount=1, mode="regression",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.15,
                                   embeddings_model='bert', epochs=60, seq_len=sequence_lenght)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(sequence_lenght, f1, precision, recall))
        results.flush()
        sequence_lenght = sequence_lenght + 1


with open(results_folder + 'bert_multiclass_insert_words_results_anomaly_ratio_0.05.txt', 'a') as results:
    sequence_lenght = 8
    while sequence_lenght <= 8:
        f1, precision, recall = experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.15,
                                   embeddings_model='bert', epochs=60, seq_len=sequence_lenght)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(sequence_lenght, f1, precision, recall))
        results.flush()
        sequence_lenght = sequence_lenght + 1


with open(results_folder + 'gpt2_regression_insert_words_results_anomaly_ratio_0.05.txt', 'w') as results:
    sequence_lenght = 4
    while sequence_lenght <= 8:
        f1, precision, recall = experiment(anomaly_type='insert_words', anomaly_amount=1, mode="regression",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.15,
                                   embeddings_model='gpt2', epochs=60, seq_len=sequence_lenght)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(sequence_lenght, f1, precision, recall))
        results.flush()
        sequence_lenght = sequence_lenght + 1


with open(results_folder + 'gpt2_multiclass_insert_words_results_anomaly_ratio_0.05.txt', 'w') as results:
    sequence_lenght = 4
    while sequence_lenght <= 8:
        f1, precision, recall = experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.15,
                                   embeddings_model='gpt2', epochs=60, seq_len=sequence_lenght)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(sequence_lenght, f1, precision, recall))
        results.flush()
        sequence_lenght = sequence_lenght + 1


with open(results_folder + 'xl_regression_insert_words_results_anomaly_ratio_0.05.txt', 'w') as results:
    sequence_lenght = 4
    while sequence_lenght <= 8:
        f1, precision, recall = experiment(anomaly_type='insert_words', anomaly_amount=1, mode="regression",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.15,
                                   embeddings_model='xl', epochs=60, seq_len=sequence_lenght)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(sequence_lenght, f1, precision, recall))
        results.flush()
        sequence_lenght = sequence_lenght + 1


with open(results_folder + 'xl_multiclass_insert_words_results_anomaly_ratio_0.05.txt', 'w') as results:
    sequence_lenght = 4
    while sequence_lenght <= 8:
        f1, precision, recall = experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.15,
                                   embeddings_model='xl', epochs=60, seq_len=sequence_lenght)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(sequence_lenght, f1, precision, recall))
        results.flush()
        sequence_lenght = sequence_lenght + 1