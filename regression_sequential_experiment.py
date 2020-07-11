from os import makedirs

from normal_learning import experiment

results_folder = 'results_sequential/'
makedirs(results_folder, exist_ok=True)


# experiment(anomaly_type='random_lines', anomaly_amount=1, mode="regression",
#            prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.05,
#            embeddings_model='gpt2', epochs=100)
#
# alteration_ratio = 0.05
# with open(results_folder + 'gpt2_regression_duplicate_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="regression",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='gpt2', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#
# alteration_ratio = 0.05
# with open(results_folder + 'gpt2_regression_delete_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='delete_lines', anomaly_amount=1, mode="regression",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='gpt2', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#
# alteration_ratio = 0.05
# with open(results_folder + 'gpt2_regression_shuffle_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='shuffle', anomaly_amount=1, mode="regression",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='gpt2', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#
#
# with open(results_folder + 'gpt2_regression_reverse_results_anomaly_ratio_0.05.txt', 'w') as results:
#     f1, precision, recall = experiment(anomaly_type='reverse_order', anomaly_amount=1, mode="regression",
#                                prediction_only=True, anomaly_ratio=0.05, alteration_ratio=0,
#                                embeddings_model='gpt2', epochs=100)
#     results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0, f1, precision, recall))
#     results.flush()
#
#
#
#
# #######################
# #######################
# # BERT
# #######################
# #######################
#
# experiment(anomaly_type='random_lines', anomaly_amount=1, mode="regression",
#            prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.05,
#            embeddings_model='bert', epochs=100)
#
# alteration_ratio = 0.05
# with open(results_folder + 'bert_regression_duplicate_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="regression",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='bert', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#
# alteration_ratio = 0.05
# with open(results_folder + 'bert_regression_delete_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='delete_lines', anomaly_amount=1, mode="regression",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='bert', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#
# alteration_ratio = 0.05
# with open(results_folder + 'bert_regression_shuffle_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='shuffle', anomaly_amount=1, mode="regression",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='bert', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#
# with open(results_folder + 'bert_regression_reverse_results_anomaly_ratio_0.05.txt', 'w') as results:
#     f1, precision, recall = experiment(anomaly_type='reverse_order', anomaly_amount=1, mode="regression",
#                                prediction_only=True, anomaly_ratio=0.05, alteration_ratio=0,
#                                embeddings_model='bert', epochs=100)
#     results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0, f1, precision, recall))
#     results.flush()



#######################
#######################
# XL
#######################
#######################


experiment(anomaly_type='random_lines', anomaly_amount=1, mode="regression",
           prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.05,
           embeddings_model='xl', epochs=100)

alteration_ratio = 0.05
with open(results_folder + 'xl_regression_duplicate_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="regression",
                                           prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                           embeddings_model='xl', epochs=100)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)

alteration_ratio = 0.05
with open(results_folder + 'xl_regression_delete_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='delete_lines', anomaly_amount=1, mode="regression",
                                           prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                           embeddings_model='xl', epochs=100)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)

alteration_ratio = 0.05
with open(results_folder + 'xl_regression_shuffle_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='shuffle', anomaly_amount=1, mode="regression",
                                           prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                           embeddings_model='xl', epochs=100)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)

with open(results_folder + 'xl_regression_reverse_results_anomaly_ratio_0.05.txt', 'w') as results:
    f1, precision, recall = experiment(anomaly_type='reverse_order', anomaly_amount=1, mode="regression",
                                       prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                       embeddings_model='xl', epochs=100)
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
    results.flush()