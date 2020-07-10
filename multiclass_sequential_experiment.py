from os import makedirs

from normal_learning import experiment
from normal_learning_glove import experiment as glove_experiment

results_folder = 'results_sequential/'
makedirs(results_folder, exist_ok=True)



# experiment(anomaly_type='random_lines', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.05,
#                                    embeddings_model='bert', epochs=100)
#
# alteration_ratio = 0.05
# with open(results_folder + 'bert_multiclass_duplicate_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='bert', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#
# alteration_ratio = 0.05
# with open(results_folder + 'bert_multiclass_delete_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='delete_lines', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='bert', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#
# alteration_ratio = 0.05
# with open(results_folder + 'bert_multiclass_shuffle_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='shuffle', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='bert', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)

with open(results_folder + 'bert_multiclass_reverse_results_anomaly_ratio_0.05.txt', 'w') as results:
    f1, precision, recall = experiment(anomaly_type='reverse_order', anomaly_amount=1, mode="multiclass",
                               prediction_only=True, anomaly_ratio=0.05, alteration_ratio=0.0,
                               embeddings_model='bert', epochs=100)
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0.0, f1, precision, recall))
    results.flush()


#
# #######################
# #######################
# # BERT
# #######################
# #######################
#

# experiment(anomaly_type='random_lines', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.05,
#                                    embeddings_model='gpt2', epochs=100)

# alteration_ratio = 0.05
# with open(results_folder + 'gpt2_multiclass_duplicate_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='gpt2', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#
# alteration_ratio = 0.05
# with open(results_folder + 'gpt2_multiclass_delete_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='delete_lines', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='gpt2', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#
# alteration_ratio = 0.05
# with open(results_folder + 'gpt2_multiclass_shuffle_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='shuffle', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='gpt2', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#

with open(results_folder + 'gpt2_multiclass_reverse_results_anomaly_ratio_0.05.txt', 'w') as results:
    f1, precision, recall = experiment(anomaly_type='reverse_order', anomaly_amount=1, mode="multiclass",
                               prediction_only=True, anomaly_ratio=0.05, alteration_ratio=0.0,
                               embeddings_model='gpt2', epochs=100)
    results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(0.0, f1, precision, recall))
    results.flush()

#######################
#######################
# GLOVE
#######################
#######################

# glove_experiment(anomaly_type='random_lines', anomaly_amount=1, mode="multiclass",
#            prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.05,
#            embeddings_model='glove', epochs=100)
#
# alteration_ratio = 0.05
# with open(results_folder + 'glove_multiclass_duplicate_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = glove_experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='glove', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#
# alteration_ratio = 0.05
# with open(results_folder + 'glove_multiclass_delete_lines_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = glove_experiment(anomaly_type='delete_lines', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='glove', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#
# alteration_ratio = 0.05
# with open(results_folder + 'glove_multiclass_shuffle_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = glove_experiment(anomaly_type='shuffle', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='glove', epochs=100)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#
# with open(results_folder + 'glove_multiclass_reverse_results_anomaly_ratio_0.05.txt', 'w') as results:
#     f1, precision, recall = glove_experiment(anomaly_type='reverse_order', anomaly_amount=1, mode="multiclass",
#                                        prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                        embeddings_model='glove', epochs=100)
#     results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#     results.flush()