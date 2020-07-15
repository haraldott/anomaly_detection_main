from os import makedirs

from normal_learning import experiment

results_folder = 'results_qualitative/'
makedirs(results_folder, exist_ok=True)

# experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.05,
#                                    embeddings_model='bert', epochs=100)
#
# experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.05,
#                                    embeddings_model='gpt', epochs=100)
#
# experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.05,
#                                    embeddings_model='xl', epochs=100)
#
# experiment(anomaly_type='insert_words', anomaly_amount=1, mode="regression",
#                                    prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.05,
#                                    embeddings_model='bert', epochs=100)
#
# experiment(anomaly_type='insert_words', anomaly_amount=1, mode="regression",
#                                    prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.05,
#                                    embeddings_model='gpt', epochs=100)
#
# experiment(anomaly_type='insert_words', anomaly_amount=1, mode="regression",
#                                    prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.05,
#                                    embeddings_model='xl', epochs=100)


# alteration_ratio = 0.05
# with open(results_folder + 'bert_multiclass_insert_words_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='bert', epochs=60)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
# #
# #
# # ########################
# #
# alteration_ratio = 0.05
# with open(results_folder + 'bert_multiclass_remove_words_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='remove_words', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='bert', epochs=60)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)
#
# ########################
#
# alteration_ratio = 0.05
# with open(results_folder + 'bert_multiclass_replace_words_results_anomaly_ratio_0.05.txt', 'w') as results:
#     while alteration_ratio <= 0.15:
#         f1, precision, recall = experiment(anomaly_type='replace_words', anomaly_amount=1, mode="multiclass",
#                                    prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
#                                    embeddings_model='bert', epochs=60)
#         results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
#         results.flush()
#         alteration_ratio = round(alteration_ratio + 0.05, 2)

######################
######################
#GPT 2
######################
######################

alteration_ratio = 0.05
with open(results_folder + 'gpt2_multiclass_insert_words_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass",
                                   prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='gpt2', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)


########################

alteration_ratio = 0.05
with open(results_folder + 'gpt2_multiclass_remove_words_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='remove_words', anomaly_amount=1, mode="multiclass",
                                   prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='gpt2', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)

########################

alteration_ratio = 0.05
with open(results_folder + 'gpt2_multiclass_replace_words_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='replace_words', anomaly_amount=1, mode="multiclass",
                                   prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='gpt2', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)


#######################
#######################
# XL
#######################
#######################

#
experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.05,
                                   embeddings_model='xl', epochs=60)
alteration_ratio = 0.05
with open(results_folder + 'xl_multiclass_insert_words_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass",
                                   prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='xl', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)


########################

alteration_ratio = 0.05
with open(results_folder + 'xl_multiclass_remove_words_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='remove_words', anomaly_amount=1, mode="multiclass",
                                   prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='xl', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)

########################

alteration_ratio = 0.05
with open(results_folder + 'xl_multiclass_replace_words_results_anomaly_ratio_0.05.txt', 'w') as results:
    while alteration_ratio <= 0.15:
        f1, precision, recall = experiment(anomaly_type='replace_words', anomaly_amount=1, mode="multiclass",
                                   prediction_only=True, anomaly_ratio=0.05, alteration_ratio=alteration_ratio,
                                   embeddings_model='xl', epochs=60)
        results.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(alteration_ratio, f1, precision, recall))
        results.flush()
        alteration_ratio = round(alteration_ratio + 0.05, 2)