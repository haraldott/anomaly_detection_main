from normal_learning import experiment

# experiment_results_f1 = {}
# i = 0.02
# experiment_results_f1[i], _ = experiment(prediction_only=False, anomaly_type='random_lines', anomaly_amount=1, mode="multiclass", anomaly_ratio=i, alteration_ratio=0.05)
# while i <= 0.08:
#     i += 0.01
#     experiment_results_f1[i], _ = experiment(anomaly_type='random_lines', anomaly_amount=1, mode="multiclass", prediction_only=True, anomaly_ratio=i, alteration_ratio=0.05)
#
# with open('anomaly_only_results.txt', 'w') as results:
#     for ratio, f1 in experiment_results_f1.items():
#         results.write("{:.2f},{:.2f}\n".format(ratio, f1))

alteration_ratio = 0.05
insert_words_experiment = {}
while alteration_ratio <= 0.4:
    experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass", prediction_only=True, anomaly_ratio=0.02, alteration_ratio=0.05)
    alteration_ratio += 0.05

with open('insert_words_results_anomaly_ratio_0.02.txt', 'w') as results:
    for ratio, f1 in insert_words_experiment.items():
        results.write("{:.2f},{:.2f}\n".format(ratio, f1))

#
#
# experiment(epochs=100, mode="multiclass", anomaly_type='random_lines', anomaly_amount=1, prediction_only=False, finetune_epochs=1, finetuning=True, experiment="finetuning_1_epoch")
# experiment(epochs=100, mode="multiclass", anomaly_type='random_lines', anomaly_amount=1, prediction_only=False, finetune_epochs=2, finetuning=True, experiment="finetuning_2_epoch")
# experiment(epochs=100, mode="multiclass", anomaly_type='random_lines', anomaly_amount=1, prediction_only=False, finetune_epochs=3, finetuning=True, experiment="finetuning_3_epoch")
# experiment(epochs=100, mode="multiclass", anomaly_type='random_lines', anomaly_amount=1, prediction_only=False, finetune_epochs=4, finetuning=True, experiment="finetuning_4_epoch")
#
#
# experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='insert_words', anomaly_amount=2, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='insert_words', anomaly_amount=3, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='insert_words', anomaly_amount=4, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='remove_words', anomaly_amount=1, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='remove_words', anomaly_amount=2, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='replace_words', anomaly_amount=1, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='replace_words', anomaly_amount=2, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='duplicate_lines', anomaly_amount=1, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='duplicate_lines', anomaly_amount=2, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='duplicate_lines', anomaly_amount=3, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='delete_lines', anomaly_amount=1, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='delete_lines', anomaly_amount=2, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='shuffle', anomaly_amount=1, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='no_anomaly', anomaly_amount=1, mode="multiclass", prediction_only=True)
# experiment(anomaly_type='reverse_order', anomaly_amount=1, mode="multiclass", prediction_only=True)