from normal_learning import experiment

experiment(epochs=100, mode="multiclass", anomaly_type='random_lines', anomaly_amount=1, prediction_only=False, finetune_epochs=1, finetuning=True, experiment="finetuning_1_epoch")
experiment(epochs=100, mode="multiclass", anomaly_type='random_lines', anomaly_amount=1, prediction_only=False, finetune_epochs=2, finetuning=True, experiment="finetuning_2_epoch")
experiment(epochs=100, mode="multiclass", anomaly_type='random_lines', anomaly_amount=1, prediction_only=False, finetune_epochs=3, finetuning=True, experiment="finetuning_3_epoch")
experiment(epochs=100, mode="multiclass", anomaly_type='random_lines', anomaly_amount=1, prediction_only=False, finetune_epochs=4, finetuning=True, experiment="finetuning_4_epoch")
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