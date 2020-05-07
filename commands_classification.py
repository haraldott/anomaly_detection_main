from normal_learning import experiment

experiment(epochs=100,
           mode="multiclass",
           anomaly_type='random_lines',
           anomaly_amount=1,
           clip=1.0,
           anomaly_only=False,
           experiment="multiclass")
experiment(anomaly_type='insert_words', anomaly_amount=1, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='insert_words', anomaly_amount=2, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='insert_words', anomaly_amount=3, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='insert_words', anomaly_amount=4, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='remove_words', anomaly_amount=1, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='remove_words', anomaly_amount=2, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='replace_words', anomaly_amount=1, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='replace_words', anomaly_amount=2, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='duplicate_lines', anomaly_amount=1, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='duplicate_lines', anomaly_amount=2, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='duplicate_lines', anomaly_amount=3, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='delete_lines', anomaly_amount=1, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='delete_lines', anomaly_amount=2, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='shuffle', anomaly_amount=1, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='no_anomaly', anomaly_amount=1, experiment="multiclass", anomaly_only=True)
experiment(anomaly_type='reverse_order', anomaly_amount=1, experiment="multiclass", anomaly_only=True)

