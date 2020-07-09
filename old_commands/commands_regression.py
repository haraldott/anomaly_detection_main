from normal_learning import experiment

experiment(epochs=100,
           mode="regression",
           anomaly_type='random_lines',
           anomaly_amount=1,
           clip=1.0,
           prediction_only=False,
           experiment="regression")
experiment(anomaly_type='insert_words', anomaly_amount=1, experiment="regression", prediction_only=True)
experiment(anomaly_type='insert_words', anomaly_amount=2, experiment="regression", prediction_only=True)
experiment(anomaly_type='insert_words', anomaly_amount=3, experiment="regression", prediction_only=True)
experiment(anomaly_type='insert_words', anomaly_amount=4, experiment="regression", prediction_only=True)
experiment(anomaly_type='remove_words', anomaly_amount=1, experiment="regression", prediction_only=True)
experiment(anomaly_type='remove_words', anomaly_amount=2, experiment="regression", prediction_only=True)
experiment(anomaly_type='replace_words', anomaly_amount=1, experiment="regression", prediction_only=True)
experiment(anomaly_type='replace_words', anomaly_amount=2, experiment="regression", prediction_only=True)
experiment(anomaly_type='duplicate_lines', anomaly_amount=1, experiment="regression", prediction_only=True)
experiment(anomaly_type='duplicate_lines', anomaly_amount=2, experiment="regression", prediction_only=True)
experiment(anomaly_type='duplicate_lines', anomaly_amount=3, experiment="regression", prediction_only=True)
experiment(anomaly_type='delete_lines', anomaly_amount=1, experiment="regression", prediction_only=True)
experiment(anomaly_type='delete_lines', anomaly_amount=2, experiment="regression", prediction_only=True)
experiment(anomaly_type='shuffle', anomaly_amount=1, experiment="regression", prediction_only=True)
experiment(anomaly_type='no_anomaly', anomaly_amount=1, experiment="regression", prediction_only=True)
experiment(anomaly_type='reverse_order', anomaly_amount=1, experiment="regression", prediction_only=True)

