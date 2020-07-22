from normal_learning import experiment

experiment(anomaly_type='insert_words', anomaly_amount=1, mode="regression",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.15,
                                   embeddings_model='bert', epochs=60)

experiment(anomaly_type='insert_words', anomaly_amount=1, mode="regression",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.15,
                                   embeddings_model='gpt2', epochs=60)

experiment(anomaly_type='insert_words', anomaly_amount=1, mode="regression",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.15,
                                   embeddings_model='xl', epochs=60)



experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.15,
                                   embeddings_model='bert', epochs=60)

experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.15,
                                   embeddings_model='gpt2', epochs=60)

experiment(anomaly_type='insert_words', anomaly_amount=1, mode="multiclass",
                                   prediction_only=False, anomaly_ratio=0.05, alteration_ratio=0.15,
                                   embeddings_model='xl', epochs=60)