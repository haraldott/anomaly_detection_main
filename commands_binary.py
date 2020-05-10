from normal_learning import experiment

experiment(mode="binary", anomaly_type='random_lines', anomaly_amount=1, experiment="binary_clip:0.9", clip=0.9)
experiment(mode="binary", anomaly_type='random_lines', anomaly_amount=1, experiment="binary_clip:0.95", clip=0.95)
experiment(mode="binary", anomaly_type='random_lines', anomaly_amount=1, experiment="binary_clip:1.0", clip=1.0)
experiment(mode="binary", anomaly_type='random_lines', anomaly_amount=1, experiment="binary_clip:1.05", clip=1.05)
experiment(mode="binary", anomaly_type='random_lines', anomaly_amount=1, experiment="binary_clip:1.1", clip=1.1)
experiment(mode="binary", anomaly_type='random_lines', anomaly_amount=1, experiment="binary_clip:1.15", clip=1.15)