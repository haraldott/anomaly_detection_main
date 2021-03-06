from normal_learning import experiment

# hyperparameters
seq_len= [7, 8, 9, 10]
n_layers = [1, 2, 3]
n_hidden_units= [128, 256, 512]
batch_size = [64, 128]
clip = [0.9, 0.95, 1.0, 1.05, 1.1]
epochs = [80, 90, 100, 110, 120, 130]

total_len_of_experiments = len(seq_len) * len(n_layers) * len(n_hidden_units) * len(batch_size) * len(clip) * len(epochs)


score_file = open("score_file.txt", 'w')

experiment_index = 0
score_file.write("F1-Score,Precision,seq_len,n_layers,n_hidden_units,batch_size,clip,epochs\n")
for len in seq_len:
    for layer in n_layers:
        for hidden in n_hidden_units:
            for batch in batch_size:
                for cl in clip:
                    for epoch in epochs:
                        print("seq_len: {}, n_layers: {}, n_hidden_units: {}, batch_size: {}, clip: {}, epochs: {}\n"
                              .format(len, layer, hidden, batch, cl, epoch))
                        experiment_index += 1
                        print("Experiment {} of {}".format(experiment_index, total_len_of_experiments))
                        f1_score, precision = experiment(seq_len=len, n_layers=layer, n_hidden_units=hidden, batch_size=batch,
                                              clip=cl, epochs=epoch)
                        score_file.write("{},{},{},{},{},{},{},{}\n"
                                         .format(f1_score, precision, len, layer, hidden, batch, cl, epoch))
                        score_file.flush()
score_file.close()

# continue here: seq_len: 7, n_layers: 2, n_hidden_units: 256, batch_size: 64, clip: 0.9, epochs: 80