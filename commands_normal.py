from normal_learning import experiment

# hyperparameters
seq_len= [7, 8, 9, 10]
n_layers = [1, 2, 3]
n_hidden_units= [128, 256, 512, 768]
batch_size = [64, 128]
clip = [0.9, 0.95, 1.0, 1.05, 1.1]
epochs = [80, 90, 100, 110, 120, 130]

score_file = open("score_file.txt", 'w')

for len in seq_len:
    for layer in n_layers:
        for hidden in n_hidden_units:
            for batch in batch_size:
                for cl in clip:
                    for epoch in epochs:
                        f1_score, precision = experiment(seq_len=len, n_layers=layer, n_hidden_units=hidden, batch_size=batch,
                                              clip=cl, epochs=epoch)
                        score_file.write("F1-Score: {}, Precision: {}, ----- "
                                         "seq_len: {}, n_layers: {}, n_hidden_units: {}, batch_size: {}, clip: {}, epochs: {}\n"
                                         .format(f1_score, precision, len, layer, hidden, batch, cl, epoch))