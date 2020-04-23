from normal_learning import experiment

# hyperparameters
seq_len= [7]
n_layers = [1]
n_hidden_units= [128]
batch_size = [64]
clip = [0.9]
epochs = [80]

score_file = open("score_file.txt", 'w')

score_file.write("F1-Score,Precision,seq_len,n_layers,n_hidden_units,batch_size,clip,epochs")
for len in seq_len:
    for layer in n_layers:
        for hidden in n_hidden_units:
            for batch in batch_size:
                for cl in clip:
                    for epoch in epochs:
                        print("seq_len: {}, n_layers: {}, n_hidden_units: {}, batch_size: {}, clip: {}, epochs: {}\n"
                              .format(len, layer, hidden, batch, cl, epoch))
                        f1_score, precision = experiment(seq_len=len, n_layers=layer, n_hidden_units=hidden, batch_size=batch,
                                              clip=cl, epochs=epoch)
                        score_file.write("{},{},{},{},{},{},{},{}\n"
                                         .format(f1_score, precision, len, layer, hidden, batch, cl, epoch))
score_file.close()

# continue here: seq_len: 7, n_layers: 2, n_hidden_units: 256, batch_size: 64, clip: 0.9, epochs: 80