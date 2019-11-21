import argparse
import math
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import loganaliser.model as lstm_model

# Parse args input
parser = argparse.ArgumentParser()
parser.add_argument('-loadglove', type=str,
                    default='../data/openstack/utah/embeddings/glove_18k_anomalies_no_norm_and_padding.model')
parser.add_argument('-loadvectors', type=str,
                    default='../data/openstack/utah/embeddings/glove_18k_anomalies_no_norm_and_padding.pickle')
parser.add_argument('-loadautoencodermodel', type=str, default='137k_normal_autoencoder_with_128_size.pth')
parser.add_argument('-savemodel', type=str, default='saved_models/lstm.pth')
parser.add_argument('-n_layers', type=int, default=4, help='number of layers')
parser.add_argument('-n_hidden_units', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('-seq_length', type=int, default=1)
parser.add_argument('-num_epochs', type=int, default=100)
parser.add_argument('-learning_rate', type=float, default=1e-7)
parser.add_argument('-batch_size', type=int, default=20)
parser.add_argument('-folds', type=int, default=1)
parser.add_argument('-clip', type=float, default=0.25)
parser.add_argument('-fold_active', type=bool, default=False)
args = parser.parse_args()
lr = args.learning_rate
eval_batch_size = 10


def get_batch(source, i):
    data = source[i]
    data = data.view(1, longest_sent, embeddings_dim)
    target = source[i + 1]
    return data, target


def split(x, n_splits):
    n_samples = len(x)
    k_fold_size = n_samples // n_splits
    indices = np.arange(n_samples)

    margin = 0
    for i in range(n_splits):
        start = i * k_fold_size
        stop = start + k_fold_size
        mid = int(0.8 * (stop - start)) + start
        yield indices[start: mid], indices[mid + margin: stop]


# load vectors and glove obj
embeddings = pickle.load(open(args.loadvectors, 'rb'))
glove = pickle.load(open(args.loadglove, 'rb'))

# padding
embeddings_dim = embeddings[0][0].shape[0]  # dimension of each of the word embeddings vectors
sentence_lens = [len(sentence) for sentence in embeddings]  # how many words a log line consists of, without padding
longest_sent = max(sentence_lens)
total_batch_size = len(embeddings)
pad_vector = np.zeros(embeddings_dim)
padded_emb = np.ones((total_batch_size, longest_sent, embeddings_dim)) * pad_vector

for i, x_len in enumerate(sentence_lens):
    sequence = embeddings[i]
    padded_emb[i, 0:x_len] = sequence[:x_len]

# normalise between 0 and 1
# p_max, p_min = padded_emb.max(), padded_emb.min()
# padded_embeddings = (padded_emb - p_min) / (p_max - p_min)

dict_size = len(glove.dictionary)  # number of different words
# embeddings_dim = padded_embeddings[0][0].shape[0]  # dimension of each of the word embeddings vectors

# load the AutoEncoder model

# use the loaded AutoEncoder model, to receive the latent space representation of the padded embeddings


n_patterns = len(padded_emb)
data_x = torch.Tensor(padded_emb)


# samples, timesteps, features
# batch_size, seq_len, embedding_dim
# dataloader_x = DataLoader(data_x, batch_size=64)
# dataloader_y = DataLoader(data_y, batch_size=64)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate():
    model.eval()
    total_loss = 0
    min_loss = None
    max_loss = None
    hidden = model.init_hidden(args.seq_length)  # TODO: check stuff with batch size
    with torch.no_grad():
        for i in range(0, n_patterns - 1):
            data, target = get_batch(data_x, i)
            prediction, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            loss = distance(prediction.view(-1), target.view(-1))
            total_loss += loss.item()
            if not min_loss or loss.item() < min_loss:
                min_loss = loss.item()
            if not max_loss or loss.item() > min_loss:
                max_loss = loss.item()
    return total_loss / n_patterns, min_loss, max_loss  # TODO: check total_loss / len(indices) is this correct?


def train():
    model.train()
    hidden = model.init_hidden(args.seq_length)  # TODO: check stuff with batch size
    for i in range(0, n_patterns - 1):
        data, target = get_batch(data_x, i)
        optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        prediction, hidden = model(data, hidden, sentence_lens[i])

        loss = distance(prediction.view(-1), target.view(-1))
        loss.backward()
        optimizer.step()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)


model = lstm_model.LSTM(embeddings_dim, args.n_hidden_units, args.n_layers)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#  check if softmax is applied on loss, if not do it yourself
#  überprüfe was mse genau macht, abspeichern
#  zb jede 10. epoche die distanz plotten
#  quadrat mean squared error mal probieren
distance = nn.MSELoss()

best_val_loss = None
min_loss = None
max_loss = None
val_loss = 0  # TODO: make sure that this is correct, can we start at 0 ?

try:
    loss_values = []
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        #  with blocking time series split: k-fold cross validation
        if args.fold_active:
            indices_generator = split(data_x, args.folds)
            # for i in range(0, args.folds):
            #     indices = next(indices_generator)
            #     train_incides = indices[0]
            #     eval_indices = indices[1]
            #     train(train_incides)
            #     this_loss, this_min_loss, this_max_loss = evaluate(eval_indices)
            #     val_loss += this_loss
            #     if not min_loss or this_min_loss < min_loss:
            #         min_loss = this_min_loss
            #     if not max_loss or this_max_loss > max_loss:
            #         max_loss = this_max_loss
        #  without cross validation
        else:
            train()
            this_loss, this_min_loss, this_max_loss = evaluate()
            val_loss += this_loss
            if not min_loss or this_min_loss < min_loss:
                min_loss = this_min_loss
            if not max_loss or this_max_loss > max_loss:
                max_loss = this_max_loss
        print('-' * 89)
        print('LSTM: | end of epoch {:3d} | time: {:5.2f}s | valid loss {} | '
              'valid ppl {}'.format(epoch, (time.time() - epoch_start_time),
                                    val_loss, math.exp(val_loss)))
        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(model.state_dict(), args.savemodel)
            best_val_loss = val_loss
        else:
            # anneal learning rate
            lr /= 2.0
            print("anneal lr to: {}".format(lr))
        loss_values.append(val_loss / args.folds)
    plt.plot(loss_values)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
