import argparse
import math
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import loganaliser.model as lstm_model
import pandas as pd
from numpy import array

# Parse args input
parser = argparse.ArgumentParser()
parser.add_argument('-logfile', type=str,
                    default='../data/openstack/utah/parsed/openstack_18k_anomalies_structured.csv')
parser.add_argument('-loadglove', type=str, default='../data/openstack/utah/embeddings/glove_137k_normal.model')
parser.add_argument('-loadvectors', type=str, default='../data/openstack/utah/embeddings/vectors_137k_normal.pickle')
parser.add_argument('-loadautoencodermodel', type=str, default='137k_normal_autoencoder_with_128_size.pth')
parser.add_argument('-n_layers', type=int, default=2, help='number of layers')
parser.add_argument('-n_hidden_units', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('-seq_length', type=int, default=4)
parser.add_argument('-num_epochs', type=int, default=100)
parser.add_argument('-learning_rate', type=float, default=1e-7)
parser.add_argument('-batch_size', type=int, default=20)
parser.add_argument('-folds', type=int, default=1)
parser.add_argument('-clip', type=float, default=0.25)
args = parser.parse_args()
lr = args.learning_rate
eval_batch_size = 10


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


p = pd.read_csv(args.logfile)
sentences = p["EventTemplate"]
new_lines = [line.split(' ') for line in sentences]
flat_words = []
for sentence in new_lines:
    for word in sentence:
        flat_words.append(word)
words_as_integers_lookup = sorted(list(set(flat_words)))
vocab_size = len(words_as_integers_lookup)

data_x = []
data_y = []
n_chars = len(flat_words)
words_as_integers = [words_as_integers_lookup.index(word) for word in flat_words]
for i in range(0, n_chars - args.seq_length - 1):
    data_x.append(words_as_integers[i: i + args.seq_length])
    data_y.append(words_as_integers[i + args.seq_length + 1])
data_x = array(data_x)
data_y = array(data_y)

# data_x = torch.Tensor(data_x)
# data_y = torch.Tensor(data_y)


# samples, timesteps, features
# dataloader_x = DataLoader(data_x, batch_size=64)
# dataloader_y = DataLoader(data_y, batch_size=64)

# data_x = np.reshape(data_x, (data_x.shape[0], args.seq_length, 1))

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(idx):
    model.eval()
    dataloader_x = DataLoader(data_x[idx], batch_size=args.batch_size)
    dataloader_y = DataLoader(data_y[idx], batch_size=args.batch_size)
    total_loss = 0
    min_loss = None
    max_loss = None
    hidden = model.init_hidden(args.seq_length)  # TODO: check stuff with batch size
    with torch.no_grad():
        for data, target in zip(dataloader_x, dataloader_y):
            prediction, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            loss = distance(prediction, target)
            total_loss += loss.item()
            if not min_loss or loss.item() < min_loss:
                min_loss = loss.item()
            if not max_loss or loss.item() > min_loss:
                max_loss = loss.item()
    return total_loss / len(indices), min_loss, max_loss  # TODO: check total_loss / len(indices) is this correct?


def train(idx):
    model.train()
    dataloader_x = DataLoader(data_x[idx], batch_size=args.batch_size)
    dataloader_y = DataLoader(data_y[idx], batch_size=args.batch_size)
    hidden = model.init_hidden(args.seq_length)  # TODO: check stuff with batch size
    for data, target in zip(dataloader_x, dataloader_y):
        optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        prediction, hidden = model(data, hidden)

        loss = distance(prediction, target)
        loss.backward()
        optimizer.step()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)


model = lstm_model.LSTM(vocab_size, args.n_hidden_units, args.n_layers)
# enhancement: check out Adabound: https://github.com/Luolc/AdaBound
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

#  check if softmax is applied on loss, if not do it yourself
#  überprüfe was mse genau macht, abspeichern
#  zb jede 10. epoche die distanz plotten
#  quadrat mean squared error mal probieren
distance = nn.CrossEntropyLoss()

best_val_loss = None
min_loss = None
max_loss = None
val_loss = 0  # TODO: make sure that this is correct, can we start at 0 ?

#y = torch.LongTensor(args.batch_size, 1).random_() %

try:
    loss_values = []
    for epoch in range(args.num_epochs):
        indices_generator = split(data_x, args.folds)
        epoch_start_time = time.time()
        for i in range(0, args.folds):
            indices = next(indices_generator)
            train_incides = indices[0]
            eval_indices = indices[1]

            train(train_incides)
            this_loss, this_min_loss, this_max_loss = evaluate(eval_indices)
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
            torch.save(model.state_dict(), './lstm.pth')
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
