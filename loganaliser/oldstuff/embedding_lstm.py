import torch.nn as nn
import torch
import torch.nn.functional as F

class EmbeddingLSTM(nn.Module):
    def __init__(self, vocab_size, n_input, n_hidden_units, n_layers, embedding_dim, dropout=0.1, tie_weights=True):
        super(EmbeddingLSTM, self).__init__()

        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers

        # Layers
        self.encoder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=n_hidden_units, num_layers=n_layers,
                            dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(n_hidden_units, n_input)

        # enhancement: read paper and see if this is useful: "Using the Output Embedding to Improve Language Models"
        if tie_weights:
            if n_input != n_hidden_units:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # enhancement: if nn.Embeddings() layer is used, turn this on
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        input = self.dropout(self.encoder(input))
        output, hidden = self.lstm(input, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output[:, -1, :])
        return decoded, hidden

    def init_hidden(self, bsz, device):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, bsz, self.n_hidden_units).to(device),
                weight.new_zeros(self.n_layers, bsz, self.n_hidden_units).to(device))