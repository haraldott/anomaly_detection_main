import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, vocab_size, n_hidden_units, n_layers, dropout=0.5, tie_weights=False, embedding_dim=30):
        super(LSTM, self).__init__()
        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers
        # enhancement: check if this should be done using nn.Embedding()

        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=n_hidden_units, num_layers=n_layers,
                            dropout=dropout)
        self.decoder = nn.Linear(n_hidden_units, vocab_size)

        # enhancement: read paper and see if this is useful: "Using the Output Embedding to Improve Language Models"
        #   if so, turn tie_weights on in constructor
        if tie_weights:
            if embedding_dim != n_hidden_units:
                raise ValueError('When using the tied flag, n_hidden_units must be equal to embedding_dim')
            self.decoder.weight = self.word_embeddings.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # enhancement: if nn.Embeddings() layer is used, turn this on
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, words, hidden):
        emb = self.word_embeddings(words)
        # emb_drop = self.dropout(emb)
        emb_drop_view = emb.view(len(words), 1, -1)
        output, hidden = self.lstm(emb_drop_view, hidden)
        # output = self.dropout(output)
        decoded = self.decoder(output.view(len(words), -1))
        decoded_scores = F.log_softmax(decoded, dim=1)
        return decoded_scores, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, bsz, self.n_hidden_units),
                weight.new_zeros(self.n_layers, bsz, self.n_hidden_units))
