import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, n_input, n_hidden_units, n_layers, dropout=0.5, tie_weights=False):
        super(LSTM, self).__init__()
        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers
        # enhancement: check if this should be done using nn.Embedding()
        # self.encoder = nn.Embedding()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=n_hidden_units, num_layers=n_layers,
                            dropout=dropout)
        self.decoder = nn.Linear(n_hidden_units, n_input)

        # enhancement: read paper and see if this is useful: "Using the Output Embedding to Improve Language Models"
        #   if so, turn tie_weights on in constructor
        if tie_weights:
            if n_input != n_hidden_units:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # enhancement: if nn.Embeddings() layer is used, turn this on
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        input = self.dropout(input)
        output, hidden = self.lstm(input, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output)
        return decoded, hidden

    # def loss(self, y_pred, y):
    #     # y = y.view(-1)
    #     # y_pred = y_pred.view(-1)
    #
    #     mask_y = y * (y != np.zeros(embeddings_dim))
    #     mask_y_pred = y_pred * (y_pred != np.zeros(embeddings_dim))
    #     distance(mask_y, mask_y_pred)
    #
    #     return distance

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, bsz, self.n_hidden_units),
                weight.new_zeros(self.n_layers, bsz, self.n_hidden_units))
