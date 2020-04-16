import torch.nn as nn
import torch.nn.functional as F
import torch

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.concat_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, hidden_size)

    def forward(self, rnn_outputs, final_hidden_state):
        # rnn_output.shape:         (batch_size, seq_len, hidden_size)
        # final_hidden_state.shape: (batch_size, hidden_size)
        batch_size, seq_len, _ = rnn_outputs.shape
        attn_weights = self.attn(rnn_outputs) # (batch_size, seq_len, hidden_dim)
        attn_weights = torch.bmm(attn_weights, final_hidden_state.unsqueeze(2))

        attn_weights = F.softmax(attn_weights.squeeze(2), dim=1)
        context = torch.bmm(rnn_outputs.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        attn_hidden = torch.tanh(self.concat_linear(torch.cat((context, final_hidden_state), dim=1)))
        return attn_hidden, attn_weights

class LSTM(nn.Module):
    def __init__(self, n_input, n_hidden_units, n_layers, n_classes, batch_size,
                 bidirectional = False, tie_weights=False, train_mode=False):
        super(LSTM, self).__init__()
        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers
        self.train_mode = train_mode
        self.num_directions = 2 if bidirectional == True else 1
        self.batch_size = batch_size
        self.hidden = None
        # Layers

        self.lstm = nn.LSTM(input_size=n_input, hidden_size=self.n_hidden_units, num_layers=self.n_layers,
                            dropout=0.2, batch_first=True, bidirectional=bidirectional)
        # TODO: batch first?!?!
        self.attn = Attention(self.n_hidden_units)
        self.decoder = nn.Linear(n_hidden_units, n_classes)

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

    def forward(self, input):
        if self.train_mode: input = nn.Dropout(p=0.1)(input)

        # go through lstm layer
        output, self.hidden = self.lstm(input, self.hidden)
        # extract last hidden state
        final_state = self.hidden[0].view(self.n_layers, self.num_directions, self.batch_size, self.n_hidden_units)[-1]
        final_hidden_state = None
        if self.num_directions == 1:
            final_hidden_state = final_state.squeeze(0)
        elif self.num_directions == 2:
            h_1, h_2 = final_state[0], final_state[1]
            # final_hidden_state = h_1 + h_2               # Add both states (requires changes to the input size of first linear layer + attention layer)
            final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states

        # go through attn layer
        X, attn_weights = self.attn(output, final_hidden_state)

        decoded = self.decoder(X)

        #if self.train_mode: output = nn.Dropout(p=0.1)(output)
        log_props = F.log_softmax(decoded, dim=1)
        return log_props

    def init_hidden(self, bsz, device):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, bsz, self.n_hidden_units).to(device),
                weight.new_zeros(self.n_layers, bsz, self.n_hidden_units).to(device))




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
