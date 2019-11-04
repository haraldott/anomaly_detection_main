import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import wordembeddings.createvectors as cv

num_epochs = 5
batch_size = 128

# Loading and Transforming data
# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
# trainset = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# dataloader = DataLoader(trainset, batch_size=32, shuffle=False, num_workers=4)
# testset = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

embeddings, dict_len = cv.create_word_vectors()
# embeddings = np.load(
#     '/Users/haraldott/Development/thesis/anomaly_detection_main/data/openstack/utah/embeddings/outfile.npy')
# np.reshape(embeddings(len(embeddings), 5,))

#dataloader = DataLoader(embeddings, batch_size=32, shuffle=False, num_workers=4)

input_size = embeddings[0][0].shape


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.LSTM(input_size, input_size, kernel_size=1, stride=1),
            nn.ReLU(True),
            nn.Linear,
            nn.Conv1d(4, 14, kernel_size=5),
            nn.ReLU(True))

        self.word_embedding = nn.Embedding(
            num_embeddings=dict_len + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )

    def forward(self, input, hidden, lengths):
        embeddings = self.encoder(input)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        output, hidden = self.rnn(packed, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)

    # def forward(self, x):
    #     x = self.encoder(x)
    #     x = self.decoder(x)
    #     return x


model = Autoencoder()
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

for epoch in range(num_epochs):
    for vec in embeddings:
        # vec = torch.from_numpy(vec)
        # ===================forward=====================
        output = model(vec)
        loss = distance(output, vec)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data()))
