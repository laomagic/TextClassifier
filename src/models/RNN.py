import torch
import torch.nn as nn


class RNNConfig:
    hidden_size = 256
    vocab_size = 50000
    requires_grad = True
    pretrain_embeddings = False
    pretrain_embeddings_path = None
    drop_out = 0.3
    num_class = 10
    num_layers = 2
    bidirectional = True
    embed_size = 300


class RNNClassifier(nn.Module):
    def __init__(self):
        super(RNNClassifier, self).__init__()
        if RNNConfig.pretrain_embeddings:
            self.embeddings = nn.Embedding.from_pretrained(RNNConfig.pretrain_embeddings_path)
        else:
            self.embeddings = nn.Embedding(RNNConfig.vocab_size, RNNConfig.embed_size)
#         self.lstm = nn.LSTM(input_size=RNNConfig.embed_size,
#                             hidden_size=RNNConfig.hidden_size,
#                             batch_first=True,
#                             num_layers=RNNConfig.num_layers,
#                             bidirectional=RNNConfig.bidirectional)
        self.gru = nn.GRU(input_size=RNNConfig.embed_size,
                          hidden_size=RNNConfig.hidden_size,
                          batch_first=True,
                          num_layers=RNNConfig.num_layers,
                          bidirectional=RNNConfig.bidirectional)
#         self.rnn = nn.RNN(input_size=RNNConfig.embed_size,
#                           hidden_size=RNNConfig.hidden_size,
#                           batch_first=True,
#                           num_layers=RNNConfig.num_layers,
#                           bidirectional=RNNConfig.bidirectional)
        self.drop_out = nn.Dropout(RNNConfig.drop_out)
        self.fc = nn.Linear(2*RNNConfig.hidden_size, RNNConfig.num_class)

    def forward(self, x):
        embedding = self.embeddings(x)  # [batch_size,seq_len,embed_size]
#         hidden, _ = self.lstm(embedding)  # [batch_size,seq_len,hidden_size]
        hidden, _ = self.gru(embedding)  # [batch_size,seq_len,2*hidden_size]

        hidden = self.drop_out(hidden)
        hidden = hidden[:, -1, :]  # 获取最后一层的输出
        prob = self.fc(hidden)
        return prob
