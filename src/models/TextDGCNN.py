# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextDGCNN(nn.Module):
    def __init__(self, config):
        super(TextDGCNN, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=0)
            # self.embedding.weight.requires_grad=True
        self.dropout = nn.Dropout(config.dropout)
        self.dgcnn = nn.ModuleList(
            DGCNNLayer(config.embed, config.embed, k_size=param[0], dilation_rate=param[1]) for param in
            config.dgccn_params)
        self.fc = nn.Linear(config.embed, config.num_classes)
        self.position_embedding = nn.Embedding(512, config.embed)

    def forward(self, x):
        word_emb = self.embedding(x)  # # [64,256,300]
        # pos_emb = self.position_embedding(x)
        mask = (x != 0)  # [64,256]
        out = word_emb
        for dgcnn in self.dgcnn:
            out = dgcnn(out, mask)
        out = torch.max(out, dim=1)[0]
        out = self.fc(out)  # [128,10]
        return out


class DGCNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, k_size=3, dilation_rate=1, dropout=0.1):
        super(DGCNNLayer, self).__init__()
        self.k_size = k_size
        self.dilation_rate = dilation_rate
        self.hid_dim = out_channels
        self.pad_size = int(self.dilation_rate * (self.k_size - 1) / 2)
        self.dropout_layer = nn.Dropout(dropout)
        # self.liner_layer = nn.Linear(int(out_channels / 2), out_channels)
        self.glu_layer = nn.GLU()
        self.conv_layer = nn.Conv1d(in_channels, out_channels * 2, kernel_size=k_size, dilation=dilation_rate,
                                    padding=(self.pad_size,))
        self.layer_normal = nn.LayerNorm(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):
        '''

        :param x: shape: [batch_size, seq_length, channels(embeddings)]
        :return:
        '''
        x_r = x
        x = x.permute(0, 2, 1)  # [batch_size, 2*hidden_size, seq_length]
        x = self.conv_layer(x)  # [batch_size, 2*hidden_size, seq_length]
        x = x.permute(0, 2, 1)  # [batch_size, seq_length, 2*hidden_size]
        x = self.glu_layer(x)  # [batch_size, seq_length, hidden_size]
        x = self.dropout_layer(x)  #
        mask = mask.unsqueeze(2).repeat(1, 1, self.hid_dim).float()
        x = x * mask
        return self.layer_normal(x + x_r)


class SelfAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super(SelfAttentionLayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert self.hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, q, k, v, mask=None):
        '''

        :param q:   shape [batch_size, seq_length, hid_dim]
        :param k:   shape [batch_size, seq_length, hid_dim]
        :param v:   shape [batch_size, seq_length, hid_dim]
        :param mask:
        :return:
        '''
        batch_size = q.shape[0]

        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # Q,K,V shape [batch_size, n_heads, seq_length, hid_dim // n_heads]

        Q = Q.contiguous().view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.contiguous().view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.contiguous().view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # energy [batch_size, n_heads, seq_length, seq_length]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        # attention [batch_size, n_heads, seq_length, seq_length]
        attention = self.dropout(torch.softmax(energy, dim=-1))
        # x [batch_size, n_heads, seq_length, hid_dim // n_heads]
        x = torch.matmul(attention, V)

        x = x.contiguous().permute(0, 2, 1, 3)
        # x [batch_size, seq_length, hid_dim]
        x = x.contiguous().view(batch_size, -1, self.n_heads * (self.hid_dim // self.n_heads))

        x = self.fc(x)

        if mask is not None:
            mask = mask.squeeze(1).squeeze(1)
            mask = mask.unsqueeze(2).repeat(1, 1, self.hid_dim).float()
            x = x * mask
        # [batch_size, seq_length, hid_dim]
        return x