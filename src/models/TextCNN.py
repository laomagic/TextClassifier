# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    learning_rate = 1e-3  # 学习率
    filter_sizes = (2, 3, 4)  # 卷积核尺寸
    num_filters = 256
    n_vocab = 50000
    embedding_pretrained = False
    embed = 300
    num_classes = 14
    


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=0)
            # self.embedding.weight.requires_grad=True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # [128,256,32]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # [128,256]
        return x  # [128,256]

    def forward(self, x):
        out = self.embedding(x)  # # [128,32,300]
        out = out.unsqueeze(1)  # [128,1,32,300]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  # [128,1,32,300]
        out = self.dropout(out)  # [128,768]
        out = self.fc(out)  # [128,10]
        return out
