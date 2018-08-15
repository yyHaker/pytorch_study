# -*_-= coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class CnnSoftmax(nn.Module):
    """Cnn softmax neural networks."""
    def __init__(self, vocab_size, embedding_dim, seq_len, num_filters=4, classes=2):
        super(CnnSoftmax, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # conv out_len = (seq_len-(5-1)-1) / 1 +  1
        # [bz, Cin, seq_len] -> [bz, Cout, out_len]
        self.conv1d = nn.Conv1d(in_channels=embedding_dim,
                                out_channels=num_filters,
                                kernel_size=5,
                                stride=1,
                                padding=0)
        # self.max_polling = nn.MaxPool1d()
        self.linear = nn.Linear(in_features=seq_len-4, out_features=classes)

    def forward(self, sent):
        """
        :param sent: [bz, seq_len]
        :return:
           "prob": [bz, out_len, classes]
        """
        # [bz, seq_len, embedding_size]
        emb_sent = self.embedding(sent)
        emb_sent = emb_sent.permute(0, 2, 1)
        conv_out = self.conv1d(emb_sent)

        # [bz, Cout, out_len]
        bz = conv_out.size(0)
        Cout = conv_out.Size(1)
        out_len = conv_out.size(2)
        conv_out = conv_out.permute(0, 2, 1).view(-1, Cout)
        # [bz*out_len, Cout]
        max_conv, _ = torch.max(conv_out, 1)
        max_conv = max_conv.view(bz, out_len)
        prob = F.softmax(self.linear(max_conv))
        return prob


class CnnClassifier(nn.Module):
    """
    CNN + softmax classifier.
    """
    def __init__(self, vocab_size, embedding_dim, output_size,
                 kernel_dim=100, kernel_sizes=(3, 4, 5), dropout=0.5):
        super(CnnClassifier, self).__init__()
        self. embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=kernel_dim,
                                              kernel_size=(K, embedding_dim)) for K in kernel_sizes])
        # Kernel size = (K, D)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)

    def init_weights(self, pretrained_word_vectors, is_static=False):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if is_static:
            self.embedding.weight.requires_grad = False

    def forward(self, inputs, is_training=False):
        """
        :param inputs: [bz, seq_len, D]
        :return:
        """
        inputs = self.embedding(inputs).unsqueeze(1)  # (bz, 1, seq_len, D)
        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]  # [(bz, Co, seq_len-K+1), ...]
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]   # [(bz, Co), .....]

        concated = torch.cat(inputs, 1)

        if is_training:
            concated = self.dropout(concated)  # (bz, len(Ks)*Co)
        out = self.fc(concated)
        return F.log_softmax(out, 1)