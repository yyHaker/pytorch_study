import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim)

        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)
        # self.dropout = nn.Dropout(0.2)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        """
        :param input: tensor, [length, 1]
        :param hidden: tensor, ([length, 1, hidden_dim], [length, 1, hidden_dim])
        :return:
            'output': [length, vocab_size]
            'hidden': the same shape with hidden
        """
        length = input.size()[0]
        embeds = self.embeddings(input).view((length, 1, -1))  # [length, 1, embedding_dim]
        output, hidden = self.lstm(embeds, hidden)                # output[length, 1, hidden_dim]
        output = F.relu(self.linear1(output.view(length, -1)))
        # output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, length=1, device="cpu"):
        return (Variable(torch.zeros(length, 1, self.hidden_dim).to(device)),
                Variable(torch.zeros(length, 1, self.hidden_dim)).to(device))
