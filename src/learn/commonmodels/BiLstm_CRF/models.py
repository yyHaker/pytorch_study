# -*- coding: utf-8 -*-
"""

reference:
[1]BiLSTM_CRF-序列标注: http://blog.csdn.net/ustbfym/article/details/78583154
[2] pytorch版的bilstm+crf实现sequence label: http://blog.csdn.net/appleml/article/details/78664824
[3] github tutorial: https://github.com/rguthrie3/DeepLearningForNLPInPytorch/blob/master/Deep%
  20Learning%20for%20Natural%20Language%20Processing%20with%20Pytorch.ipynb
"""
import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

from src.learn.commonmodels.BiLstm_CRF.utils import *

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        """

        :param vocab_size:
        :param tag_to_ix: a list of index of tags (contanin start and end tags)
        :param embedding_dim: the dim of word embeddings
        :param hidden_dim: the output dim of BiLSTM
        """
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)  # tag sizes

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim/2,
            num_layers=1,
            bidirectional=True
        )

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters. Entry i, j is the score of transitioning to i from j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer *to* the start tag,
        # and we never transfer *from* the stop tag (the model would probably learn this anyway,
        # so this enforcement is likely unimportant)
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, STOP_TAG] = -10000

        self.hidden = self.init_hidden()  # (h_n, c_n)

    def init_hidden(self):
        """
        BiLSTM Outputs: output, (h_n, c_n):
          - **output** (seq_len, batch, hidden_size * num_directions),
          - **h_n** (num_layers * num_directions, batch, hidden_size,
          - **c_n** (num_layers * num_directions, batch, hidden_size)
        :return: h_n, c_n 的初始值
        """
        return Variable(torch.randn(2, 1, self.hidden_dim)), Variable(torch.randn(2, 1, self.hidden_dim))

    def _forward_alg(self, feats):
        """
        Do the forward algorithm to compute the partition function
        :param feats: (seq_len, tag_sizes)
        :return:
        """
        # torch.FloatTensor of size 1 x tag_size
        init_alphas = torch.FloatTensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the edge (i -> next_tag)
                # before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        """
        输入sentence, 得到lstm_feats
        :param sentence:
        :return:
        """
        self.hidden = self.init_hidden()
        # BiLSTM **input** (seq_len, batch, input_size=embedding_dim)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        """
        Gives the score of a provided tag sequence.
        :param feats: (seq_len, tag_sizes)
        :param tags:the givens tag sequence,  torch.LongTensor, a list of index of tags
            (not including the START_TAG and STOP_TAG)
        :return:
        """
        # Initialize score
        score = Variable(torch.FloatTensor([0]))
        # 将START_TAG的标签拼接到tags上
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])

        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
            # feat[tags[i+1]], feat是step i 的输出结果，有５个值，对应B, I, E, START_TAG, END_TAG, 取对应标签的值
            score = score + self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score
