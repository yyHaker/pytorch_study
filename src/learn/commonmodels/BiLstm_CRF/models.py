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
            hidden_size=hidden_dim // 2,  # make sure is int
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
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

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
        Do the forward algorithm to compute the partition function.
        计算所有路径分数之和
        :param feats: (seq_len, tag_sizes)，like as
           tags
           o     [- - - ,,,  - ]
           b     [- - - ,,,  - ]
           b-p  [- - - ,,,  - ]
            ...    .....
           L-o   [- - - ,,,  - ]
     steps: s  1 2 3 ... n     e
        :return: alpha, the sum of scores of all paths, compute as
           start - - - - - - - - - -
           1     -    -     -      -
           2    -    -    -      -
           .
           .    -    -     -      -
           .
           n   -    -    -      -
           end -   -    -    -
        """
        init_alpha = torch.FloatTensor(1, self.tagset_size).fill_(-10000.)  # 1 x tag_sizes
        # START_TAG has all of the score
        init_alpha[0][self.tag_to_ix[START_TAG]] = 0.
        # wrap in a Variable so that we will get automatic backprop
        forward_var = Variable(init_alpha)

        # iterate through the sentence
        for feat in feats:
            alphas_t = []  # the forward variables at this time step
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)   # 1 x tag_sizes

                # the (next_tag)th  of trans_score is the score of transitioning to next_tag from others
                trans_score = self.transitions[next_tag].view(1, -1)  # 1 x tag_sizes
                # the ith entry of next_tag_var is the value for the edge (i->next_tag) before we do log_sum_exp
                next_tag_var = forward_var + trans_score + emit_score
                # the forward variable for this tag is the log_sum_exp of all the scores
                alphas_t.append(log_sum_exp(next_tag_var))

            forward_var = torch.cat(alphas_t).view(1, -1)  # 1 x tag_sizes
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        """
        输入sentence, 得到lstm_feats
        :param sentence:  LongTensor `(N, W)`, N = mini-batch, W = number of indices to extract per mini-batch
        :return:
        """
        self.hidden = self.init_hidden()
        # BiLSTM **input** (seq_len, batch, input_size=embedding_dim)
        # print(self.word_embeds(sentence))  # 1x11x5
        embeds = self.word_embeds(sentence).view(sentence.size()[1], 1, -1)  # seq_len x 1 x embedding_dim
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = lstm_out.view(-1, self.hidden_dim)   # (seq_len * batch, hidden_size * num_directions)
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
        score = Variable(torch.FloatTensor([0.]))
        # 将START_TAG的标签拼接到tags上
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])

        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
            # feat[tags[i+1]], feat是step i 的输出结果，有５个值，对应B, I, E, START_TAG, END_TAG, 取对应标签的值
            score = score + self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        """
        decode, get the best sequence and sequence score.
        :param feats: (seq_len, tag_sizes)，like as
           tags
           o     [- - - ,,,  - ]
           b     [- - - ,,,  - ]
           b-p  [- - - ,,,  - ]
            ...    .....
           L-o   [- - - ,,,  - ]
      steps: s  1 2 3 ... n     e
        :return: path_score, best sequence score.
                     best_pah, best sequence.
        """
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.FloatTensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the previous step,
                # plus the score of transitioning from tag i to next_tag
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # add the emission scores and assign forward_var to the set of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)  # 1 x tagset_size
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the backpointers to decode the path (从后往前找)
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag
        start = best_path.pop()
        assert self.tag_to_ix[START_TAG] == start   # sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        """
        get the negative log likelihood.
        -log(p(Y|X)) = log_sum_exp(score(X, Y_hat)) - score(X, Y)
        :param sentences:
        :param tags:
        :return:
        """
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)

        return forward_score - gold_score

    def forward(self, sentence):
        """
        正向传播输入值， 神经网络分析输出值.
        :param sentence: a list of index of words
        :return:
        """
        # get the emission score from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path and the best score
        best_score, best_tag_seq = self._viterbi_decode(lstm_feats)
        return best_score, best_tag_seq


# Make up some training data
training_data = [("the wall street journal reported today that apple corporation made money".split(),
                  "B I I I O O O B I O O".split()),
                 ("georgia tech is a university in georgia".split(),
                  "B I O O O O B".split())]

# set the word dictionary and tag dictionary
word_to_ix = {}
for sentence, tag in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

for epoch in range(100):
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Variables of word indices.
        sentence_in = Variable(torch.LongTensor(prepare_sequence(sentence, word_to_ix)).view(1, -1))
        targets = torch.LongTensor([tag_to_ix[t] for t in tags])

        # Step3. Run our forward pass.
        neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)
        print("Epoch %i, cost %lf" % (epoch, neg_log_likelihood.data.numpy()))
        # Step4. Compute gradients and update the parameters
        neg_log_likelihood.backward()
        optimizer.step()

# Check prediction after training
precheck_sent = Variable(torch.LongTensor(prepare_sequence(sentence, word_to_ix)).view(1, -1))
best_score, best_path = model(precheck_sent)
print(best_score)   # the best score
print("-----------------------")
print(best_path)   # the best tag sequence
