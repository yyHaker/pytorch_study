# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
init_alphas = torch.FloatTensor(1, 5).fill_(-10000.)
sentence_in = Variable(torch.LongTensor([1, 2, 3, 4]).view(1, -1))
print(init_alphas)
print(sentence_in)