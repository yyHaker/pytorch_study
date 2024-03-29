# coding:utf-8
import torch
import sys
import os
import pickle as p
from utils import *

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

# use the trained model to generate poet
model = torch.load('poetry-gen.pt', map_location=lambda storage, loc: storage)
# define the generate seq length
max_length = 5
with open('wordDict', 'rb') as f:
    word_to_ix = p.load(f)

# get ix_to_word
ix_to_word = invert_dict(word_to_ix)


# Sample from a category and starting letter
def sample(startWord='<START>'):
    input = make_one_hot_vec_target(startWord, word_to_ix)
    hidden = model.initHidden(device=device)
    output_name = ""
    if startWord != "<START>":
        output_name = startWord
    for i in range(max_length):
        input = input.to(device)
        output, hidden = model(input, hidden)
        topv, topi = output.data.topk(1)   # output [length, vocab_size]
        topi = topi[0][0]
        w = ix_to_word[topi.item()]
        if w == "<EOP>":
            break
        else:
            output_name += w
        input = make_one_hot_vec_target(w, word_to_ix)
    return output_name


print(sample("春"))
print(sample("花"))
print(sample("秋"))
print(sample("月"))
print(sample("夜"))
print(sample("山"))
print(sample("水"))
print(sample("葉"))
