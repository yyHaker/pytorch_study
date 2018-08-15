# coding:utf-8
import random
import torch.nn as nn
import torch.optim as optim
import dataHandler
from model import PoetryModel
from utils import *
import pickle as p

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

# load data
print("load data...")
data = dataHandler.parseRawData(constrain=5)

with open("train_data.txt", 'w', encoding='utf-8') as f:
    for poem in data:
        f.write(poem + "\n")

# build word_to_idx
print("build word_to_idx...")
word_to_ix = {}
for sent in data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
word_to_ix['<EOP>'] = len(word_to_ix)
word_to_ix['<START>'] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
print("VOCAB_SIZE:", VOCAB_SIZE)
print("data_size", len(data))

# convert the poem data to list
for i in range(len(data)):
    data[i] = toList(data[i])
    data[i].append("<EOP>")

# save the word dic for sample method
with open('wordDict', 'wb') as f:
    p.dump(word_to_ix, f)

# save all available word
# wordList = open('wordList','w')
# for w in word_to_ix:
#     wordList.write(w.encode('utf-8'))
# wordList.close()

# create model
print("create model...")
model = PoetryModel(vocab_size=len(word_to_ix), embedding_dim=256, hidden_dim=256)
model.to(device)
optimizer = optim.RMSprop(model.parameters(), lr=0.01, weight_decay=0.0001)

# use negative log likelihood loss.
criterion = nn.NLLLoss()

# one-hot vec representation of word
one_hot_var_target = {}
for w in word_to_ix:
    one_hot_var_target.setdefault(w, make_one_hot_vec_target(w, word_to_ix))

# hyper params
epochs = 10
data_size = len(data)
batch_size = 1000


def evaluate():
    """ use the last batch to evaluate the model"""
    v = int(data_size / batch_size)
    loss = 0
    counts = 0
    for case in range(v * batch_size, min((v + 1) * batch_size, data_size)):
        s = data[case]
        hidden = model.initHidden(device=device)
        s_in, s_o = makeForOneCase(s, one_hot_var_target)
        s_in, s_o = s_in.to(device), s_o.to(device)
        output, hidden = model(s_in, hidden)
        loss += criterion(output, s_o)
        counts += 1
    loss = loss / counts
    return loss.item()


print("start training")
for epoch in range(epochs):
    for batchIndex in range(int(data_size / batch_size)):
        loss = 0
        counts = 0
        for case in range(batchIndex * batch_size, min((batchIndex + 1) * batch_size, data_size)):
            s = data[case]
            hidden = model.initHidden(device=device)
            s_in, s_o = makeForOneCase(s, one_hot_var_target)
            s_in, s_o = s_in.to(device), s_o.to(device)
            output, hidden = model(s_in, hidden)
            loss += criterion(output, s_o)
            counts += 1
        loss = loss / counts
        print("epoch {}, batch {}, loss {}".format(epoch, batchIndex, loss.item()))
        # compute accuracy and record loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # every epoch evaluate the model
    valid_loss = evaluate()
    print("epoch {}, current valid loss {}".format(epoch, valid_loss))
torch.save(model, 'poetry-gen.pt')
