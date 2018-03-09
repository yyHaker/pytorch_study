# -*- coding: utf-8 -*-
from io import open
import unicodedata
import string
import re
import random

import torch
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1


class Lang(object):
    """a helper class contains word2index index2word and word2count
    dictionaries."""
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    """Turn a Unicode string to plain ASCII"""
    # http://stackoverflow.com/a/518232/2809427
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """Turn unicode to Ascii, Lowercase, trim, and remove non-letter
    characters"""
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    """read the data from file"""
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in line.split('\t')] for line in lines]
    # print(pairs[:5])
    # reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        # print(pairs[:5])
        input_Lang = Lang(lang2)
        output_Lang = Lang(lang1)
    else:
        input_Lang = Lang(lang1)
        output_Lang = Lang(lang2)

    return input_Lang, output_Lang, pairs


# filtering the data
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    """filter pair
    1. 长度小于10
    2. 以固定格式开头
    """
    # print(p[0][0])
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    """
    1. Read text file and split into lines, split lines into pairs
    2. Normalize text, filter by length and content
    3. Make word lists from sentences in pairs
    :param lang1:
    :param lang2:
    :param reverse:
    :return:
    """
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse=reverse)
    print("read %s sentence pairs." % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs." % len(pairs))
    print("Counting words.....")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    """得到一个句子中每个单词的索引"""
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    """得到句子的Variable变量"""
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    # tensor [seq_len, 1]
    result = Variable(torch.LongTensor(indexes)).view(-1, 1)
    if USE_CUDA:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair, input_lang, output_lang):
    """得到输入输出变量的元组
    :param pair: a pair of sentence containing input_lang
    and output_lang
    :param input_lang: object Lang,
    :param output_lang: object Lang,
    :return:
    """
    input_variable = variableFromSentence(input_lang, pair[0])
    output_variable = variableFromSentence(output_lang, pair[1])
    return input_variable, output_variable


if __name__ == "__main__":
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    print(random.choice(pairs))






