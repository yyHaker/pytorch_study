import torch
import torch.autograd as autograd


def make_one_hot_vec(word, word_to_ix):
    rst = torch.zeros(1, 1, len(word_to_ix))
    rst[0][0][word_to_ix[word]] = 1
    return autograd.Variable(rst)


def make_one_hot_vec_target(word, word_to_ix):
    """use one-hot vec to represent word.
    :param word:
    :param word_to_ix:
    :return:
    """
    rst = autograd.Variable(torch.LongTensor([word_to_ix[word]]))
    return rst


def prepare_sequence(seq, word_to_ix):
    idxs = [word_to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


def toList(sen):
    rst = []
    for s in sen:
        rst.append(s)
    return rst


def invert_dict(d):
    return dict((v, k) for k, v in d.items())


def makeForOneCase(s, one_hot_var_target):
    """convert one sentence to two list for RNN to train.
    tmpOut: A2, A3, A4, .... An.
    tmpIn:   A1, A2, A3,..... An-1.
    :param s: a sentence.
    :param one_hot_var_target: one-hot vec representation of word.
    :return:
    """
    tmpIn = []
    tmpOut = []
    for i in range(1, len(s)):
        w = s[i]
        w_b = s[i - 1]
        tmpIn.append(one_hot_var_target[w_b])
        tmpOut.append(one_hot_var_target[w])
    return torch.cat(tmpIn), torch.cat(tmpOut)
