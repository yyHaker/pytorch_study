# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn
import numpy as np


def to_scalar(var):
    """
    return a python float
    :param var: Variable, dim=1
    :return:
    """
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    """
    return the argmax as a python int
    :param vec: Variable, ()
    :return:
    """
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    """
    Compute log sum exp in a numerically stable way for the forward algorithm.
    x* = max(x1, x2, x3, ..., xn)
    log(exp(x1)+exp(x2)+exp(x3)+...+exp(xn))=x*+ log(exp(x1-x*)+exp(x2-x*)+exp(x3-x*)+...+exp(xn-x*))
    :param vec: ?
    :return:
    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(vec - max_score_broadcast))