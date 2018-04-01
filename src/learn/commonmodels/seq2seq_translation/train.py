# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import random
import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from datatools import MAX_LENGTH
from datatools import USE_CUDA
from datatools import SOS_token
from datatools import EOS_token
from datatools import variablesFromPair, prepareData

from model import EncoderRNN, AttnDecoderRNN, DecoderRNN


teacher_forcing_ratio = 0.5


def train(input_variable, target_variable, encoder,
          decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):
    """
    :param input_variable:
    :param target_variable:
    :param encoder:
    :param decoder:
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param criterion: 判断预测单词和目标单词之间的匹配程度
    :param max_length:
    :return:
    """
    encoder_hidden = encoder.initHidden()
    # clear gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # tensor [max_length, hidden_size]
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() \
        if USE_CUDA else encoder_outputs

    loss = 0

    # encode the input
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        # 填充encoder的输出Variable, 即encoder的每一时刻的输出
        encoder_outputs[ei] = encoder_output[0][0]

    # tensor [1 x 1]
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    # 将encoder的最后一层hidden_state作为decoder初始hidden
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio \
        else False

    # decoder the input
    if use_teacher_forcing:
        # Teacher forcing: Feed the true target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attetion = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])

            decoder_input = target_variable[di]
    else:
        #  without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # 每次选取概率最大的那个
            topv, topi = decoder_output.data.topk(1)
            # get the index
            ni = topi[0][0]
            # 将当预测的输出作为下一个输入
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break
    # 向后传播
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.data[0] / target_length


def asMinutes(s):
    """将秒化为分钟"""
    # 向下取整
    m = math.floor(s/60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    """get time elapsed and estimate time remaining
    :param since: since time
    :param percent: current progress
    :return:
    """
    now = time.time()
    s = now - since
    es = s / percent
    # estimate remaining time
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    """plot the losses array"""
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


def trainIters(pairs, input_lang, output_lang, encoder, decoder, n_iters=7500,
               print_every=1000, plot_every=100, learning_rate=0.01):
    """
    :param pairs:
    :param input_lang:
    :param output_lang:
    :param encoder:
    :param decoder:
    :param n_iters: 迭代次数
    :param print_every:
    :param plot_every:
    :param learning_rate:
    :return:
    """
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0    # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # prepare data pairs
    training_pairs = [variablesFromPair(random.choice(pairs), input_lang,
                                        output_lang) for i in range(n_iters)]

    # use The negative log likelihood loss.
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        output_variable = training_pair[1]

        loss = train(input_variable, output_variable, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s,  iteration: (%d %d%%) , average loss:%.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # 显示训练过程中的losses
    showPlot(plot_losses)


if __name__ == "__main__":
    # prepare data
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    # RNN hidden size and wordVector dim
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
    # decoder1 = DecoderRNN(hidden_size, output_lang.n_words)

    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)

    if USE_CUDA:
        encoder1 = encoder1.cuda()
        decoder1 = attn_decoder1.cuda()
        # attn_decoder1 = attn_decoder1.cuda()

    # train
    trainIters(pairs, input_lang, output_lang, encoder1, attn_decoder1, n_iters=7500,
               print_every=1000, plot_every=100, learning_rate=0.01)

    # save model
    torch.save(encoder1, './models/encoder1')
    torch.save(attn_decoder1, './models/decoder1')




