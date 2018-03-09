# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable

import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from datatools import MAX_LENGTH, USE_CUDA
from datatools import SOS_token, EOS_token
from datatools import variableFromSentence
from datatools import prepareData


def evaluate(encoder, decoder, input_lang, output_lang, sentence,
                                                       max_length=MAX_LENGTH):
    """
     输入句子，得到翻译之后的句子， 并且得到decoder's attention
    :param encoder:
    :param decoder:
    :param input_lang: object(Lang), 输入的语言
    :param output_lang: object(Lang), 输出的语言
    :param sentence:
    :param max_length: 输入句子的最大长度
    :return:
    """
    # encode the sentence
    input_variable = variableFromSentence(input_lang, sentence)
    encoder_hidden = encoder.initHidden()
    input_length = input_variable.size()[0]

    # set a Variable, [max_length, hidden_size]
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if USE_CUDA else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    # decode the sentence
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    # 将encoder的最后一个hidden作为decoder的hidden
    decoder_hidden = encoder_hidden

    # 记录decode的sentence和decoder_attentions
    decoded_words = []
    decoded_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)

        decoded_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        # 将当前的输出作为下一个输入
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    return decoded_words, decoded_attentions[: di + 1]


def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('source sentence: ', pair[0])
        print("object sentence: ", pair[1])
        output_words, attentions = evaluate(encoder, decoder, input_lang, output_lang,
                                           pair[0])
        output_sentence = ' '.join(output_words)
        print("predicted sentence: ", output_sentence)
        print("")


def showAttention(input_sentence, output_words, attentions):
    """show attention"""
    # set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(output_words, attentions, input_sentence):
    print("input sentence: ", input_sentence)
    print("output = ", ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


if __name__ == "__main__":
    # load data
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    # load trained model
    print("loading the model.......")
    encoder = torch.load("./models/encoder.pkl")
    decoder = torch.load("./models/decoder.pkl")
    # evaluate
    print("evaluating sentences........")
    evaluate_sentences = [
        "je suis trop froid .",
        "elle a cinq ans de moins que moi .",
        "elle est trop petit .",
        "je ne crains pas de mourir .",
        "c est un jeune directeur plein de talent ."]
    for sent in evaluate_sentences:
        output_words, attentions = evaluate(encoder, decoder, input_lang, output_lang, sent)
        evaluateAndShowAttention(output_words, attentions, sent)







