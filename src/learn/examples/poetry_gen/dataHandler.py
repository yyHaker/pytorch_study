# coding:utf-8
import sys
import os
import json
import re
import random


def parseRawData(src='./chinese-poetry/json/', poem_class='poet.tang',
                 author=None, constrain=None, shuffle=True):
    """
    :param src: the data path.
    :param poem_class: poem class, e.x. 'poet.tang' or 'poet.song'.
    :param author: All author if None.
    :param constrain: constrain the length of the poem.
    :param shuffle: if shuffle the data.
    :return: a poem data list.
    """
    data = []
    for filename in os.listdir(src):
        if filename.startswith(poem_class):
            data.extend(split_poem(handleJson(src+filename, constrain=constrain)))
    # clear the empty string in data
    while "" in data:
        data.remove("")
    if shuffle:
        random.shuffle(data)
    return data


def sentenceParse(para):
    """
    parse the sentence.
    (sub some char to empty).
    :param para:
    :return:
    """
    # para = "-181-村橋路不端，數里就迴湍。積壤連涇脉，高林上笋竿。早嘗甘蔗淡，生摘琵琶酸。（「琵琶」，嚴壽澄校《張祜詩集》云：疑「枇杷」之誤。）好是去塵俗，煙花長一欄。"
    result, number = re.subn("（.*）", "", para)
    result, number = re.subn("（.*）", "", para)
    result, number = re.subn("{.*}", "", result)
    result, number = re.subn("《.*》", "", result)
    result, number = re.subn("《.*》", "", result)
    result, number = re.subn("[\]\[]", "", result)
    r = ""
    for s in result:
        if s not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
            r += s
    r, number = re.subn("。。", "。", r)
    return r


def handleJson(file, author=None, constrain=None):
    """handle poet json data,
    :param file:
    :param author: All author if None.
    :param constrain: constrain the length of the poem.
    :return:
    """
    rst = []
    data = json.loads(open(file, 'r', encoding='utf-8').read())
    for poetry in data:
        pdata = ""
        # skip the poet whose author is not the author
        if author != None and poetry.get("author") != author:
            continue
        # constrain the length of the poem
        p = poetry.get("paragraphs")
        flag = False
        for s in p:
            sp = re.split("[，！。]", s)
            for tr in sp:
                if constrain != None and len(tr) != constrain and len(tr) != 0:
                    flag = True
                    break
                if flag:
                    break
        if flag:
            continue
        # add the required poet
        for sentence in poetry.get("paragraphs"):
            pdata += sentence
        pdata = sentenceParse(pdata)
        if pdata != "":
            rst.append(pdata)
    return rst


def split_poem(poems):
    """
    split every poem to several sentence.
    :param poems: poem list.
    :return:
    """
    sents = []
    for poem in poems:
        poem_list = re.split("[，！。]", poem)[: -1]
        sents.extend(poem_list)
    # clear the empty string
    for sent in sents:
        if sent == "":
            sents.remove(sent)
    return sents


if __name__ == '__main__':
    # data = parseRawData(author="李白".decode('utf-8'),constrain=5)
    print(sentenceParse("熱暖將來賓鐵文，暫時不動聚白雲。撥卻白雲見青天，掇頭裏許便乘仙。（見影宋蜀刻本《李太白文集》卷二十三。）（以上繆氏本《太白集》）-362-。"))
    data = parseRawData(constrain=5)
    for poem in data:
        if poem.strip() == "":
            print("empty string", poem)
    print(len(data))
