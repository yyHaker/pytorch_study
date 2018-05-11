# -*- coding:utf-8 -*-
"""
plot the result.
losses_dict = {"train_loss": [], "valid_loss": []}
    prec_dic = {"train_p1": [], "train_p3": [], "valid_p1": [], "valid_p3": []}
"""
import numpy as np
import matplotlib.pyplot as plt
from myutils import load_data_from_file

losses_dict = load_data_from_file("loss_dic.pkl")
prec_dict = load_data_from_file("prec_dict.pkl")

