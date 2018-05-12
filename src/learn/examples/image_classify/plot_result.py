# -*- coding:utf-8 -*-
"""
plot the result.
losses_dict = {"train_loss": [], "valid_loss": []}
    prec_dic = {"train_p1": [], "train_p3": [], "valid_p1": [], "valid_p3": []}
"""
import numpy as np
import matplotlib.pyplot as plt
from myutils import load_data_from_file

losses_dict = load_data_from_file("result/res34/loss_dict.pkl")
prec_dict = load_data_from_file("result/res34/prec_dict.pkl")

train_loss, valid_loss = losses_dict["train_loss"], losses_dict["valid_loss"]
train_p1, train_p3, valid_p1, valid_p3 = prec_dict["train_p1"], \
                                         prec_dict["train_p3"], prec_dict["valid_p1"], prec_dict["valid_p3"]

assert len(train_loss) == len(valid_loss) and len(train_p1) == len(valid_p1)
# plot data
epoches = np.arange(1, len(train_loss)+1)

plt.figure()
plt.plot(epoches, train_loss, label="train_loss")
plt.plot(epoches, valid_loss, color='red', linewidth=1.0, linestyle='--', label="valid_loss")
plt.legend()
plt.xlabel("EPOCH")
plt.ylabel("loss")
plt.title("train and valid loss")
plt.show()

plt.figure()
plt.plot(epoches, train_p1, label="train_p1")
plt.plot(epoches, train_p3, label="train_p3")
plt.plot(epoches, valid_p1, color='red', linewidth=1.0, linestyle='--', label="valid_p1")
plt.plot(epoches, valid_p3, color='green', linewidth=1.0, linestyle='--', label="valid_p3")
plt.legend()
plt.xlabel("EPOCH")
plt.ylabel("precision")
plt.title("train and valid precision")
plt.show()

