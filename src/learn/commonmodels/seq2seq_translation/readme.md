# Translation with a Sequence to Sequence Network and attention
我们按照pytorch tutorial实现了两个简单的seq2seq和seq2seq with attention（这个attention计算方法不是很好理解）

##model
the seq2seq model's basic idea

![seq2seq](./files/Seq2Seq.png)


the encoder

![encoder](./files/encoder.png)


the decoder

![decoder](./files/decoder.png)


attention decoder

![attention decoder](./files/attention_decoder.png)


## train
训练过程中的loss

![train loss](./files/seq_seq_train.png)

attention的可视化

![attention show](./files/attention_show.png)



##resource
[1] [pytorch 官方教程](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)