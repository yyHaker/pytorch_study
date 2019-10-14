#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: basic_text_classification_example.py
@time: 2019/10/7 15:37
"""

import numpy as np
import pandas as pd
from functools import partial
from overrides import overrides
from pathlib import Path

from typing import List, Dict, Iterator, Callable, Optional, Any, Iterable
import torch
import torch.optim as optim
import torch.nn as nn

from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ArrayField
from allennlp.data.iterators import BucketIterator, BasicIterator

from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.training.trainer import Trainer


# 参数设置
class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


config = Config(
    testing=True,
    seed=1,
    batch_size=64,
    lr=3e-4,
    epochs=5,
    hidden_sz=64,
    max_seq_len=100,  # necessary to limit memory usage
    max_vocab_size=100000,
)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class JigsawDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer]=None,
                 max_seq_len: Optional[int]=config.max_seq_len) -> None:
        super(JigsawDatasetReader, self).__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer}
        self.max_seq_len = max_seq_len

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path)
        if config.testing:
            df = df.head(1000)
        for i, row in df.iterrows():
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(row["comment_text"])],
                row["id"], row[label_cols].values,
            )

    @overrides
    def text_to_instance(self, tokens: List[Token], id: str=None, labels: np.ndarray=None) -> Instance:
        sentence_filed = TextField(tokens, self.token_indexers)
        fileds = {"tokens": sentence_filed}

        id_filed = MetadataField(id)
        fileds["id"] = id_filed

        if labels is None:
            labels = np.zeros(len(label_cols))
        label_field = ArrayField(array=labels)

        fileds["label"] = label_field
        return Instance(fileds)


def tokenizer(x: str):
    return [w.text for w in SpacyWordSplitter(language='en_core_web_sm', pos_tags=False).split_words(x)[: config.max_seq_len]]


# read data
token_indexer = SingleIdTokenIndexer()
reader = JigsawDatasetReader(tokenizer=tokenizer, token_indexers={"tokens": token_indexer})
DATA_ROOT = Path("data") / "jigsaw"
train_ds, test_ds = (reader.read(DATA_ROOT / fname) for fname in ["train.csv", "test_proced.csv"])
val_ds = None

# prepare vocab
vocab = Vocabulary.from_instances(train_ds, max_vocab_size=config.max_vocab_size)

# prepare iterator
iterator = BucketIterator(batch_size=config.batch_size, sorting_keys=[("tokens", "num_tokens")])
iterator.index_with(vocab)

# test data
# batch = next(iter(iterator(train_ds)))
# print(batch)
# print(batch.keys())
# print(batch["tokens"]["tokens"].shape)


class BaselineModel(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 out_sz: int=len(label_cols)):
        super(BaselineModel, self).__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.projection = nn.Linear(self.encoder.get_output_dim(), out_sz)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, tokens: Dict[str, torch.Tensor],
                id: Any, label: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, mask)
        class_logits = self.projection(state)

        output = {"class_logits": class_logits}
        output["loss"] = self.loss(class_logits, label)

        return output


USE_CUDA = torch.cuda.is_available()
token_embedding = Embedding(num_embeddings=config.max_vocab_size + 2,
                            embedding_dim=300, padding_index=0)
word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})
encoder: Seq2VecEncoder = PytorchSeq2VecWrapper(nn.LSTM(word_embeddings.get_output_dim(),
                                                        config.hidden_sz, bidirectional=True, batch_first=True))
model = BaselineModel(word_embeddings, encoder)
if USE_CUDA:
    model.cuda()


optimizer = optim.Adam(model.parameters(), lr=config.lr)
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_ds,
    cuda_device=0 if USE_CUDA else -1,
    num_epochs=config.epochs
)

trainer.train()

# predictor
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor
tagger = SentenceTaggerPredictor(model, reader)
print(tagger.predict("this tutorial was great!"))



