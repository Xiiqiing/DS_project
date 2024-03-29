#!/home/projects/ku_00039/people/zelili/programs/miniconda2/envs/phyluce-1.7.1/bin/python3.6
# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []

    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    return np.array(all_ids)

def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts,
        #return_attention_masks=False,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )

    return np.array(enc_di['input_ids'])

def build_model(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)

    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model

AUTO = tf.data.experimental.AUTOTUNE
#12 64, 12 128, 12 32, 6 64, 6 128, 6 32.
EPOCHS = 6
#BATCH_SIZE = 12 * strategy.num_replicas_in_sync
BATCH_SIZE = 32
MAX_LEN = 256
MODEL = 'jplu/tf-xlm-roberta-large'

tokenizer = AutoTokenizer.from_pretrained(MODEL)

sub_news = pd.read_csv('/home/people/zelili/ds_p/final/data/processed_sub_news.csv')

from sklearn.model_selection import  train_test_split
# Size of training is 80%
train, test = train_test_split(sub_news, test_size=0.20, random_state=42)

# Use the 40% test set to split further into test and validation set with 50/50 split
test, val = train_test_split(test, test_size=0.50, random_state=42)

x_train = regular_encode(train.content.tolist(), tokenizer, maxlen=MAX_LEN)
x_valid = regular_encode(val.content.tolist(), tokenizer, maxlen=MAX_LEN)
x_test = regular_encode(test.content.tolist(), tokenizer, maxlen=MAX_LEN)

y_train = train.type.values
y_valid = val.type.values
y_test = test.type.values

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_test,y_test))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)


#with strategy.scope():
transformer_layer = TFAutoModel.from_pretrained(MODEL)
model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()

n_steps = x_train.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)

model.evaluate(test_dataset)