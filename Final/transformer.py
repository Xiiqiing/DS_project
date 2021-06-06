# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %% [markdown]
# ## Setup

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np 
import pandas as pd 
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
nltk.download('punkt')
from typing import Tuple


# %%
# setup tpu
tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
print('Running on TPU ', tpu.master())
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)
print("Number of accelerators: ", strategy.num_replicas_in_sync)


# %%
from google.colab import drive
drive.mount('/content/drive')

'''
---------------------------Below code is for process sub newsdata, ignore if precessed!----------------
# %% [markdown]
# ## Load subdata

# %%
# # 0-> fake, 1->true
# sub_news = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Final/sub_news.csv')
# sub_news.head()

# %% [markdown]
# ## Data processing

# %%
# # define regexp for date

# monthsShort = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
# monthsLong = "January|February|March|April|May|June|July|August|September|October|November|December"
# months = "(" + monthsShort + "|" + monthsLong + ")"
# separators = "[-/\s,.]"
# days = "\d{2}"
# years = "\d{4}"
# regex1 = "(" + months + separators + days + "|" + years + ")"
# regex2 = "(" + days + separators + months + "|" + years + ")"
# regex3 = '^([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|-|\/)([1-9]|0[1-9]|1[0-2])(\.|-|\/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])$|^([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\.|-|\/)([1-9]|0[1-9]|1[0-2])(\.|-|\/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])$'


# # define regexp for the text 

# def my_clean_text(text):
#     #replace multiple \s to single space 
#     text = re.sub(r'(\\n)+|(\\t)+|\s{2,}', ' ', text).lower()
#     #delete email
#     text = re.sub(r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$', ' ', text)
#     #delete URLs
#     text = re.sub(r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', ' ', text)
#     #delete date
#     text = re.sub(regex1, ' ', text); text = re.sub(regex2, ' ', text); text = re.sub(regex3, ' ', text)
#     #remove sign at start or end of the words
#     text = re.sub(r'[^\w\s>]+\s|\s+[^\w\s<]', ' ', text)
#     #remove in-words sign expect -,<,> and '
#     text = re.sub(r'[^\w\s\-<>\']', '', text)
#     #replace any numbers to <NUM>
#     text = re.sub(r'[\d+,?]\.?\d*', ' ', text)
#     #replace multiple \s to single space again
#     text = re.sub(r'\s{2,}', ' ', text)
#     #text = re.sub(r'["”“@()*|\'#!≥+.,$€%&"]', ' ', text) #remove spical char
#     #lower the text
#     text = [w.lower() for w in word_tokenize(text)]
#     return text

# sub_news['content'] = sub_news.apply(lambda x: my_clean_text(x['content']), axis=1)


# %%
# #from nltk.stem import WordNetLemmatize
# nltk.download('wordnet')
# nltk.download('stopwords')
# sub_news['content'] = sub_news.apply(lambda x: ([w for w in x['content'] if not w in stopwords.words('english')]), axis=1)
# #sub_news['content'] = sub_news.apply(lambda x: ([WordNetLemmatizer().lemmatize(w) for w in x['content']]), axis=1)
# sub_news['content'] = sub_news.apply(lambda x: ([PorterStemmer().stem(w) for w in x['content']]), axis=1)


# %%
# sub_news.to_csv('/content/drive/MyDrive/Colab Notebooks/Final/processed_sub_news.csv',index=False)

---------------------------Above code is for process sub newsdata, ignore if precessed!----------------
'''
# %% [markdown]
# ## Load processed data
sub_news_path=''
# %%
sub_news = pd.read_csv('sub_news_path')


# %%
sub_news.head()


# %%
vocab_size=50000
maxlen=1024


# %%
def tok_and_pad(data,vocab_size=vocab_size, maxlen=maxlen):
    oov_tok = '<OOV>'
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(data)
    x_seq = tokenizer.texts_to_sequences(data)
    data_padded = pad_sequences(x_seq, padding='post', maxlen=maxlen)
    return data_padded
data_padded = tok_and_pad(sub_news['content'])

# %% [markdown]
# ## Model

# %%
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# %%
with strategy.scope():
    embed_dim = 40
    num_heads = 4
    ff_dim = 32

    inputs = layers.Input(shape = (maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(10, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs = inputs, outputs = outputs)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# %%
model.summary()


# %%
def to_tensor(data):
  data = tf.convert_to_tensor(data, dtype=tf.float32)
  return data
label = to_tensor(np.array(sub_news['type']))

data_padded = to_tensor(data_padded)


# %%
def split_train_test(features: tf.Tensor,
                     labels: tf.Tensor,
                     test_size: float,
                     random_state: int = 1729) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    random = tf.random.uniform(shape=(tf.shape(features)[0],), seed=random_state)
    train_mask = random >= test_size
    test_mask = random < test_size

    train_features, train_labels = tf.boolean_mask(features, mask=train_mask), tf.boolean_mask(labels, mask=train_mask)
    test_features, test_labels = tf.boolean_mask(features, mask=test_mask), tf.boolean_mask(labels, mask=test_mask)

    return train_features, test_features, train_labels, test_labels

# Size of training is 60%
X_train, X_test, y_train, y_test = split_train_test(data_padded, label, test_size=0.40, random_state=42)

# Use the 40% test set to split further into test and validation set with 50/50 split
X_test, X_val, y_test, y_val = split_train_test(X_test, y_test, test_size=0.50, random_state=42)

print(len(y_train))
print(len(y_val))
print(len(y_test))


# %%
import os
# checkpoint_path = "gs://colab-tpu-bucket/checkpoint/base"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.00001,patience=2,restore_best_weights=True)

# %% [markdown]
# ## Train

# %%
# model.load_weights(checkpoint_path)
history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 20, batch_size = 32,callbacks=[earlystop_callback])


# %%
# plot history
# ...

# %% [markdown]
# ## Test

# %%
with strategy.scope():
  results = model.evaluate(X_test, y_test, batch_size=32)
  print("test loss, test acc:", results)

'''
518/518 [==============================] - 10s 15ms/step - loss: 0.1841 - accuracy: 0.9341
test loss, test acc: [0.1841350644826889, 0.9341470003128052]
'''
# %%



