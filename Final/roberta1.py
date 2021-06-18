#!/home/projects/ku_00039/people/zelili/programs/miniconda2/envs/phyluce-1.7.1/bin/python3.6
# -*- coding: utf-8 -*-
data_path = '/home/people/zelili/ds_p/final/data/'
output_path = '/home/people/zelili/ds_p/final/out/'

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
#from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator

from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from collections import Counter
#from nltk.stem import WordNetLemmatize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# define regexp for date
monthsShort = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
monthsLong = "January|February|March|April|May|June|July|August|September|October|November|December"
months = "(" + monthsShort + "|" + monthsLong + ")"
separators = "[-/\s,.]"
days = "\d{2}"
years = "\d{4}"
regex1 = "(" + months + separators + days + "|" + years + ")"
regex2 = "(" + days + separators + months + "|" + years + ")"
regex3 = '^([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|-|\/)([1-9]|0[1-9]|1[0-2])(\.|-|\/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])$|^([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\.|-|\/)([1-9]|0[1-9]|1[0-2])(\.|-|\/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])$'

def my_clean_text(text):
    #replace multiple \s to single space
    text = re.sub(r'(\\n)+|(\\t)+|\s{2,}', ' ', text).lower()
    #delete email
    text = re.sub(r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$', ' ', text)
    #delete URLs
    text = re.sub(r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', ' ', text)
    #delete date
    text = re.sub(regex1, ' ', text); text = re.sub(regex2, ' ', text); text = re.sub(regex3, ' ', text)
    #remove sign at start or end of the words
    text = re.sub(r'[^\w\s>]+\s|\s+[^\w\s<]', ' ', text)
    #remove in-words sign expect -,<,> and '
    text = re.sub(r'[^\w\s\-<>\']', '', text)
    #replace any numbers
    #text = re.sub(r'[\d+,?]\.?\d*', ' ', text)
    #replace multiple \s to single space again
    text = re.sub(r'\s{2,}', ' ', text)
    #text = re.sub(r'["”“@()*|\'#!≥+.,$€%&"]', ' ', text) #remove spical char
    #lower the text
    text = [w.lower() for w in word_tokenize(text)]
    return text

stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)
ps=PorterStemmer()

def process_data(data,col):
  data[col] = data[col].apply(my_clean_text)
  data[col] = data.apply(lambda x: ([w for w in x[col] if not w in stopwords_dict]),axis=1)
  data[col] = data.apply(lambda x: ([ps.stem(w) for w in x[col]]),axis=1)
  data[col] = data[col].apply(lambda x: " ".join(x))

all_news = pd.read_csv('/home/people/zelili/ds_p/final/data/all_news.csv')

# replace to numeric
all_news['type'].replace(['political','reliable'], 1, inplace = True)
all_news['type'].replace(['conspiracy', 'junksci', 'rumor', 'hate', 'unreliable', 'clickbait', 'satire', 'fake','bias'], 0, inplace = True)
# delete 'unknown'
all_news = all_news.drop(all_news[all_news['type']=='unknown'].index)
all_news['type']=pd.to_numeric(all_news['type'])
# remove NaN
all_news = all_news.dropna()
all_news.groupby('type').size().sort_values(ascending = False)

sub_news=all_news.sample(frac=0.1)
sub_news.groupby('type').size().sort_values(ascending = False)

sub_news = process_data(sub_news, 'content')

sub_news.to_csv(f"/home/people/zelili/ds_p/final/data/prep_news.csv")

print('Data cleaned and sub!')


# Size of training is 80%
train, test = train_test_split(sub_news, test_size=0.20, random_state=42)

# Use the 40% test set to split further into test and validation set with 50/50 split
test, val = train_test_split(test, test_size=0.50, random_state=42)
'''
'''
# Set random seed and set device to GPU.
torch.manual_seed(12)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device('cpu')

print('Running device is '+device)

# Initialize tokenizer.
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# Set tokenizer hyperparameters.
MAX_SEQ_LEN = 256
BATCH_SIZE = 32
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)


# Define columns to read.
label_field = Field(sequential=False, use_vocab=False, batch_first=True)
text_field = Field(use_vocab=False,
                   tokenize=tokenizer.encode,
                   include_lengths=False,
                   batch_first=True,
                   fix_length=MAX_SEQ_LEN,
                   pad_token=PAD_INDEX,
                   unk_token=UNK_INDEX)

fields = {'content' : ('content', text_field), 'type' : ('type', label_field)}


# Read preprocessed CSV into TabularDataset and split it into train, test and valid.
train_data, valid_data, test_data = TabularDataset(path=f"{data_path}/prep_news.csv",
                                                   format='CSV',
                                                   fields=fields,
                                                   skip_header=False).split(split_ratio=[0.80, 0.1, 0.1], # 80% train 10% val 10% test
                                                                            stratified=True,
                                                                            strata_field='type')

# Create train and validation iterators.
train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),
                                               batch_size=BATCH_SIZE,
                                               device=device,
                                               shuffle=True,
                                               sort_key=lambda x: len(x.titletext),
                                               sort=True,
                                               sort_within_batch=False)

# Test iterator, no shuffling or sorting required.
test_iter = Iterator(test_data, batch_size=BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)


# Functions for saving and loading model parameters and metrics.
def save_checkpoint(path, model, valid_loss):
    torch.save({'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}, path)


def load_checkpoint(path, model):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

    return state_dict['valid_loss']


def save_metrics(path, train_loss_list, valid_loss_list, global_steps_list):
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, path)


def load_metrics(path):
    state_dict = torch.load(path, map_location=device)
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

# Model with extra layers on top of RoBERTa
class ROBERTAClassifier(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ROBERTAClassifier, self).__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(64, 2)

    def forward(self, input_ids, attention_mask):
        _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        x = self.l2(x)

        return x


def pretrain(model,
             optimizer,
             train_iter,
             valid_iter,
             scheduler = None,
             valid_period = len(train_iter),
             num_epochs = 5):

    # Pretrain linear layers, do not train bert
    for param in model.roberta.parameters():
        param.requires_grad = False

    model.train()

    # Initialize losses and loss histories
    train_loss = 0.0
    valid_loss = 0.0
    global_step = 0

    # Train loop
    for epoch in range(num_epochs):
        for (source, target), _ in train_iter:
            mask = (source != PAD_INDEX).type(torch.uint8)

            y_pred = model(input_ids=source,
                           attention_mask=mask)

            loss = torch.nn.CrossEntropyLoss()(y_pred, target)

            loss.backward()

            # Optimizer and scheduler step
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            # Update train loss and global step
            train_loss += loss.item()
            global_step += 1

            # Validation loop. Save progress and evaluate model performance.
            if global_step % valid_period == 0:
                model.eval()

                with torch.no_grad():
                    for (source, target), _ in valid_iter:
                        mask = (source != PAD_INDEX).type(torch.uint8)

                        y_pred = model(input_ids=source,
                                       attention_mask=mask)

                        loss = torch.nn.CrossEntropyLoss()(y_pred, target)

                        valid_loss += loss.item()

                # Store train and validation loss history
                train_loss = train_loss / valid_period
                valid_loss = valid_loss / len(valid_iter)

                model.train()

                # print summary
                print('Epoch [{}/{}], global step [{}/{}], PT Loss: {:.4f}, Val Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_iter),
                              train_loss, valid_loss))

                train_loss = 0.0
                valid_loss = 0.0

    # Set bert parameters back to trainable
    for param in model.roberta.parameters():
        param.requires_grad = True

    print('Pre-training done!')


# Training Function

def train(model,
          optimizer,
          train_iter,
          valid_iter,
          scheduler = None,
          num_epochs = 5,
          valid_period = len(train_iter),
          output_path = output_path):

    # Initialize losses and loss histories
    train_loss = 0.0
    valid_loss = 0.0
    train_loss_list = []
    valid_loss_list = []
    best_valid_loss = float('Inf')

    global_step = 0
    global_steps_list = []

    model.train()

    # Train loop
    for epoch in range(num_epochs):
        for (source, target), _ in train_iter:
            mask = (source != PAD_INDEX).type(torch.uint8)

            y_pred = model(input_ids=source,
                           attention_mask=mask)
            #output = model(input_ids=source,
            #              labels=target,
            #              attention_mask=mask)

            loss = torch.nn.CrossEntropyLoss()(y_pred, target)
            #loss = output[0]

            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            # Optimizer and scheduler step
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            # Update train loss and global step
            train_loss += loss.item()
            global_step += 1

            # Validation loop. Save progress and evaluate model performance.
            if global_step % valid_period == 0:
                model.eval()

                with torch.no_grad():
                    for (source, target), _ in valid_iter:
                        mask = (source != PAD_INDEX).type(torch.uint8)

                        y_pred = model(input_ids=source,
                                       attention_mask=mask)
                        #output = model(input_ids=source,
                        #               labels=target,
                        #               attention_mask=mask)

                        loss = torch.nn.CrossEntropyLoss()(y_pred, target)
                        #loss = output[0]

                        valid_loss += loss.item()

                # Store train and validation loss history
                train_loss = train_loss / valid_period
                valid_loss = valid_loss / len(valid_iter)
                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)
                global_steps_list.append(global_step)

                # print summary
                print('Epoch [{}/{}], global step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_iter),
                              train_loss, valid_loss))

                # checkpoint
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    save_checkpoint(output_path + '/model.pkl', model, best_valid_loss)
                    save_metrics(output_path + '/metric.pkl', train_loss_list, valid_loss_list, global_steps_list)

                train_loss = 0.0
                valid_loss = 0.0
                model.train()

    save_metrics(output_path + '/metric.pkl', train_loss_list, valid_loss_list, global_steps_list)
    print('Training done!')


# Main training loop
NUM_EPOCHS = 6
steps_per_epoch = len(train_iter)

model = ROBERTAClassifier(0.4)
model = model.to(device)


optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=steps_per_epoch*1,
                                            num_training_steps=steps_per_epoch*NUM_EPOCHS)

print("======================= Start pretraining ==============================")

pretrain(model=model,
         train_iter=train_iter,
         valid_iter=valid_iter,
         optimizer=optimizer,
         scheduler=scheduler,
         num_epochs=NUM_EPOCHS)

NUM_EPOCHS = 12
print("======================= Start training =================================")
optimizer = AdamW(model.parameters(), lr=2e-6)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=steps_per_epoch*2,
                                            num_training_steps=steps_per_epoch*NUM_EPOCHS)

train(model=model,
      train_iter=train_iter,
      valid_iter=valid_iter,
      optimizer=optimizer,
      scheduler=scheduler,
      num_epochs=NUM_EPOCHS)

plt.figure()
train_loss_list, valid_loss_list, global_steps_list = load_metrics(output_path + '/metric.pkl')
plt.plot(global_steps_list, train_loss_list, label='Training')
plt.plot(global_steps_list, valid_loss_list, label='Validation')
plt.xlabel('Total steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/home/people/zelili/ds_p/final/out/acc_rate_changes.pdf')

# Evaluation Function

def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (source, target), _ in test_loader:
                mask = (source != PAD_INDEX).type(torch.uint8)

                output = model(source, attention_mask=mask)

                y_pred.extend(torch.argmax(output, axis=-1).tolist())
                y_true.extend(target.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax = plt.subplot()

    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])

model = ROBERTAClassifier()
model = model.to(device)

load_checkpoint(output_path + '/model.pkl', model)

evaluate(model, test_iter)

print(len(train_data))
print(len(valid_data))
print(len(test_data))

"""### Kaggle"""

kaggle_challenge = pd.read_csv('/home/people/zelili/ds_p/final/data/kaggle_json_to_csv.csv')

kaggle_challenge['article'] =kaggle_challenge['article'].astype(str)

process_data(kaggle_challenge,'article')

kaggle_challenge = kaggle_challenge.rename(columns={'article': 'content'})

kaggle = kaggle_challenge[['content']]

def load_dataset_from_tfds_pred(pd_df,batch_size,preprocess_model):
  dataset = tf.data.Dataset.from_tensor_slices(pd_df.to_dict('list'))
  num_examples = len(pd_df)
  dataset = dataset.batch(batch_size)
  #dataset = dataset.map(lambda ex: (bert_preprocess_model(ex), ex['type']))
  dataset = dataset.map(lambda ex: (preprocess_model(ex)))
  dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
  return dataset, num_examples

kaggle_dataset, _ = load_dataset_from_tfds_pred(
      kaggle, 32, model)

kaggle_challenge['label']=(classifier_model.predict(kaggle_dataset)>0.5).astype("int32")

kaggle_challenge['label'] = np.where(kaggle_challenge['label']==0, 'FAKE', 'REAL')

#kaggle_challenge

res = kaggle_challenge[['id','label']]

res.to_csv('res.csv',index=False)