import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
import transformers
from transformers import AutoModel, BertTokenizerFast
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


data = pd.read_csv("pre_processed_data.csv")

# Código pra visualizar quantidade de notícias reais e falsas no dataset
# plot = data['label'].value_counts().plot.pie(subplots=True)
# plt.show()

# Convertendo o campo label pra um booleano onde:
# fake = 0
# true = 1
data["target"] = data.label
data["label"] = pd.get_dummies(data.label)["true"]
# print(data.head(20))


train_text, temp_text, train_labels, temp_labels = train_test_split(
    data["preprocessed_news"],
    data["label"],
    random_state=2018,
    test_size=0.3,
    stratify=data["target"],
)

val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text, temp_labels, random_state=2018, test_size=0.5, stratify=temp_labels
)

bert = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Plotando o histograma do número de palavras e tokenizando o texto
# seq_len = [len(i.split()) for i in train_text if len(i.split()) < 500]

# pd.Series(seq_len).hist(bins=40, color="firebrick")
# plt.xlabel("Number of Words")
# plt.ylabel("Number of texts")
# plt.show()


MAX_LENGHT = 200
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length=MAX_LENGHT,
    pad_to_max_length=True,
    truncation=True,
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length=MAX_LENGHT,
    pad_to_max_length=True,
    truncation=True,
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length=MAX_LENGHT,
    pad_to_max_length=True,
    truncation=True,
)

## convert lists to tensors

train_seq = torch.tensor(tokens_train["input_ids"])
train_mask = torch.tensor(tokens_train["attention_mask"])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val["input_ids"])
val_mask = torch.tensor(tokens_val["attention_mask"])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test["input_ids"])
test_mask = torch.tensor(tokens_test["attention_mask"])
test_y = torch.tensor(test_labels.tolist())

# define a batch size
batch_size = 32

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataloader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataloader for validation set
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
