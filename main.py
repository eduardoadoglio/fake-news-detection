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

data = pd.read_csv("pre_processed_data.csv")

# Código pra visualizar quantidade de notícias reais e falsas no dataset
# plot = data['label'].value_counts().plot.pie(subplots=True)
# plt.show()

# Convertendo o campo label pra um booleano onde:
# fake = 0
# true = 1
data["target"] = data.label
data["label"] = pd.get_dummies(data.label)["true"]
print(data.head(20))


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
seq_len = [len(i.split()) for i in train_text]

pd.Series(seq_len).hist(bins=40, color="firebrick")
plt.xlabel("Number of Words")
plt.ylabel("Number of texts")
plt.show()
