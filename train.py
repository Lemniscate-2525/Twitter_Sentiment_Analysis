# Twitter Sentiment Analysis Complete Pipeline : 

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


#Loading Dataset

fp1 = "/content/twitter_training.csv"
fp2 = "/content/twitter_validation.csv"
train_df = pd.read_csv(fp1)
val_df   = pd.read_csv(fp2)

train_df.columns = ["id", "entity", "label", "text"]
val_df.columns   = ["id", "entity", "label", "text"]

train_df.dropna(inplace=True)
val_df.dropna(inplace=True)


#Text Preprocessing: 

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

train_df["text"] = train_df["text"].apply(clean_text)
val_df["text"]   = val_df["text"].apply(clean_text)



#Label Encoding

le = LabelEncoder()
y_train = le.fit_transform(train_df["label"])
y_val   = le.transform(val_df["label"])

X_train_text = train_df["text"]
X_val_text   = val_df["text"]



#BOW+ Logistic Regression:


bow = CountVectorizer(max_features=20000)
X_train_bow = bow.fit_transform(X_train_text)
X_val_bow   = bow.transform(X_val_text)

bow_model = LogisticRegression(max_iter=1000)
bow_model.fit(X_train_bow, y_train)

bow_preds = bow_model.predict(X_val_bow)
bow_acc = accuracy_score(y_val, bow_preds)

print(f"BoW Accuracy: {bow_acc:.4f}")


#TF-IDF + Logistic Regression

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_val_tfidf   = tfidf.transform(X_val_text)

tfidf_model = LogisticRegression(max_iter=1000)
tfidf_model.fit(X_train_tfidf, y_train)

tfidf_preds = tfidf_model.predict(X_val_tfidf)
tfidf_acc = accuracy_score(y_val, tfidf_preds)

print(f"TF-IDF Accuracy: {tfidf_acc:.4f}")


#Tokenization:

MAX_WORDS = 20000
MAX_LEN   = 50

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_val_seq   = tokenizer.texts_to_sequences(X_val_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post")
X_val_pad   = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding="post")



#LSTM Training : 

lstm_model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
    LSTM(128),
    Dropout(0.5),
    Dense(4, activation="softmax")
])

lstm_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = lstm_model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_val_pad, y_val),
    epochs=5,
    batch_size=64
)


#LSTM Report : 

lstm_preds = np.argmax(lstm_model.predict(X_val_pad), axis=1)
lstm_acc = accuracy_score(y_val, lstm_preds)

print(f"LSTM Accuracy: {lstm_acc:.4f}")
print("\nClassification Report (LSTM):\n")
print(classification_report(y_val, lstm_preds, target_names=le.classes_))



#Accuracy Comparison:

models = ["BoW", "TF-IDF", "LSTM"]
accuracies = [bow_acc, tfidf_acc, lstm_acc]

plt.figure()
plt.bar(models, accuracies)
plt.title("Accuracy Comparison Across NLP Models")
plt.ylabel("Accuracy")
plt.show()



#Confusion Matrix (LSTM)

cm = confusion_matrix(y_val, lstm_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - LSTM")
plt.show()
