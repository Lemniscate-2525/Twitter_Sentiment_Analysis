# Twitter_Sentiment_Analysis

Twitter Sentiment Analysis; Classical NLP vs Deep Learning (BOW, TF-IDF, LSTM)

This mini project demonstrates a complete NLP pipeline for Twitter sentiment classification
using three different text representations:

1. Bag of Words + Logistic Regression
2. TF-IDF + Logistic Regression
3. Learned Word Embeddings + LSTM

## Objective
To understand how different NLP feature representations affect model performance
and why sequence-aware deep learning models outperform classical approaches.

## Dataset
Twitter sentiment dataset with four classes:
- Positive
- Negative
- Neutral
- Irrelevant

## Key Learnings
- Bag of Words ignores word importance and order, has out of vocabulary(OOV) and sparsity problems.
- TF-IDF improves word weighting but remains orderless, still dosen't fix out of vocabulary and Sparse vector problems.
- LSTM with embeddings captures context, semantics, and word order, solve sparsity problem completely and reduces out of vocabulary problem by a huge margin.
- “Word embeddings eliminate sparse high-dimensional representations by mapping words to dense vectors and partially mitigate the out-of-vocabulary problem through shared or subword-based representations.”
-  Classical NLP methods remain strong baselines for text classification.

## Results
The LSTM model achieves the best performance due to its ability to model sequential
dependencies and learn dense semantic representations.

## Conclusions
The LSTM model outperforms classical approaches largely due to the embedding layer, which converts sparse word representations into dense semantic vectors. These embeddings allow the model to capture similarity between words and, when combined with LSTM-based sequence modeling, enable better understanding of context and sentiment in text.

## Notes
This is a learning-focused project intended to explore NLP representations
and model behavior, not a production-ready system.


Raw Tweets
   ↓
Text Cleaning
   ↓
Label Encoding
   ↓
Feature Extraction
   ├── Bag of Words
   ├── TF-IDF
   └── Tokenization + Padding
   ↓
Models
   ├── Logistic Regression (BoW)
   ├── Logistic Regression (TF-IDF)
   └── LSTM (Embeddings)
   ↓
Evaluation & Comparison

