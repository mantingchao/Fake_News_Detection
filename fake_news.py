#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:54:49 2020

@author: Manting
"""

import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier

#%% 讀資料，分割符號為tab(\t)
train_df = pd.read_csv('train.csv' ,delimiter="\t").drop([1615])
test_df = pd.read_csv('test.csv' ,delimiter="\t")
label_df = pd.read_csv('sample_submission.csv' ,delimiter="\t")
label_df[['id','label']] = label_df['id,label'].str.split(",",expand=True) 
X_train = train_df.loc[:,['text']]
X_test = test_df.loc[:,['text']] 
y_train = train_df.loc[:,['label']] 
y_test = label_df.loc[:,['label']]

#%% TFI-DF
print("stop_words = english")
vectorizer = TfidfVectorizer(stop_words = "english")

# =============================================================================
# # NLTK
# from nltk.corpus import stopwords
# stop_word = set(stopwords.words('english'))
# vectorizer = TfidfVectorizer(stop_words = stop_word)
# print("stop_words = NLTK")
# =============================================================================

# =============================================================================
# # Gensim
# stop_word = gensim.parsing.preprocessing.STOPWORDS
# vectorizer = TfidfVectorizer(stop_words = stop_word)
# print("stop_words = Gensim")
# =============================================================================

# =============================================================================
# # spaCy
# from spacy.lang.en.stop_words import STOP_WORDS
# vectorizer = TfidfVectorizer(stop_words = STOP_WORDS)
# print("stop_words = spaCy")
# =============================================================================

tfidf_train = vectorizer.fit_transform(X_train['text'])
tfidf_test = vectorizer.transform(X_test['text'])

#%% 建模
# xgboost
xgb = XGBClassifier()
xgb.fit(tfidf_train, y_train.values.ravel())
xgb_pred = xgb.predict(tfidf_test)
print("xgboost")
print(" Accuracy: ", accuracy_score(y_test, xgb_pred))
print(" Precision: ", precision_score(y_test, xgb_pred, pos_label='1'))
print(" Recall: ", recall_score(y_test, xgb_pred, pos_label='1'))
print(" F-measure: ", f1_score(y_test, xgb_pred, pos_label='1'))

# GBDT
gbr = GradientBoostingClassifier()
gbr.fit(tfidf_train, y_train.values.ravel())
gbr_pred = gbr.predict(tfidf_test)
print("GBDT")
print(" Accuracy: ", accuracy_score(y_test, gbr_pred))
print(" Precision: ", precision_score(y_test, gbr_pred, pos_label='1'))
print(" Recall: ", recall_score(y_test, gbr_pred, pos_label='1'))
print(" F-measure: ", f1_score(y_test, gbr_pred, pos_label='1'))

# LightGBM
lgbm = LGBMClassifier()
lgbm.fit(tfidf_train, y_train.values.ravel())
lgbm_pred = lgbm.predict(tfidf_test)
print("LightGBM")
print(" Accuracy: ", accuracy_score(y_test, lgbm_pred))
print(" Precision: ", precision_score(y_test, lgbm_pred, pos_label='1'))
print(" Recall: ", recall_score(y_test, lgbm_pred, pos_label='1'))
print(" F-measure: ", f1_score(y_test, lgbm_pred, pos_label='1'))

