# Fake_News_Detection
###### 針對假新聞作分析，預測一則新聞是否reliable
###### 1: fake  0: true
###### 分別利用GBDT、LightGBM、xgboost對train.csv資料建模，並用test.csv進行測試
###### 利用TF-IDF將文字轉成向量
###### 停頓詞分別用未使用、TfidfVectorizer內建、NLTK、gensim、spacy來測試
