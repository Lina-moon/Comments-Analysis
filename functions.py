import numpy as np
import pandas as pd
from dostoevsky.models import FastTextSocialNetworkModel
from dostoevsky.tokenization import RegexTokenizer
from pymystem3 import Mystem
from sklearn.externals import joblib
import xgboost as xgb
from xgboost import XGBClassifier
from tqdm import tqdm
tqdm.pandas()



def sentiment(data):
    text = data['Text'].tolist()
    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)
    results = model.predict(text, k=1)

    sentiment = []
    for i, j in enumerate(text):
        sentiment.append(list(results[i].keys())[0])

    df = pd.DataFrame(sentiment, columns=['Sentiment'])
    df['Text'] = pd.Series(text)
    df['Valuation'] = data['Valuation'].tolist()

    def f(x):
        if x == 'positive':
            return 1
        if x == 'negative':
            return -1
        return 0

    df['Sentiment'] = df['Sentiment'].apply(f)
    return df

#Word2Vec vectarization
#Лемматизация, удаление стоп слов, частеречное токенизирование и перевод частей речи к стандарту

conversion_table = {
    'A': 'ADJ',
    'ADV': 'ADV',
    'ADVPRO': 'ADV',
    'ANUM': 'ADJ',
    'APRO': 'DET',
    'COM': 'ADJ',
    'CONJ': 'SCONJ',
    'INTJ': 'INTJ',
    'NONLEX': 'X',
    'NUM': 'NUM',
    'PART': 'PART',
    'PR': 'ADP',
    'S': 'NOUN',
    'SPRO': 'PRON',
    'UNKN': 'X',
    'V': 'VERB'
}
def tag(sentence):
    m = Mystem()
    processed = m.analyze(sentence)
    processed = [x for x in processed if 'analysis' in x and x['analysis']]
    lemmas = [x["analysis"][0]["lex"].lower().strip() for x in processed]
    poses = [x["analysis"][0]["gr"].split(',')[0] for x in processed]
    poses = [x.split('=')[0].strip() for x in poses]
    poses = list(map(conversion_table.get, poses))
    tagged = [lemma+'_'+pos for lemma, pos in zip(lemmas, poses)]
    return tagged

#Word to vector векторизация с усреднением по комментарию,

def get_average_word2vec(tokens_list, vector, k, generate_missing=False):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(vectors, text_taged, df, vec_len,  generate_missing=False):
    embeddings = text_taged.progress_apply(lambda x: get_average_word2vec(x, vectors, vec_len,  generate_missing=generate_missing))
    word2vec = pd.DataFrame(list(embeddings))
    word2vec['Sentiment'] = df['Sentiment'].tolist()
    word2vec['Valuation'] = df['Valuation'].tolist()
    return word2vec


def foo(x):
    if x >=0.5:
        return 1
    else:
        return 0


def get_labels_XGB_300(word2vec):
    model_XGB_300 = xgb.Booster({'nthread': 4})
    model_XGB_300.load_model('XGBoost_W2v_model_300.bin')
    word2vec = xgb.DMatrix(word2vec)
    a =  model_XGB_300.predict(word2vec).tolist()
   # a = a.apply(foo)
    return a


def get_labels_RF_300(word2vec):
    model_RF_300 = joblib.load("RandomForest_W2v_model_300.bin")
    return model_RF_300.predict(word2vec).tolist()


def get_labels_XGB_100(word2vec):
    model_XGB_100 = xgb.Booster({'nthread': 4})
    model_XGB_100.load_model('XGBoost_W2v_model_100.bin')
    word2vec = xgb.DMatrix(word2vec)
    a = model_XGB_100.predict(word2vec).tolist()
    # a = a.apply(foo)
    return a

def get_labels_RF_100(word2vec):
    model_RF_100 = joblib.load("RandomForest_W2v_model_100.bin")
    return model_RF_100.predict(word2vec).tolist()
