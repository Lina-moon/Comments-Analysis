'''В связке работают ruscorpora_300 как embedding с RandomForest_W2v_model_300 и XGBoost_W2v_model_300
W2v_model_100 c RandomForest_W2v_model_100 и XGBoost_W2v_model_100
Первый вариант дает качество на 2-3% лучше, но модель ruscorpora_300 весит 430 мб'''

from functions import sentiment, tag, get_average_word2vec, get_word2vec_embeddings, get_labels_RF_300,\
    get_labels_XGB_300, get_labels_XGB_100, get_labels_RF_100
from sklearn.externals import joblib
import pandas as pd

def predict_class(data, vec_len = 100):

    df = sentiment(data)

    text_taged =  df.Text.progress_apply(tag)


    if vec_len == 300:
        model_word2vec = joblib.load("ruscorpora_300.bin")
        word2vec = get_word2vec_embeddings(model_word2vec, text_taged, df, vec_len)
        labels_RF_300 = get_labels_RF_300(word2vec)
        labels_XGB_300 = get_labels_XGB_300(word2vec)
        two_lists_of_labels = [labels_RF_300, labels_XGB_300]
        return two_lists_of_labels
        print(labels_RF_300, '\n' ,  labels_XGB_300)
    else:
        model_word2vec = joblib.load("W2v_model_100.bin")
        word2vec = get_word2vec_embeddings(model_word2vec, text_taged, df, vec_len)
        labels_RF_100 = get_labels_RF_100(word2vec)
        labels_XGB_100 = get_labels_XGB_100(word2vec)
        two_lists_of_labels = [labels_RF_100, labels_XGB_100]
        return two_lists_of_labels
        print(labels_RF_100, '\n' ,  labels_XGB_100)


if __name__ == "__main__":

    vec_len = 0
    while vec_len != 300 and vec_len!= 100:
        vec_len = int(input('Введите длинну вектора, 300 или 100: '))
    print('Okay', '\n')

    data = pd.read_csv(r'comments_state+none.csv', engine='python', encoding='UTF-8')
    print('Файл загружен')
    notcheat = data.loc[data['State'] != "cheat"]
    data1 = notcheat.head(100)
    cheat = data.loc[data['State'] == "cheat"]
    data2 = cheat.head(100)
    data = pd.concat([data1, data2], axis=0)
    data.loc[data.State == 'cheat', 'State'] = 1
    data.loc[data.State == 'none', 'State'] = 0
    y = data['State'].tolist()

    two_lists_of_labels =  predict_class(data)



    #model_XGBoost = xgb.XGBClassifier()
    #booster = xgb.Booster()
    #booster.load_model('XGBoost_W2v_model1.bin')
    print(data.State)



