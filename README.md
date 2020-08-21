# Comments-Analysis
Classification of classes for comments
One class is comments left by real users without any interests. Another one is class with fake comments left by bots or with intense of changing opinion of others. 
### Data  
Data used in the project is confidential labeled dataset of objects collected from wildberries.ru. 
### Models 
Best results on the crossvalidation were presented by Word2Vec vectorization by ruscorpora_300 model on XGBoost_300 model for classification and was equal to 89% of f1 metrics. Models trained on 15 000 objects each and consist of 302 or 102 features for input. 300 or 100 for vectorization, Sentiment feature which can take values of -1 or 1 and Valuation of the product of service which can take values from 0 to 5.
RandomForest_100 model that didn't fit on github: https://drive.google.com/file/d/1qhvhrV5bOvuxs9bmPjQCW4RGGcIHmofv/view?usp=sharing
Ruscorpora_300 model taken from ruscorpora project: https://github.com/RaRe-Technologies/gensim-data/tree/word2vec-ruscorpora-300 

### Web application
File bootle_start.py contains simple web-app which returns class of the given object by the Get request in the json format. 
Example of the request: 
{http://127.0.0.1:8080/comment/%7B"Text":"Было%20написано,что%20брюки,%20оказались%20легинсами.%20Очень%20приятный%20материал,%20достаточно%20тёплые,%20для%20животика%20места%20ещё%20много,%20регулируется,%20не%20жмут,%20очень%20комфортно.%20На%20мой%2048%20размер,нужно%20было%20заказывать%20на%20размер%20меньше,%20тогда%20было%20бы%20идеально.%20",%20"NmID":"3687171",%20"WBUserID":%20"11363977",%20"Valuation":%20"4",%20"State":%20"0"%20%20%7D}
