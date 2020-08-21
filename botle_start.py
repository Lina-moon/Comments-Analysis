from bottle import run, route
from main import predict_class
import json
import urllib
import pandas as pd



@route('/login')
def login():
    return '<h1>You are on the first page</h1>'

@route('/comment/<text>')
def commemt(text):
    '''    text = urllib.parse.unquote(text)'''
    dic = json.loads(text)
    data = pd.DataFrame(dic, index=[0])
    data['Valuation'] = int(data['Valuation'])
    two_lists_of_labels = predict_class(data)
    return '<h1>Label from RF ' + str(two_lists_of_labels[0][0])+ ' Label from XGB ' + str(int(two_lists_of_labels[1][0])) + '</h1>'


if __name__ == '__main__':
    run(debug=True, reloader=True)