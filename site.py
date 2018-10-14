from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from pymystem3 import Mystem
mystem = Mystem()
import numpy as np
import os
from math import log
import pandas as pd
import operator
import re
from sklearn.feature_extraction.text import CountVectorizer
import math
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
from operator import itemgetter
from flask import Flask
from flask import request
from flask import render_template
app = Flask(__name__)

data = pd.read_csv('avito_df.csv')


lemmatized_texts = []
for x in data['lemmatized']:
    if type(x) != str:
        lemmatized_texts.append('')
    else:
        lemmatized_texts.append(x)
corpus = lemmatized_texts


def preprocessing(input_text, del_stopwords=True, del_digit=True):
    """
    :input: raw text
        1. lowercase, del punctuation, tokenize
        2. normal form
        3. del stopwords
        4. del digits
    :return: lemmas
    """
    russian_stopwords = set(stopwords.words('russian'))
    if del_digit:
        input_text = re.sub('[0-9]', '', input_text)
    words = [x.lower().strip(string.punctuation+'»«–…') for x in word_tokenize(input_text)]
    lemmas = [mystem.lemmatize(x)[0] for x in words if x]

    lemmas_arr = []
    for lemma in lemmas:
        if del_stopwords:
            if lemma in russian_stopwords:
                continue
        lemmas_arr.append(lemma)
    return lemmas_arr


vec = CountVectorizer()
X = vec.fit_transform(lemmatized_texts)
df_index = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
words = list(vec.get_feature_names())


def inverted_index(df) -> dict:
    """
    Create inverted index by input doc collection
    :return: inverted index
    """
    files = []
    for word in df:
        sub = []
        docs = np.where(df[word] > 0)[0]
        for f in docs:
            dl = len(lemmatized_texts[f].split())
            fr = round(df[word][f]/dl, 4)
            sub.append([f, dl, fr])
        files.append(sub)
    index = pd.DataFrame(data={'Слово': words, 'Информация': files})
    return index

index = inverted_index(df_index)


k1 = 2.0
b = 0.75
avgdl = round(sum([len(q.split(' ')) for q in lemmatized_texts])/len(lemmatized_texts))#средняя длина док-ов в коллекции
N = len(lemmatized_texts)

def score_BM25(qf, dl, avgdl, k1, b, N, n) -> float:
    """
    Compute similarity score between search query and documents from collection
    :return: score
    """
    score = math.log((N-n+0.5)/(n+0.5)) * (k1+1)*qf/(qf+k1*(1-b+b*(dl/avgdl)))
    return score


def compute_sim(lemma, index) -> float:
    """
    Compute similarity score between word in search query and all document  from collection
    :return: score
    """
    doc_list = list(index.loc[index['Слово'] == lemma]['Информация'])[0]
    #print(len(doc_list))
    relevance_dict = {}
    for doc in doc_list:
        relevance_dict[doc[0]] = score_BM25(doc[2], doc[1], avgdl, k1, b, N, len(doc_list))
    return relevance_dict


def get_search_result(query, top=5) -> list:
    """
    Compute sim score between search query and all documents in collection
    Collect as pair (doc_id, score)
    :param query: input text
    :return: list of lists with (doc_id, score)
    """
    query = [que for que in preprocessing(query) if que in words]
    #print(query)
    res = {}
    for word in query:
        #print(word)
        relevance_dict = compute_sim(word, index)
        #print(relevance_dict)
        res = {k: res.get(k, 0) + relevance_dict.get(k, 0) for k in set(res) | set(relevance_dict)}
    return sorted(res.items(), key=operator.itemgetter(1), reverse=True)[0:top]



    
def list_ans(answers):
    res = []
    for ans in [g[0] for g in answers]:
        title = data.iloc[ans]['title']
        num_date = data.iloc[ans]['num_date']
        author = data.iloc[ans]['author']
        address = data.iloc[ans]['address']
        breed = data.iloc[ans]['breed']
        price = data.iloc[ans]['price']
        description = data.iloc[ans]['description']
        one = [title, num_date, author, address, breed, price, description]
        res.append(one)
    return res


def search(search_method, query, top=5):
    try:
        if search_method == 'inverted_index':
            search_result = get_search_result(query, top=top)
        else:
            raise TypeError('unsupported search method')
    except:
        search_result = 'Неправильный запрос!'
    return search_result

@app.route("/", methods=['GET', 'POST'])
def first():
    if request.method == 'POST':
        query = request.form['query']
        if 'inverted_index' in request.form:
            answers = search("inverted_index", query, top=5)
            if answers == 'Неправильный запрос!':
                return render_template('index.html', answers=[["Неправильный запрос, попробуйте ввести по-другому", "", "", "", "", "", ""]])
            else:
                answers = list_ans(answers)
                return render_template('index.html', answers=answers)
        else:
            return render_template("index.html")
    elif request.method == 'GET':
        print("No Post Back Call")
    return render_template("index.html")

if __name__ == '__main__':    
    app.run()
