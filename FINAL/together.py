from gensim.models import Word2Vec, KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import judicial_splitter
import collections
from pymystem3 import Mystem
mystem = Mystem()
import numpy as np
from gensim import matutils
from itertools import groupby
from tqdm import tqdm_notebook as tqdm
import os
from math import log
import pandas as pd
from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import operator
import re
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")
import math
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
from operator import itemgetter
from ast import literal_eval
from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)

model_without_pos = Word2Vec.load('/Users/irene/Downloads/IR/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model')
fname = get_tmpfile("model_doc2vec_avito")
model_doc2vec = Doc2Vec.load(fname)
data = pd.read_csv('avito_df.csv')
df = pd.read_csv('avito_df_vecors.csv')

lemmatized_texts = []
for x in data['lemmatized']:
    if type(x) != str:
        lemmatized_texts.append('')
    else:
        lemmatized_texts.append(x)
corpus = lemmatized_texts

def to_list(string):
    string = string.strip('[]')
    string = string.replace('\n', '')
    if ', ' not in string:
        l = string.split(' ')
    else:
        l = string.split(', ')
    return [float(e) for e in l]

df['w2v'] = df['w2v'].apply(to_list)
df['w2v_tfidf'] = df['w2v_tfidf'].apply(to_list)
df['d2v_hypo'] = df['d2v_hypo'].apply(to_list)

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

def compute_tf(text):
    tf_text = collections.Counter(text)
    for i in tf_text:
        tf_text[i] = tf_text[i]/float(len(text))
    return tf_text

def compute_idf(word, corpus):
    return math.log10(len(corpus)/sum([1.0 for i in corpus if word in i]))

def compute_new_tfidf(word, quary, corpus):
    try:
        quary = preprocessing(quary)
        computed_tf = compute_tf(quary)[word]
        #print(computed_tf)
        tfidf = computed_tf * compute_idf(word, corpus)
    except:
        tfidf = 0.0
    return tfidf

def get_w2v_vectors_paragraph(paragraph, model, tfidf, ind, multiply_tfidf=True, pos=False):
    #"""Получает вектор для параграфа"""
    lemmas_paragraph = preprocessing(paragraph)
    #print('lemmas_paragraph', lemmas_paragraph)
    if len(lemmas_paragraph) == 0:
        return np.zeros(300)
    else:
        vector_paragraph = []
        for lemma in lemmas_paragraph:
            if pos:
                lemma = lemma + '_' + get_pos(lemma)
            try:
                if multiply_tfidf:
                    #print(lemma)
                    tfidf = compute_new_tfidf(lemma, paragraph, corpus)
                    #print(lemma, tfidf)
                    vector = model.wv[lemma] * tfidf
                else:
                    vector = model.wv[lemma]
            except:
                vector = np.zeros(300)
            vector_paragraph.append(vector)
        vec = np.array(vector_paragraph).sum(axis=0) / len(vector_paragraph)
        return vec.tolist()

def get_d2v_vectors(paragraph, model_doc2vec, steps=5, alpha=0.1):
    #"""Получает вектор параграфа"""
    lemmas_paragraph = preprocessing(paragraph, del_stopwords=False)
    model_doc2vec.random.seed(100)
    vector = model_doc2vec.infer_vector(lemmas_paragraph, steps=steps, alpha=alpha)
    return vector.tolist()


def get_d2v_vectors(paragraph, model_doc2vec, steps=5, alpha=0.1):
    #"""Получает вектор параграфа"""
    lemmas_paragraph = preprocessing(paragraph, del_stopwords=False)
    model_doc2vec.random.seed(100)
    vector = model_doc2vec.infer_vector(lemmas_paragraph, steps=steps, alpha=alpha)
    return vector.tolist()

def similarity(v1, v2):
    v1_norm = matutils.unitvec(np.array(v1))
    v2_norm = matutils.unitvec(np.array(v2))
    return np.dot(v1_norm, v2_norm)


def res_v(vectors, names_doc, v_quary):
    res = []
    for i, vector in enumerate(vectors):
        cos_sim = similarity(v_quary, vector)
        res.append([names_doc[i], cos_sim, i])
    res.sort(key=operator.itemgetter(1), reverse=True)
    return res

def res_without_dupl(res, top=5):
    res_without_dupl = set()
    inds = []
    for ind, r in enumerate(res):
        if r[0] in res_without_dupl:
            continue
        else:
            if len(res_without_dupl) == top:
                break
            res_without_dupl.add(r[0])
            inds.append(ind)
        ind += 1
    return itemgetter(*inds)(res)

def search_w2v(quary, model, vectors_w2v, names_doc, tfidf, ind, multiply_tfidf=True, pos=False, top=5):
    v_quary = get_w2v_vectors_paragraph(quary, model, tfidf, ind, multiply_tfidf=multiply_tfidf, pos=pos)
    #print(v_quary)
    res = res_v(vectors_w2v, names_doc, v_quary)
    res = res_without_dupl(res, top=top)
    return res

def search_d2v(quary, model, vectors_d2v, names_doc, steps=5, alpha=0.1, top=5):
    v_quary = get_d2v_vectors(quary, model, steps=steps, alpha=alpha)
    res = res_v(vectors_d2v, names_doc, v_quary)
    res = res_without_dupl(res, top=top)
    return res


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


def blend_d2v_w2v(res_w2v, res_d2v, v, top=5):
    res_w2v = sorted(res_w2v, key = lambda x: (x[0], x[2]))
    res_d2v = sorted(res_d2v, key = lambda x: (x[0], x[2]))
    ranges = []
    for i, res3 in enumerate(res_w2v):
        new_range = res3[1] * v + res_d2v[i][1] * (1-v)
        ranges.append((res3[0], new_range))
    return sorted(ranges, key = lambda x: (x[1]), reverse=True)[0:top]

def norm(vector):
    a = np.asarray(vector)
    return np.interp(a, (a.min(), a.max()), (-1, +1))
def mean_w2v(res_w2v):
    df = pd.DataFrame(list(res_w2v))
    df = df.groupby(0)[1].agg(["count", "sum", "mean"])
    m = df['mean']
    return [(i, el) for i, el in enumerate(m)]

def blend_w2v_index(res_w2v, res_index, v, top=5):
    res_w2v = mean_w2v(res_w2v)
    #print(res_w2v)
    res_w2v = sorted(res_w2v, key = lambda x: (x[0]))
    res_index = sorted(res_index, key = lambda x: (x[0]))
    files_ind = [r[0] for r in res_index]
    res_w2v = np.asarray(res_w2v)[files_ind]
    res_w2v_norm = [(i, j) for i, j in enumerate(norm([l[1] for l in res_w2v]))]
    res_index_norm = [(i, j) for i, j in enumerate(norm([d[1] for d in res_index]))]
    ranges = []
    for i, res3 in enumerate(res_w2v_norm):
        new_range = res3[1] * v + res_index_norm[i][1] * (1-v)
        ranges.append((res3[0], new_range))
    return sorted(ranges, key = lambda x: (x[1]), reverse=True)[0:top]
    
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


def search(search_method, query, model_without_pos, model_doc2vec, vectors_w2v, vectors_d2v, names_doc, tfidf, ind_text, num_par_coll, num_text_coll, top=5):
    try:
        if search_method == 'inverted_index':
            search_result = get_search_result(query, top=top)
        elif search_method == 'word2vec':
            search_result = search_w2v(query, model_without_pos, vectors_w2v, names_doc, tfidf, ind_text, multiply_tfidf=True, pos=False, top=top)
        elif search_method == 'doc2vec':
            search_result = search_d2v(query, model_doc2vec, vectors_d2v, names_doc, steps=5, alpha=0.1, top=top)
        elif search_method == 'doc2vec+word2vec':
            top_w2v = search_w2v(query, model_without_pos, vectors_w2v, names_doc, tfidf, ind_text, multiply_tfidf=True, pos=False, top=num_par_coll)
            top_d2v = search_d2v(query, model_doc2vec, vectors_d2v, names_doc, steps=5, alpha=0.1, top=num_par_coll)
            search_result = blend_d2v_w2v(top_w2v, top_d2v, v=0.5, top=5)
        elif search_method == 'word2vec+inverted_index':
            top_w2v = search_w2v(query, model_without_pos, vectors_w2v, names_doc, tfidf, ind_text, multiply_tfidf=True, pos=False, top=num_par_coll)
            top_ind = get_search_result(query, top=num_text_coll)
            search_result = blend_w2v_index(top_w2v, top_ind, v=0.5, top=5)
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
            answers = search("inverted_index", query, model_without_pos, model_doc2vec, df['w2v_tfidf'], df['d2v_hypo'], df['id_answer'], 'tfidf', 'del', len(df['id_answer']), len(data['description']), top=5)
            answers = list_ans(answers)
            return render_template('index.html', answers=answers)
        elif  'word2vec' in request.form:
            answers = search("word2vec", query, model_without_pos, model_doc2vec, df['w2v_tfidf'], df['d2v_hypo'], df['id_answer'], 'tfidf', 'del', len(df['id_answer']), len(data['description']), top=5)
            answers = list_ans(answers)
            return render_template('index.html', answers=answers)
        elif  'doc2vec' in request.form:
            answers = search("doc2vec", query, model_without_pos, model_doc2vec, df['w2v_tfidf'], df['d2v_hypo'], df['id_answer'], 'tfidf', 'del', len(df['id_answer']), len(data['description']), top=5)
            answers = list_ans(answers)
            return render_template('index.html', answers=answers)
        elif  'doc2vec+word2vec' in request.form:
            answers = search("doc2vec+word2vec", query, model_without_pos, model_doc2vec, df['w2v_tfidf'], df['d2v_hypo'], df['id_answer'], 'tfidf', 'del', len(df['id_answer']), len(data['description']), top=5)
            answers = list_ans(answers)
            return render_template('index.html', answers=answers)
        elif  'word2vec+inverted_index' in request.form:
            answers = search("word2vec+inverted_index", query, model_without_pos, model_doc2vec, df['w2v_tfidf'], df['d2v_hypo'], df['id_answer'], 'tfidf', 'del', len(df['id_answer']), len(data['description']), top=5)
            answers = list_ans(answers)
            return render_template('index.html', answers=answers)
        else:
            return render_template("index.html")
    elif request.method == 'GET':
        print("No Post Back Call")
    return render_template("index.html")

if __name__ == '__main__':    
    app.run(port=3333, debug=True)