from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def first():
    if request.method == 'POST':
        query = request.form['query']
        print(query)
        if 'inverted_index' in request.form:
            print("inverted_index")
            answers = list(query)
            answers = search(search_method, query, model_without_pos, model_doc2vec, df['w2v_tfidf'], df['d2v_hypo'], df['id_answer'], tfidf, 'del', len(df['id_answer']), len(data['description']), top=5)
            return render_template('index.html', answers=answers)
        elif  'word2vec' in request.form:
            pass
        elif  'doc2vec' in request.form:
            pass
        elif  'doc2vec+word2vec' in request.form:
            pass
        elif  'word2vec+inverted_index' in request.form:
            pass           
        else:
            return render_template("index.html")
    elif request.method == 'GET':
        print("No Post Back Call")
    return render_template("index.html")

if __name__ == '__main__':
    app.run(port=3333, debug=True)
