import re
from flask import Flask, render_template, request 
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/input-data', methods=['post'])
def get_data():
    t1 = request.form.get('text1')
    t2 = request.form.get('text2')

    print(t1)
    print(t2)

    documents = [ t1, t2]

    count_vector=CountVectorizer(stop_words='english')
    sparse_matrix=count_vector.fit_transform(documents)

    doc_matrix=sparse_matrix.todense()
    df = pd.DataFrame(doc_matrix)

    data = cosine_similarity(df,df)

    if data[0][1] > 0.7:
        print(f"Both text are having {data[0][1]} percent cosine similarity")
    else:
        print('text are not similar')
    return "hey i got executed"

app.run()