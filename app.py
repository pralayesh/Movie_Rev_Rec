import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect,url_for
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import urllib.request
import requests
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from pytz import timezone
from tzlocal import get_localzone
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import random
import re
import os
import pickle

df = pd.read_csv('data.csv')
df1=pd.read_csv('review_sentiment_data.csv')
X=list(df1['review'].values)
tv = TfidfVectorizer(max_features=5000,ngram_range=(1,2))
X=tv.fit_transform(X)

main_movie={}
rec_movies=[]
reviews=[]

cv = CountVectorizer()
count_matrix = cv.fit_transform(df['comb'])
similarity = cosine_similarity(count_matrix)

app = Flask(__name__,template_folder='template')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test1.db'
db = SQLAlchemy(app)

class Review(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    movie=db.Column(db.String(100), nullable=False)
    content = db.Column(db.String(400), nullable = False)
    sentiment = db.Column(db.String(400), nullable = False)
    polarity = db.Column(db.Float)
    date = db.Column(db.DateTime,default = datetime.utcnow)

    def __repr__(self):
        return '<Review %r>' % self.id

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/rec',methods=['POST'])
def recommend():
    global main_movie,rec_movies,reviews
    movie_name = request.form['movie_name']
    url = "https://movie-database-imdb-alternative.p.rapidapi.com/"

    querystring = {"page":"1","r":"json","s":movie_name}

    headers = {
        'x-rapidapi-host': "movie-database-imdb-alternative.p.rapidapi.com",
        'x-rapidapi-key': "8cef60f9bemsh2cc57fd709ca060p19f481jsn74becaf834bc"
        }
    try:
        response = requests.request("GET", url, headers=headers, params=querystring)
        res = json.loads(response.text)

        movie_title=res['Search'][0]['Title'].lower()
        movie_id = res['Search'][0]['imdbID']
    except:
        #print("---------------")
        return render_template('recommend.html',status=0)

    if movie_title in df['movie_title'].unique():
        i = df[df['movie_title']==movie_title]['movie_title'].index.values[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[0:11]
        print(lst)
        movie_ids=[]
        for smi in lst:
            sm=df.iloc[smi[0],-2]
            print(sm)
            url = "https://movie-database-imdb-alternative.p.rapidapi.com/"

            querystring = {"page":"1","r":"json","s":sm}

            headers = {
                'x-rapidapi-host': "movie-database-imdb-alternative.p.rapidapi.com",
                'x-rapidapi-key': "8cef60f9bemsh2cc57fd709ca060p19f481jsn74becaf834bc"
                }
            try:
                response = requests.request("GET", url, headers=headers, params=querystring)
                res = json.loads(response.text)

                movie_ids.append(res['Search'][0]['imdbID'])
                #print(movie_ids)
            except:
                pass

        url = "https://movie-database-imdb-alternative.p.rapidapi.com/"

        querystring = {"i":movie_ids[0],"r":"json"}

        headers = {
            'x-rapidapi-host': "movie-database-imdb-alternative.p.rapidapi.com",
            'x-rapidapi-key': "8cef60f9bemsh2cc57fd709ca060p19f481jsn74becaf834bc"
            }

        response = requests.request("GET", url, headers=headers, params=querystring)
        res = json.loads(response.text)
        #print(res)
        main_movie={"Title":res['Title'],
                    "Released":res['Released'],
                    "Runtime":res['Runtime'],
                    "Genre":res['Genre'],
                    "Director":res['Director'],
                    "Actors":res['Actors'],
                    "imdbRating":res['imdbRating'],
                    "Plot":res['Plot'],
                    "Language":res['Language'],
                    "Country":res['Country'],
                    "Awards":res['Awards'],
                    "Poster":res['Poster'],
                    "Production":res['Production'] }

        l = Review.query.filter_by(movie=main_movie['Title']).all()
        l = l[::-1]
        l = l[:5]
        reviews=l

        rec_movies=[]
        for id in movie_ids[1:]:
            url = "https://movie-database-imdb-alternative.p.rapidapi.com/"

            querystring = {"i":id,"r":"json"}

            headers = {
                'x-rapidapi-host': "movie-database-imdb-alternative.p.rapidapi.com",
                'x-rapidapi-key': "8cef60f9bemsh2cc57fd709ca060p19f481jsn74becaf834bc"
                }

            response = requests.request("GET", url, headers=headers, params=querystring)
            res = json.loads(response.text)

            sm = {"Title":res['Title'],"Poster":res['Poster']}
            rec_movies.append(sm)
        
        return render_template('recommend.html',main_movie=main_movie,rec_movies=rec_movies,status=1,rec_status=1,rev_status=1,reviews=reviews)

    else:
        #print(".........................")
        url = "https://movie-database-imdb-alternative.p.rapidapi.com/"

        querystring = {"i":movie_id,"r":"json"}

        headers = {
            'x-rapidapi-host': "movie-database-imdb-alternative.p.rapidapi.com",
            'x-rapidapi-key': "8cef60f9bemsh2cc57fd709ca060p19f481jsn74becaf834bc"
            }

        response = requests.request("GET", url, headers=headers, params=querystring)
        res = json.loads(response.text)
        try:
            main_movie={"Title":res['Title'],
                        "Released":res['Released'],
                        "Runtime":res['Runtime'],
                        "Genre":res['Genre'],
                        "Director":res['Director'],
                        "Actors":res['Actors'],
                        "imdbRating":res['imdbRating'],
                        "Plot":res['Plot'],
                        "Language":res['Language'],
                        "Country":res['Country'],
                        "Awards":res['Awards'],
                        "Poster":res['Poster'],
                        "Production":res['Production'] }
        except:
            return render_template('recommend.html',status=0)
        rec_movies={}
        l = Review.query.filter_by(movie=main_movie['Title']).all()
        l = l[::-1]
        l = l[:5]
        reviews=l
        return render_template('recommend.html',main_movie=main_movie,rec_movies=rec_movies,status=1,rec_status=0,rev_status=1,reviews=reviews)

@app.route('/rev',methods=['POST'])
def predict():
    global main_movie,rec_movies,reviews
    
    if request.method == "POST":
        movie = request.form['movie_name']
        review1  = request.form['review']
        ps=PorterStemmer()
        review = re.sub('[^a-zA-Z]',' ',review1)
        review = review.lower()
        review = review.split()
        review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
        review=' '.join(review)

        revs=list(df1['review'].values)
        revs.append(review)

        test_rev=tv.fit_transform(revs)
        rev=test_rev[-1,:]

        filename = 'sentiment_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        sentiment=loaded_model.predict(rev)[0]
        print(sentiment)

        if(sentiment==0):
            s='Negetive'
        else:
            s='Positive'
        now_utc = datetime.now(timezone('UTC'))
        now_local = now_utc.astimezone(get_localzone())
        
        r = Review(movie=movie,content= review1,sentiment=s,date=now_local)
        db.session.add(r)
        db.session.commit()
        
        l = Review.query.filter_by(movie=main_movie['Title']).all()
        l = l[::-1]
        l = l[:5]
        reviews=l

        return render_template('recommend.html',main_movie=main_movie,rec_movies=rec_movies,status=1,rec_status=1,rev_status=1,reviews=reviews)

# @app.route('/show',methods=['POST'])
# def show_reviews():
#     movie = request.form['movie_name']
#     l = Review.query.filter_by(movie=movie).all()
#     l = l[::-1]
#     l = l[:5]

#     return render_template('recommend.html',main_movie=main_movie,rec_movies=rec_movies,status=1,rec_status=1,reviews=l,rev_status=1)

if(__name__=="__main__"):
    app.run(debug=True)

