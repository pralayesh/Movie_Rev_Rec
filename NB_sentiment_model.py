import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import random
import re
import pickle

data = pd.read_csv('desktop/IMDB_Dataset.csv')

print(data.shape)
data=data.dropna()
data=data.sample(frac=1).reset_index(drop=True)
data=data.iloc[:12000,:]
X=data['review']
Y=data['sentiment']
y=[]
for i in range(len(Y)):
    if(Y[i]=='positive'):
        y.append(1)
    else:
        y.append(0)
        
print(y[:10])

ps=PorterStemmer()
corpus=[]
for i in range(0,len(data)):
    review = re.sub('[^a-zA-Z]',' ',data.loc[i,['review']].values[0])
    review = review.lower()
    review = review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

review=list(corpus[:-8])
sentiment=list(y)
print(len(review))
print(len(sentiment))
dic={'review':review,'sentiment':y}
df=pd.DataFrame(dic)
df.to_csv('review_sentiment_data.csv')

tv = TfidfVectorizer(max_features=5000,ngram_range=(1,2))
X=tv.fit_transform(X)

y=np.array(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

clf = MultinomialNB(alpha=6)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print(accuracy_score(y_test,y_pred))

filename = 'sentiment_model.sav'
pickle.dump(clf, open(filename, 'wb'))
 
loaded_model = pickle.load(open(filename, 'rb'))

print(loaded_model.score(X_test,y_test))
