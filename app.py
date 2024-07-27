import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle

nltk.download('punkt')


df = pd.read_csv('songdata.csv')

df = df.sample(n=5000).drop('link', axis=1).reset_index(drop=True)

df['text'] = df['text'].str.lower().replace(r'[^\w\s]', '', regex=True).replace(r'\n', ' ', regex=True)

stemmer = PorterStemmer()


def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)


df['text'] = df['text'].apply(lambda x: tokenization(x))

tfidvector = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidvector.fit_transform(df['text'])


similarity = cosine_similarity(matrix)


def recommendation(song_title):
    try:
        idx = df[df['song'] == song_title].index[0]
    except IndexError:
        return "Song not found in the dataset."

    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

    songs = []
    for m_id in distances[1:11]:  # Skip the first one as it is the same song
        songs.append(df.iloc[m_id[0]].song)

    return songs



print(recommendation('Alma Mater'))

pickle.dump(similarity,open('similarity.pkl','wb'))
pickle.dump(df,open('df.pkl','wb'))