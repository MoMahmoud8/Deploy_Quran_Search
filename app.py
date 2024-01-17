import pandas as pd
import numpy as np 
import string
import re
import sklearn
import pyarabic.araby as araby
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import arabicstopwords.arabicstopwords as arab_sw # arabic stopwords
from nltk.stem.snowball import ArabicStemmer # Arabic Stemmer gets rot word
# import qalsadi.lemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

st = ArabicStemmer()
# lemmer = qalsadi.lemmatizer.Lemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

import threading
from qalsadi.lemmatizer import Lemmatizer
# Create a thread-local storage object
thread_local = threading.local()
def get_lemmatizer():
    # Retrieve the lemmatizer object for the current thread
    if not hasattr(thread_local, 'lemmatizer'):
        thread_local.lemmatizer = Lemmatizer()
    return thread_local.lemmatizer



# Load the saved vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load the saved corpus_vectorized
with open('corpus_vectorized.pkl', 'rb') as file:
    corpus_vectorized = pickle.load(file)

####################################################

df=pd.read_csv('cleaned_txt_quran.csv')
#print(df.head())

#####################################################
def normalize_chars(txt):
    txt = re.sub("[إأٱآا]", "ا", txt)
    txt = re.sub("ى", "ي", txt)
    txt = re.sub("ة", "ه", txt)
    return txt

stopwordlist = set(list(arab_sw.stopwords_list()) + stopwords.words('arabic'))
stopwordlist = [normalize_chars(word) for word in stopwordlist]


def clean_txt(txt):
    lemmer = get_lemmatizer()

    # remove tashkeel & tatweel
    txt = araby.strip_diacritics(txt)
    txt = araby.strip_tatweel(txt)
    # normalize chars
    txt = normalize_chars(txt)
    # remove stopwords & punctuation
    txt = ' '.join([token.translate(str.maketrans('','',string.punctuation)) for token in txt.split(' ') ])
    # lemmatizer
    #txt_lemmatized = ' '.join([lemmer.lemmatize(token) for token in txt.split(' ')])
    return txt




def show_best_results(df_quran, scores_array, top_n=50):
    sorted_indices = scores_array.argsort()[::-1]
    results = []  # Initialize an empty list to store results
    for position, idx in enumerate(sorted_indices[:top_n]):
        row = df_quran.iloc[idx]
        ayah = row["ayah"]
        ayah_num = row["ayah_num"]
        surah_name = row["surah_name"]
        score = scores_array[idx]
        if score >= 0.01:
            result = {
                "ayah": ayah,
                "ayah_num": ayah_num,
                "surah_name": surah_name,
                "score": score
            }
            results.append(result)

    return results


def run_tfidf(query):
    if not query:
        return []  # Return an empty list if no query is provided

    query = clean_txt(query)
    
    query_vectorized = vectorizer.transform([query])
    scores = query_vectorized.dot(corpus_vectorized.transpose())
    scores_array = scores.toarray()[0]
    results = show_best_results(df, scores_array)
    return results


#####################################################    
#query = "محمد رسول الله"
#run_tfidf(query)
########################################################

from flask import Flask,  render_template
from flask import Flask, request ,render_template



app = Flask(__name__)

@app.route('/', methods=['GET'])
def start_page():
    return render_template('QuranPage.html')



@app.route('/Search system', methods=['POST'])
def predict():
   if request.method == 'POST':
        ayah_text = request.form.get('Ayah')
        results = run_tfidf(ayah_text)
        print(results)
        # if not results:
            # return 'Error: no results found'  # Handle empty results
        if not results:
            error_message = 'لا يوجد نتائج'
            return render_template('QuranPage.html', error_message=error_message)



        return render_template('QuranPage.html', results=results)  # Pass results to the template
    # If it's a GET request or no results yet, render the initial page
   return render_template('QuranPage.html', results=None)






if __name__ == '__main__':

    app.run(port=3000,debug=True,threaded=True)
