from pyspark.sql.functions import *
from pyspark.sql.types import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import ast
import re
import nltk
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
REPLACE_BY_SPACE_RE = re.compile('[/(){}—[]|@,;‘?|।!-॥–’-]')


def filter_stopwords():
    stop_words = stopwords.words('english')
    negative_words = ['no', 'not', "don't", "aren't", "couldn't", "didn't", "doesn't", "hadn't", "hasn't",
                      "haven't", "isn't", "mightn't", "mustn't", "needn't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't"]
    for negative_word in negative_words:
        stop_words.remove(negative_word)
    return stop_words


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L


def convertCast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1


def clean_text(sample):
  stop_words = filter_stopwords()
  sample = sample.lower()
  sample = sample.replace("", "")
  sample = REPLACE_BY_SPACE_RE.sub(' ', sample)
  sample = re.sub("[^a-z]+", " ", sample)
  sample = sample.split(" ")
  sample = [word for word in sample if word not in stop_words]
  sample = [lemmatizer.lemmatize(word) for word in sample]
  sample = " ".join(sample)
  return sample

convert_udf = udf(convert)
convertCast_udf = udf(convertCast)
fetch_director_udf = udf(fetch_director)
collapse_udf = udf(collapse)
clean_text_udf = udf(clean_text)