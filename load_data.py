import pickle
import re
import string
import urllib.request
from collections import Counter

import demoji
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Comment
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(url, body):
    soup = BeautifulSoup(body, 'html.parser')
    if "github.com" in url:
            soup = soup.find("div", {"id": "readme"})
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return [t.strip() for t in visible_texts]

data = []
df = pd.read_csv("/Users/beyzacevik/Downloads/dataset-latest2.csv")

for index, row in df.iterrows():
    url = row["url"]
    try:
        req = urllib.request.Request(url, data=None, 
                                        headers={
                                                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
                                                })

        html = urllib.request.urlopen(url).read()
        raw_sequences = text_from_html(url, html)
        seq = [row["name"], row["cat"], " ".join(raw_sequences)]
    except Exception as ex:
        print(ex)
        seq = [row["name"], row["cat"], str(ex)]
        print(row)
    data.append(seq)

with open("raw_data.pickle", 'wb') as target:
    pickle.dump(data, target)

stop_words = stopwords.words("english")
lemmatizer=WordNetLemmatizer()
for d in data:
    #Remove emoloji
    d[2] = demoji.replace(d[2], repl="")
    #Convert lowercase
    d[2] = d[2].lower()
    #Remove numbers
    d[2] = re.sub(r"\d+", "", d[2])
    #Remove punctuation
    d[2] = d[2].translate(str.maketrans('', '', string.punctuation))
    #Remove whitespaces
    d[2] = d[2].strip()
    #Stop word elimination
    tokens = word_tokenize(d[2])
    tokens = [i for i in tokens if not i in stop_words]
    #Lemmatization
    tokens = [lemmatizer.lemmatize(i) for i in tokens]
    d[2] = tokens

with open("preprocessed-latest.pickle", 'wb') as target:
    pickle.dump(data, target)

