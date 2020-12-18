import pickle
from model import run

filename = "preprocessed-latest.pickle"
with open(filename, 'rb') as target:
    data = pickle.load(target)

data = [d for d in data if len(d[2]) > 10]
categories = ['ADVERTISEMENTS', 'ANALYTICS','WRITE AND READ', 'MEDIA']
data = [d for d in data if d[1] in categories]
run(data)

