import pickle
from sklearn.utils import shuffle
from model import run
import pandas as pd
filename = "preprocessed-latest.pickle"
with open(filename, 'rb') as target:
    data = pickle.load(target)

data  = shuffle([d for d in data if len(d[2]) > 10])
eliminated_categories = ["ANALYSE", "NETWORK", "TEXTTOSPEECH", "USB","SECURITY"]
#eliminated_categories = ["AUDIO", "PERMISSIONS", "MEDIA", "DATABASE", "PURCHASE"]
data = [d for d in data if d[1] not in eliminated_categories]


def conv(x):
    if x in "LOCALIZATION" or x in "PERMISSIONS":
        return "5"
    elif x in "AUDIO" or x in "CAMERA" or x in "MEDIA" or x in "FACE RECOGNITION":
        return "6"
    elif x in "FILE_SYSTEM" or x in "STORAGE" or x in "CLOUD STORAGES" or x in "DATABASE":
        return "7"
    elif x in "BLUETOOTH":
        return "3"
    elif x in "LOCATION":
        return "4"
    elif x in "DATE & TIME PICKERS":
        return "8"
    elif x in "MAIL" or x in "MESSAGE":
        return "9"
    elif x in "PURCHASE":
        return "10"
    elif x in "SENSOR":
        return "11"
    elif x in "ADVERTISEMENTS":
        return "1"
    elif x in "ANALYTICS":
        return "2"
    else:
        return None

cat = set([d[1] for d in data])
num = [0 for a in range(24)]
dic = dict(zip(cat,num))
list=[]
count =0
data = [d for d in data if d[1] not in eliminated_categories]
for i in data:
    if dic[i[1]] < 50:
        dic[i[1]]+=1
        list.append(count)
        count+=1
        temp =data[count-1][1]
        data[count-1][1]=conv(data[count-1][1])
        print(temp," ",data[count-1][1])

data = [data[i] for i in list]



run(data)
