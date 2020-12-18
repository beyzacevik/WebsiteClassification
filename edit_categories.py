import pandas as pd
import numpy as np

dataset = pd.read_csv("/Users/beyzacevik/Downloads/dataset-latest.csv")

util = ['LOCALIZATION', 'PERMISSIONS']
media = ['AUDIO', 'CAMERA', 'MEDIA' , 'FACE RECOGNITION']
write_read = ['FILE_SYSTEM' , 'STORAGE' , 'CLOUD STORAGES' ,'DATABASE']
message = ['MAIL', 'MESSAGE']

will_be_removed = ['ANALYSE','NETWORK','TEXTTOSPEECH','USB']

for cat in will_be_removed:
    dataset = dataset[dataset.cat != cat]

for index, row in dataset.iterrows():

    if dataset.loc[index, 'cat'] in util:
        dataset.loc[index, 'cat'] = 'UTIL'

    if dataset.loc[index, 'cat'] in media:
        dataset.loc[index, 'cat'] = 'MEDIA'

    if dataset.loc[index, 'cat'] in write_read:
        dataset.loc[index, 'cat'] = 'WRITE AND READ'

    if dataset.loc[index, 'cat'] in message:
       dataset.loc[index, 'cat'] = 'MESSAGE'


dataset.to_csv('/Users/beyzacevik/Downloads/dataset-latest2.csv')