import pandas as pd

object = pd.read_pickle('preprocessed-latest.pickle')

unique = list()
for o in object:
    if o[0] not in unique:
        unique.append(o)

print(len(unique))

lst = []
