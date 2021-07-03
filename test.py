import os	
import sys
import math
import scipy.fft 
import joblib
import pathlib
import warnings
import numpy as np
import pandas as pd
import scipy.io.wavfile
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
from collections import namedtuple
from prettytable import PrettyTable

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

x = []
y = []

genres = os.listdir('./gtzan')
test_file = sys.argv[1]

pathlib.Path(f'./testing{test_file[:-4]}').mkdir(parents=True, exist_ok=True)
os.system("ffmpeg -t 30 -i " + test_file + f' ./testing{test_file[:-4]}/music.wav')
sample_rate, song_array = scipy.io.wavfile.read(f'./testing{test_file[:-4]}/music.wav')
fft_features = abs(scipy.fft.fft(song_array[:10000]))

for label,genre in enumerate(genres):
	y.append(label)
	
x.append(fft_features)
x = np.array(x)

if x.ndim == 3:
    x = x.reshape((x.shape[0]*x.shape[1]), x.shape[2])
    x = x.transpose()

clf = joblib.load('./model.pkl')
probs = clf.predict_proba(x)

print("\n")

list_genres = []
list_prob_genres = []

for x in genres:
    list_genres.append(x)

for x in probs[0]:
    list_prob_genres.append(float("{0:.4f}".format(x)))

list_prob_index =[item for item in range(1,len(list_genres)+1)]

my_table = PrettyTable()

my_table.field_names = ["Prob. index","Genres", "% Probablity"]
for x, y,z in zip(list_prob_index,list_genres, ([i + " %" for i in [str(i)for i in [i*100 for i in list_prob_genres]]])):
    my_table.add_row([x,y,z])

print(my_table)

print("\n")
#for row in probs:
    #print(*row)

probs=probs[0]
max_prob = max(probs)


for i,j in enumerate(probs):
    if probs[i] == max_prob:
        max_prob_index=i
   
print("Maximum probability index: ",max_prob_index+1)
predicted_genre = genres[max_prob_index]
print("\nTherefore, Predicted Genre = ",predicted_genre,"\n")
os.system("rm -r "+ f'testing*')