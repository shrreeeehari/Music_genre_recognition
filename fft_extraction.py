import os
import scipy
from scipy import *
import scipy.io.wavfile
import numpy as np
import pathlib
import warnings
warnings.filterwarnings('ignore')

genres = os.listdir('./gtzan')
for i in genres:
    pathlib.Path(f'extracted_fft/{i}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'./gtzan/{i}'):
        songname = f'./gtzan/{i}/{filename}'
        sample_rate, song_array = scipy.io.wavfile.read(songname)
        fft_features = abs(scipy.fft.fft(song_array[:10000]))
        np.save(f'extracted_fft/{i}/{filename[:-3].replace(".", "")}.fft', fft_features)
    
    

