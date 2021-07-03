# Music_genre_recognition

I have used Logistic Regresssion in this project which is a Machine Learning classification algorithm that is often used to predict the probability of a categorical dependent variable, in this case to predict the genre of a given input song sample. <br>
It measures the relationship between the categorical dependent variable and one or more independent variables by estimating the probability of occurrence of an event using its logistics function. Logistic Regression is a linear classifier which is generally meant for binary classification tasks. For this multi-class classification task, Logistic Regression is implemented as a one-vs-rest method. That is, 10 separate binary classifiers are trained. During test time, the class with the highest probability from among the 10 classifiers is chosen as the predicted class.

<h2>Sequence of execution</h2>
<h3>Extracting FFT</h3>
Running fft_extraction.py generates the fft values for all wav audio files and stores them as numpy files in ./extracted_fft
