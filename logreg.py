import os
import sys
import glob
import joblib
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def read_fft(list_genre, base_dir):
	x = []
	y = []
	for label, genre in enumerate(list_genre):
		genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
		file_list = glob.glob(genre_dir)
		
		for file in file_list:
			fft_features = np.load(file)
			x.append(fft_features)
			y.append(label)
	
	return np.array(x), np.array(y)


def logreg(x_train, y_train, x_test, y_test, list_genre):

	print("x_train = " + str(len(x_train)), "y_train = " + str(len(y_train)), "x_test = " + str(len(x_test)), "y_test = " + str(len(y_test)))
	logreg_classifier = (linear_model.LogisticRegression(max_iter=350)).fit(x_train,y_train)
	
	prediction = logreg_classifier.predict(x_test)
	matrix_confusion = confusion_matrix(y_test, prediction)
	accuracy = accuracy_score(y_test, prediction)
	
	print("\n\nlogistic accuracy = " + str(accuracy))
	print("\nlogistic confusion matrix :\n",matrix_confusion)
	joblib.dump(logreg_classifier, 'model.pkl')
	print("Model Saved\n")
	
	show_cm(matrix_confusion, "Confusion matrix", list_genre)


def show_cm(confusion_matrix, title, list_genre, cmap=plt.cm.Blues):
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(list_genre))
    plt.xticks(ticks, list_genre, rotation=45)
    plt.yticks(ticks, list_genre)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_fft.png')
	
    print("Confusion Matrix saved")
    

def main():
	base_dir_fft  = sys.argv[1]
	list_genre = os.listdir('./gtzan')
	

	# using FFT
	x, y = read_fft(list_genre, base_dir_fft)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
	
	print('\nUsing FFT')
	logreg(x_train, y_train, x_test, y_test, list_genre)


if __name__ == "__main__":
	main()