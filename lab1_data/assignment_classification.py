#!/usr/bin/python3
from assignment_features import assignment_features
from sklearn.svm import SVC
import sklearn
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def assignment_classification(plot=False):
	## Load the features (requires assignment 1 to be completed)
	if Path('assignment_features.npy').is_file():
		data = np.load('assignment_features.npy', allow_pickle=True).item()
	else:
		data = assignment_features()

	# Store classifiers in the data dict as well
	data["features"]["int"]["classifier"] = {}
	data["features"]["hog"]["classifier"] = {}
	data["features"]["cnn"]["classifier"] = {}

	## Exercise 2.1: Train the svm on all three feature sets
	# Complete the code below
	# Save the resulting trained svm in the dictionary
	# Use kernel='rbf' and C=100
	kernel='rbf'
	C=2
	#data["int"]["classifier"]["svm"] = <TRAINED_SVM_ON_INT>
	#data["hog"]["classifier"]["svm"] = <TRAINED_SVM_ON_INT>
	#data["cnn"]["classifier"]["svm"] = <TRAINED_SVM_ON_INT>
# #YOUR_CODE_HERE

	# Check if the SVM is stored and can be used in later parts of the assignment
	if not 'svm' in data["features"]["int"]["classifier"]:
		print('store the trained svm in data["features"]["int"]["classifier"]["svm"], exiting..')
		sys.exit(-1)
	if not 'svm' in data["features"]["hog"]["classifier"]:
		print('store the trained svm in data["features"]["hog"]["classifier"]["svm"], exiting..')
		sys.exit(-1)
	if not 'svm' in data["features"]["cnn"]["classifier"]:
		print('store the trained svm in data["features"]["cnn"]["classifier"]["svm"], exiting..')
		sys.exit(-1)

	## Exercise 2.2: Train the k-NN classifier on all three feature sets with k = 1
	# Complete the code below
	# Save the resulting trained classifier in the dictionary
	#data["int"]["classifier"]["knn"] = <TRAINED_SVM_ON_INT>
	#data["hog"]["classifier"]["knn"] = <TRAINED_SVM_ON_INT>
	#data["cnn"]["classifier"]["knn"] = <TRAINED_SVM_ON_INT>
# #YOUR_CODE_HERE

	# Check if the k-NN classifier is stored and can be used in later parts of the assignment
	if not 'knn' in data["features"]["int"]["classifier"]:
		print('store the trained knn in data["features"]["int"]["classifier"]["knn"], exiting..')
		sys.exit(-1)
	if not 'knn' in data["features"]["hog"]["classifier"]:
		print('store the trained knn in data["features"]["hog"]["classifier"]["knn"], exiting..')
		sys.exit(-1)
	if not 'knn' in data["features"]["cnn"]["classifier"]:
		print('store the trained knn in data["features"]["cnn"]["classifier"]["knn"], exiting..')
		sys.exit(-1)


	## Exercise 2.3: Plot the confusion matrices for all feature and classifier combinations (6 in total)
	# You will need to complete the code below
	if plot:
		fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,10))
		for i, (feat, items) in enumerate(data["features"].items()):
			for j, (classifier, model) in enumerate(items["classifier"].items()):
				index = j + i * len(items["classifier"].keys())
				ax = axes.flatten()[index]
# #YOUR_CODE_HERE
		plt.suptitle('Confusion matrices for the feature/classifier combinations')
		plt.show()

	## Exercise 2.4: Construct a new test set by adding an intensity of 30 to the original test set
	# Recalculate and plot the confusion matrices
# #YOUR_CODE_HERE

	## Exercise 2.5: Apply PCA to reduce the dimensionality to 20
	# Use sklearn.decomposition.PCA
	# Recompute and plot the confusion matrices for all feature and classifier combinations (6 in total)
	# Take a look at the code of the previous exercises and use the relevant parts to complete this exercise
	if plot:
		fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,10))
		for i, (feat, items) in enumerate(data["features"].items()):
			for j, (classifier, model) in enumerate(items["classifier"].items()):
				index = j + i * len(items["classifier"].keys())
				ax = axes.flatten()[index]
# #YOUR_CODE_HERE
		plt.suptitle('Confusion matrices after applying dimensionality reduction using PCA')
		plt.show()


	## Exercise 2.6: Evaluate the accuracy_score for varying values of k of the k-NN
	# Plot the accuracy_score against the k parameter
	if plot:
# #YOUR_CODE_HERE


	## Exercise 2.7: Evaluate the accuracy_score for varying values of C of the SVM
	# Plot the accuracy_score against the k parameter
	if plot:
# #YOUR_CODE_HERE

	## Exercise 2.8: Plot ROC curves
	# Create a single plot with the three ROC curves (one for each feature type) with the SVM
	if plot:
# #YOUR_CODE_HERE

	# Save the results to disk to use in later exercises
	np.save('assignment_classification.npy', data)

	return data

if __name__ == '__main__':
	assignment_classification(plot=True)
