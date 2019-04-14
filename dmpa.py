import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.pyplot as mp
from sklearn import tree, metrics, model_selection
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB

def giniIndex(dataFrame):
	
	#print(dataFrame.head())
	#print(dataFrame.info())

	#all values are non-null objects, but we need it in integer format for performing analysis on it for training the decision tree
	dataFrame['Class'],class_values = pd.factorize(dataFrame['Class'])
	#print(class_values) #to see what values of class are there. 2 represents benign, 4 represents malignant

	#print(dataFrame['Class'].unique())
	#corr = dataFrame.corr()
	'''fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
	fig.colorbar(cax)
	ticks = np.arange(0,len(dataFrame.columns),1)
	ax.set_xticks(ticks)
	plt.xticks(rotation=90)
	ax.set_yticks(ticks)
	ax.set_xticklabels(dataFrame.columns)
	ax.set_yticklabels(dataFrame.columns)
	plt.show()'''


	#bareNuclei only has 683 values instead of 699. So attempting to smooth the values by replacing null values by mean of the attribute
	#print(dataFrame.isnull().values.any())

	temp=0
	temp = dataFrame['bareNuclei'].sum()
	temp = int(temp/683) #to get mean of values
	temp = float(temp)

	for i in dataFrame['bareNuclei']:
		if i==np.NaN:
			i=temp
	
	dataFrame['bareNuclei'].fillna(temp, inplace=True) #smoothing all missing values by mean
	#print(dataFrame['bareNuclei'])

	col_corr = set() # Set of all the names of deleted columns

	#to remove redundant attributes
	corr_matrix = dataFrame.corr()
	for i in range(len(corr_matrix.columns)-1):
		for j in range(i):
			if (corr_matrix.iloc[i, j] >= 0.9 and corr_matrix.columns[j] not in col_corr):
				colname = corr_matrix.columns[i]
				col_corr.add(colname)
				if colname in dataFrame.columns:
					del dataFrame[colname]

	#next, we need to select prediction variables (that will help in classification), and the labels given to those variable choices.
	x = dataFrame.iloc[:,:-1] #all attributes except class
	y = dataFrame.iloc[:,-1] #only last attribute

	#every supervised predictor has training data and testing data. let us split our data into those two fragments.
	x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.3,random_state=0) #test data is 30% of all data

	#print(x_test)

	dTree = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None, random_state=0) #need to find optimum value for max_depth of 		tree
	dTree.fit(x_train, y_train)

	#now to get the predicted value for test variables
	y_pred = dTree.predict(x_test)

	print("Using the batch method: ")
	kfold = KFold(n_splits=4, random_state=4)
	cv_results=cross_val_score(dTree, x_test, y_test, cv=kfold)

	print(cv_results.mean()*100, "%")

	print("Using the sequential method: ")
	#now to check performance metrics and see how many mismatches
	count_misclassified = (y_test!=y_pred).sum()
	print("Misclassified samples: {}".format(count_misclassified))
	
	accuracy = metrics.accuracy_score(y_test, y_pred)
	print("Accuracy: {:.2f}".format(accuracy))

	#return accuracy


def svmRBFN(dataFrame):
	dataFrame['Class'],class_values = pd.factorize(dataFrame['Class'])

	temp=0
	temp = dataFrame['bareNuclei'].sum()
	temp = int(temp/683) #to get mean of values
	temp = float(temp)

	dataFrame['bareNuclei'].fillna(temp, inplace=True) #smoothing all missing values by mean

	col_corr = set() # Set of all the names of deleted columns
	corr_matrix = dataFrame.corr()

	#to remove redundant attributes
	for i in range(len(corr_matrix.columns)-1):
		for j in range(i):
			if (corr_matrix.iloc[i, j] >= 0.9 and corr_matrix.columns[j] not in col_corr):
				colname = corr_matrix.columns[i]
				col_corr.add(colname)
				if colname in dataFrame.columns:
					del dataFrame[colname]

	x = dataFrame.iloc[:,:-1] #all attributes except class
	y = dataFrame.iloc[:,-1] #only last attribute

	#every supervised predictor has training data and testing data. let us split our data into those two fragments.
	x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.3,random_state=0) #test data is 30% of all data
	#print(x_test)
	params = {"C": [0.1,1,10,100,1000], "gamma": [100,10,1,0.1,0.01,0.001]}

	#Making the RBFN model. gamma=1/numberOfFeatures, C is the penalty for error of prediction, RBFN kernel being used in the SVM
	RBFNModel = svm.SVC(C=100, gamma=100, kernel='rbf', random_state=0)

	grid_search = GridSearchCV(RBFNModel, params) #tuning hyperparamters of the RBFN model
	RBFNModel.fit(x_train, y_train) #fitting data for the model

	y_pred = RBFNModel.predict(x_test)

	#Using the batch method of training
	print("Using the batch method: ")
	kfold = KFold(n_splits=4, random_state=4)
	cv_results=cross_val_score(RBFNModel, x_test, y_test, cv=kfold)

	print(cv_results.mean()*100, "%")

	#clf = GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, n_jobs=-1)

	print("Using the sequential method: ")
	#now to check performance metrics and see how many mismatches
	count_misclassified = (y_test!=y_pred).sum()
	print("Misclassified samples: {}".format(count_misclassified))
	
	accuracy = metrics.accuracy_score(y_test, y_pred)
	print("Accuracy: {:.2f}".format(accuracy))

	#return accuracy


def naivebayes(dataFrame):

	dataFrame['Class'],class_values = pd.factorize(dataFrame['Class'])

	temp=0
	temp = dataFrame['bareNuclei'].sum()
	temp = int(temp/683) #to get mean of values
	temp = float(temp)

	dataFrame['bareNuclei'].fillna(temp, inplace=True) #smoothing all missing values by mean

	col_corr = set() # Set of all the names of deleted columns
	corr_matrix = dataFrame.corr()

	#to remove redundant attributes
	for i in range(len(corr_matrix.columns)-1):
		for j in range(i):
			if (corr_matrix.iloc[i, j] >= 0.8 and corr_matrix.columns[j] not in col_corr):
				colname = corr_matrix.columns[i]
				col_corr.add(colname)
				if colname in dataFrame.columns:
					del dataFrame[colname]


	x = dataFrame.iloc[:,:-1] #all attributes except class
	y = dataFrame.iloc[:,-1] #only last attribute

	#every supervised predictor has training data and testing data. let us split our data into those two fragments.
	x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.3,random_state=0) #test data is 30% of all data

	naiveBayes = GaussianNB() #creating a naive bayes model

	naiveBayes.fit(x_train, y_train)
	y_pred = naiveBayes.predict(x_test)

	print("Using the batch method: ")
	kfold = KFold(n_splits=4, random_state=4)
	cv_results=cross_val_score(naiveBayes, x_test, y_test, cv=kfold)

	print(cv_results.mean()*100, "%")

	print("Using the sequential method: ")
	count_misclassified = (y_test!=y_pred).sum()
	print("Misclassified samples: {}".format(count_misclassified))
	
	accuracy = metrics.accuracy_score(y_test, y_pred)
	print("Accuracy: {:.2f}".format(accuracy))

	#return accuracy


dataFrame = pd.read_csv('dataset.csv', names = ['ID','clumpThickness', 'cellSize', 'cellShapeUniformity', 
						'marginalAdhesion', 'singleEpithelialCellSize', 'bareNuclei', 'blandChromatin','normalNucleoli', 'Mitoses',
						'Class'])


print("\n***** Using a decision tree: *****")
giniIndex(dataFrame)

print("\n***** Using the RBFN Kernel of an SVM *****\n")
svmRBFN(dataFrame)

print("\n***** Using the Naive Bayes Classifier *****\n")
naivebayes(dataFrame)