#Lending Club Analysis

#Eric Jenvey

#The following code is the third in a series of files that processes, visualizes, and models publicly available data from
#a peer-to-peer lending company called Lending Club.  The purpose of the analysis is to predict the probability that a loan goes
#into default, using only the data we know about the loan and the borrower at the time the loan is requested.

#This file builds both a Logistic Regression and Decision Tree model to predict 
#the probability of loan default, then evaluates these models using classic evaluation 
#metrics, Precision and Recall.  Future work here will include improving the models 
#as well as employing a set of other classification models.

#I would like to give credit to Andrew Bruce for some of the general framework of this analysis, his post about this dataset +
#analysis can be found at https://turi.com/learn/gallery/notebooks/predict-loan-default.html

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from collections import Counter
import seaborn as sns
from matplotlib.mlab import PCA
import statsmodels.api as sm
import pylab as pl
import math
import sklearn.metrics as metric

#Read in the cleaned dataset (ensure the data types match)
LendingClub = pd.read_csv("LoanStats_keycols_cleaned.csv")


LendingClub['intercept'] = 1.0

#Create the independent (X) and dependent (y) vectors
X = LendingClub.iloc[:,LendingClub.columns != 'bad_loans'].values
y = LendingClub.iloc[:,LendingClub.columns == 'bad_loans'].values


#Splitting the dataset into training and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.2, random_state=123)

#Fitting Logistic Regression Model to the Training Set
logit = sm.Logit(y_train, X_train)
result = logit.fit()

#print out the summary of the model, which shows the coefficients for each independent
#variable, along with the confidence interval for that coefficient and the corresponding
#z-score
print result.summary()

#convert the coefficients from log odds to odds, shows us how a 1-unit increase/decrease
#in one of these independent variables changes the odds of a loan being in default
print np.exp(result.params)

#use the model to predict whether a loan will default or not from the test set
y_pred = result.predict(X_test)

y_pred_binary = []

#this turns the continuous prediction to a binary 1/0, given an arbitrary threshold
#of 0.5
for i in y_pred:
    if i > 0.5:
        y_pred_binary.append(1)
    else:
        y_pred_binary.append(0)

#Creating a confusion matrix
cm_logit = confusion_matrix(y_test,y_pred_binary)

#Calculate Precision/Recall to measure the model

#Precision = TruePositive/(TruePositive + FalsePositive)
logit_precision = metric.precision_score(y_test,y_pred_binary)

#Recall = TruePositive/(TruePositive + FalseNegative)
logit_recall = metric.recall_score(y_test,y_pred_binary)

#-----Model Improvement--------
#The above model is fairly poor, noted by its low recall and middling precision

LendingClub = LendingClub.drop(['grade_B','grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G'], axis=1)

#----Decision Tree Classification-------
# Fitting the Decision Tree Classifier to the Training Set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)

#Predict the Test set results
y_pred = classifier.predict(X_test)

#Creating a confusion matrix
cm_decisiontree = confusion_matrix(y_test,y_pred)

#Calculate Precision/Recall to measure the model

#Precision = TruePositive/(TruePositive + FalsePositive)
decisiontree_precision = metric.precision_score(y_test,y_pred)

#Recall = TruePositive/(TruePositive + FalseNegative)
decisiontree_recall = metric.recall_score(y_test,y_pred)


