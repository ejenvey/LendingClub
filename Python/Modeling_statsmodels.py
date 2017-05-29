

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

#Read in the cleaned dataset (ensure the data types match)
LendingClub = pd.read_csv("/Users/ejenvey/Desktop/Lending Club Data and Analysis/LoanStats_keycols_cleaned.csv")


LendingClub['intercept'] = 1.0

#Create the independent (X) and dependent (y) vectors
X = LendingClub.iloc[:,LendingClub.columns != 'bad_loans']#.values
y = LendingClub.iloc[:,LendingClub.columns == 'bad_loans']#.values


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
import sklearn.metrics as metric
cm = confusion_matrix(y_test,y_pred_binary)

#Calculate Precision/Recall to measure the model

#Precision = TruePositive/(TruePositive + FalsePositive)
metric.precision_score(y_test,y_pred_binary)

#Recall = TruePositive/(TruePositive + FalseNegative)
metric.recall_score(y_test,y_pred_binary)

#-----Model Improvement--------
#The above model is fairly poor, noted by its low recall and middling precision

LendingClub = LendingClub.drop(['grade_B','grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G'], axis=1)

"""
#Feature scaling
list(X_train.select_dtypes(include=[np.number]).columns.values)

for y in X_train.columns:
    print X_train[y].dtype

X_train.apply(pd.to_numeric)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

