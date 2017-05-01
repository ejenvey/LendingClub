# Initial Exploration of the Data

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
LendingClub0711 = pd.read_csv("/Users/ejenvey/Desktop/Lending Club/LoanStats_2007-11.csv")

#Remove columns with little-to-no information

noInfoColumns = []

for i in range(len(LendingClub0711.columns)):
    if len(LendingClub0711.iloc[:,i].unique()) == 1:
        noInfoColumns.append(i)

infoColumns = [item for item in range(len(LendingClub0711)) if item not in noInfoColumns and item <111]

LendingClub0711 = LendingClub0711.iloc[:,infoColumns]

#Check summary level information for dataset
LendingClub0711.describe()

LendingClub0711.chargeoff_within_12_mths.unique()

y = dataset.iloc[:,3].values

#Splitting the dataset into training and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.2, random_state=0)


#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

