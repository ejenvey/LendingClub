import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from collections import Counter
import seaborn as sns
from matplotlib.mlab import PCA
import statsmodels.api as sm
import pylab as pl

#Read in the cleaned dataset (ensure the data types match)


#Create the independent (X) and dependent (y) vectors
X = LendingClub.iloc[:,LendingClub.columns != 'bad_loans']#.values
y = LendingClub.iloc[:,LendingClub.columns == 'bad_loans']#.values

#X_numeric = numeric_features.iloc[:,numeric_features.columns != 'bad_loans']#.values
#y_numeric = numeric_features.iloc[:,numeric_features.columns == 'bad_loans']#.values

#X = pd.DataFrame(X)
#y = pd.DataFrame(y)

#Splitting the dataset into training and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.2, random_state=123)


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


#Fitting Logistic Regression Model to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#Examine the coefficients
pd.DataFrame(zip(X_numeric.columns, np.transpose(classifier.coef_)))

#Show our R^2 value
print ("R^2 is: \n", classifier.score(X_test, y_test))

#Predict the Test set results, and append to the test set
y_pred = classifier.predict(X_test)
X_test['actuals'] = y_test
X_test['predicted'] = y_pred

#See the root-mean-squared-error
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, y_pred))

# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print scores
print scores.mean()

#plot actuals vs predicted
#plt.scatter(y_pred, y_test, alpha=.75,
#            color='b') 
#plt.xlabel('Predicted')
#plt.ylabel('Actual')
#plt.title('Logistic Regression Model')
#plt.show()

#Creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
