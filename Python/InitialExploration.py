# Initial Exploration of the Data

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from collections import Counter
import seaborn as sns
from matplotlib.mlab import PCA
import statsmodels.api as sm
import pylab as pl

#Importing the dataset
LendingClub = pd.read_csv("/Users/ejenvey/Desktop/Lending Club Data and Analysis/LoanStats_keycols_cleaned.csv")

#------Data Exploration---------#

print LendingClub.describe()
print LendingClub.std()

#Explore the target variable (bad_loans)
LendingClub.groupby('bad_loans').mean() 
#above shows a higher loan amount, higher interest rate, lower annual income, lower grade, 
#longer term, renting a house seem to correlate with a bad loan

#correlations among the numeric features in the dataset, plot the correlation matrix
corr = LendingClub.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)
#highlights: number of open accounts is correlated with acc_open_past_24_mths
#also, credit limit is correlated with annual income & revolving balance



X_pca = LendingClub.loc[:,['loan_amnt', 'installment', 'int_rate', 'annual_inc',
                               'dti']]
y = LendingClub.bad_loans

#For some plots, split the data frame into bad loans and good loans
LendingClub0711bad = LendingClub[LendingClub['bad_loans']==0]
LendingClub0711good = LendingClub[LendingClub['bad_loans']==1]

#For plotting
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

#------Exploratory Plots---------#

LendingClub.hist()
plt.show()

##Numeric variables
plt.hist(LendingClub.loan_amnt, color='blue')
loan_fig = plt.figure()
loan_ax = loan_fig.add_subplot(111)
plt.show()

plt.hist(LendingClub.int_rate, color='blue')
plt.show()

annualInc_fig = plt.figure()
annualInc_ax = annualInc_fig.add_subplot(111)
plt.hist(LendingClub.annual_inc, color='blue', bins=1000)
annualInc_ax.set_xlim([0,350000])
plt.show()

plt.hist(LendingClub.dti, color='blue', bins=40)
plt.show()

##Box Plots to see outliers for certain numeric variables
plt.boxplot(LendingClub.loan_amnt)

plt.boxplot(LendingClub.int_rate)

plt.boxplot(LendingClub.dti)

##Selected Categorical variables bar chart of # records per category
termCounter = Counter(LendingClub.term)
plt.bar(range(len(list(termCounter.keys()))),list(termCounter.values()), align='center')
plt.xticks(range(len(list(termCounter.keys()))),list(termCounter.keys()))
plt.show()

c = Counter(LendingClub.grade)
plt.bar(range(len(list(c.keys()))),list(c.values()), align='center')
plt.xticks(range(len(list(c.keys()))),list(c.keys()))
plt.show()

c = Counter(LendingClub.emp_length)
plt.bar(range(len(list(c.keys()))),list(c.values()), align='center')
plt.xticks(range(len(list(c.keys()))),list(c.keys()))
plt.show()

c = Counter(LendingClub.home_ownership)
plt.bar(range(len(list(c.keys()))),list(c.values()), align='center')
plt.xticks(range(len(list(c.keys()))),list(c.keys()))
plt.show()

##Plot the target variable
c = Counter(LendingClub.bad_loans)
plt.bar(range(len(list(c.keys()))),list(c.values()), align='center')
plt.xticks(range(len(list(c.keys()))),list(c.keys()))
plt.show()

##Same Categorical Variables with Loan Status included
pd.crosstab(LendingClub.grade, LendingClub.bad_loans).plot(kind='bar')
plt.title('Grade of Loan by Loan Status')
plt.xlabel('Loan Grade')
plt.ylabel('Frequency')

pd.crosstab(LendingClub.term, LendingClub.bad_loans).plot(kind='bar')
plt.title('Loan Term by Loan Status')
plt.xlabel('Loan Term')
plt.ylabel('Frequency')

pd.crosstab(LendingClub.emp_length, LendingClub.bad_loans).plot(kind='bar')
plt.title('Length of Current Employment by Loan Status')
plt.xlabel('Length of Current Employment')
plt.ylabel('Frequency')

pd.crosstab(LendingClub.home_ownership, LendingClub.bad_loans).plot(kind='bar')
plt.title('Home Ownership Status by Loan Status')
plt.xlabel('Home Ownership Status')
plt.ylabel('Frequency')

##Stacked bars (same categoricals as above)
#CAN I DO A FULL PROPORTION PLOT, where X-axis is also taken into account?

badloan_grade = pd.crosstab(LendingClub.grade, LendingClub.bad_loans)
badloan_grade.div(badloan_grade.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Bad Loan % by Loan Grade')
plt.xlabel('Loan Grade')
plt.ylabel('Percentage')

badloan_term = pd.crosstab(LendingClub.term, LendingClub.bad_loans)
badloan_term.div(badloan_term.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Bad Loan % by Loan Term')
plt.xlabel('Loan Term')
plt.ylabel('Percentage')

badloan_emp_length = pd.crosstab(LendingClub.emp_length, LendingClub.bad_loans)
badloan_emp_length.div(badloan_emp_length.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Bad Loan % by Number Years Employed')
plt.xlabel('Number Years Employed')
plt.ylabel('Percentage')

badloan_home_ownership = pd.crosstab(LendingClub.home_ownership, LendingClub.bad_loans)
badloan_home_ownership.div(badloan_home_ownership.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Bad Loan % by Home Ownership Status')
plt.xlabel('Home Ownership Status')
plt.ylabel('Percentage')

#Loan Amounts by Loan Outcome
LendingClub.pivot_table(index='loan_status',
                                    values='loan_amnt', aggfunc=np.mean).plot(kind='bar', color='blue')
plt.xlabel('Loan Status')
plt.ylabel('Average Loan Amount ($K)')
plt.xticks(rotation=0)
plt.show()

#Loan Amounts by Home Ownership
LendingClub.pivot_table(index='home_ownership',
                                    values='loan_amnt', aggfunc=np.mean).plot(kind='bar', color='blue')
plt.xlabel('Home Ownership Status')
plt.ylabel('Average Loan Amount ($K)')
plt.xticks(rotation=0)
plt.show()

##Plots against the business problem, what is the impact of the bad loans?  What would happen if
##the loans were removed?  Recovery fees, etc.



##ANOVA testing

##Other t-testing

##Principal Components Analysis
pca = PCA(X_pca)
pca.fit(X_pca,y)
pca.explained_variance_

dataMatrix = np.array(LendingClub)   
myPCA = PCA(dataMatrix) 

##
