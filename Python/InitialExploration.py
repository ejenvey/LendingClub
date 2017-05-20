# Initial Exploration of the Data

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from collections import Counter

#Importing the dataset
LendingClub0711 = pd.read_csv("/Users/ejenvey/Desktop/Lending Club Data and Analysis/LoanStats_2007-11.csv")

#Null handling
nulls = LendingClub0711.isnull().sum().sort_values(ascending=False)[:25]
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls

#Remove completely null columns


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

#------Data Cleaning/Preprocessing---------
#Create the dependent variable column, bad_loans, where 1 is a Charged Off or Defaulted Loan, and
#1 is a good loan, being paid off
LendingClub0711['bad_loans'] = LendingClub0711.apply(lambda row: 1 if row['loan_status'] == 'Charged Off' or row['loan_status'] == 'Default' else 0, axis=1)

#Subset the data to ONLY those loans that are no longer active
##Create the inactive flag
LendingClub0711['inactive'] = LendingClub0711.apply(lambda row: 1 if row['loan_status'] == 'Fully Paid' or row['loan_status'] == 'Charged Off' or row['loan_status'] == 'Default' else 0, axis=1)
##Filter on the inactive flag
LendingClub0711 = LendingClub0711[LendingClub0711['inactive']==1]

#Explore the target variable (bad_loans)
LendingClub0711.bad_loans.describe()

#Convert target to Category datatype
LendingClub0711['bad_loans'] = LendingClub0711['bad_loans'].astype('category')
LendingClub0711.bad_loans.describe()

#Other Data type conversions and data preprocessing
LendingClub0711['term'] = LendingClub0711['term'].astype('category')
LendingClub0711['id'] = LendingClub0711['id'].astype('category')
LendingClub0711['grade'] = LendingClub0711['grade'].astype('category')
LendingClub0711['sub_grade'] = LendingClub0711['sub_grade'].astype('category')
LendingClub0711['emp_length'] = LendingClub0711['emp_length'].astype('category')
LendingClub0711['home_ownership'] = LendingClub0711['home_ownership'].astype('category')
LendingClub0711['verification_status'] = LendingClub0711['verification_status'].astype('category')
LendingClub0711['loan_status'] = LendingClub0711['loan_status'].astype('category')
LendingClub0711['purpose'] = LendingClub0711['purpose'].astype('category')
LendingClub0711['title'] = LendingClub0711['title'].astype('category')
LendingClub0711['addr_state'] = LendingClub0711['addr_state'].astype('category')
LendingClub0711['policy_code'] = LendingClub0711['policy_code'].astype('category')
LendingClub0711['application_type'] = LendingClub0711['application_type'].astype('category')
LendingClub0711['acc_now_delinq'] = LendingClub0711['acc_now_delinq'].astype('category')
LendingClub0711['addr_state'] = LendingClub0711['addr_state'].astype('category')
LendingClub0711['chargeoff_within_12_mths'] = LendingClub0711['chargeoff_within_12_mths'].astype('category')
LendingClub0711['delinq_amnt'] = LendingClub0711['delinq_amnt'].astype('category')
#LendingClub0711['pub_rec_bankruptcies'] = LendingClub0711['pub_rec_bankruptices'].astype('category')
LendingClub0711['tax_liens'] = LendingClub0711['tax_liens'].astype('category')
LendingClub0711['inactive'] = LendingClub0711['inactive'].astype('category')

LendingClub0711['int_rate'] = LendingClub0711['int_rate'].apply(lambda x: x.strip('%'))
LendingClub0711['int_rate'] = LendingClub0711['int_rate'].astype('float64')
LendingClub0711['loan_amnt'] = LendingClub0711['int_rate'].astype('float64')
LendingClub0711['installment'] = LendingClub0711['installment'].astype('float64')
LendingClub0711['annual_inc'] = LendingClub0711['annual_inc'].astype('float64')
#LendingClub0711['payment_inc_ratio'] = LendingClub0711['payment_inc_ratio'].astype('float64')
LendingClub0711['dti'] = LendingClub0711['dti'].astype('float64')

#select only the numeric features, for numeric plotting and PCA
numeric_features = LendingClub0711.select_dtypes(include=[np.number])

#select only the non-numeric features
categoricals = LendingClub0711.select_dtypes(exclude=[np.number])
categoricals.describe()

#one-hot encoding
LendingClub0711['enc_purpose'] = pd.get_dummies(LendingClub0711.purpose, drop_first=True)
###Insert other columns here

#correlations among the numeric features in the dataset
corr = numeric_features.corr()


X_pca = LendingClub0711.loc[:,['loan_amnt', 'installment', 'int_rate', 'annual_inc',
                               'dti']]
y = LendingClub0711.bad_loans

#For some plots, split the data frame into bad loans and good loans
LendingClub0711bad = LendingClub0711[LendingClub0711['bad_loans']==0]
LendingClub0711good = LendingClub0711[LendingClub0711['bad_loans']==1]

#For plotting
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

#Exploratory Plotting

##Numeric variables
plt.hist(LendingClub0711.loan_amnt, color='blue')
loan_fig = plt.figure()
loan_ax = loan_fig.add_subplot(111)
plt.show()

plt.hist(LendingClub0711.int_rate, color='blue')
plt.show()

annualInc_fig = plt.figure()
annualInc_ax = annualInc_fig.add_subplot(111)
plt.hist(LendingClub0711.annual_inc, color='blue', bins=1000)
annualInc_ax.set_xlim([0,350000])
plt.show()

plt.hist(LendingClub0711.dti, color='blue', bins=40)
plt.show()

##Box Plots to see outliers for certain numeric variables
plt.boxplot(LendingClub0711.loan_amnt)

plt.boxplot(LendingClub0711.int_rate)

plt.boxplot(LendingClub0711.dti)

##Selected Categorical variables bar chart of # records per category
termCounter = Counter(LendingClub0711.term)
plt.bar(range(len(list(termCounter.keys()))),list(termCounter.values()), align='center')
plt.xticks(range(len(list(termCounter.keys()))),list(termCounter.keys()))
plt.show()

c = Counter(LendingClub0711.grade)
plt.bar(range(len(list(c.keys()))),list(c.values()), align='center')
plt.xticks(range(len(list(c.keys()))),list(c.keys()))
plt.show()

c = Counter(LendingClub0711.emp_length)
plt.bar(range(len(list(c.keys()))),list(c.values()), align='center')
plt.xticks(range(len(list(c.keys()))),list(c.keys()))
plt.show()

c = Counter(LendingClub0711.home_ownership)
plt.bar(range(len(list(c.keys()))),list(c.values()), align='center')
plt.xticks(range(len(list(c.keys()))),list(c.keys()))
plt.show()

##Plot the target variable
c = Counter(LendingClub0711.bad_loans)
plt.bar(range(len(list(c.keys()))),list(c.values()), align='center')
plt.xticks(range(len(list(c.keys()))),list(c.keys()))
plt.show()

##Plot selected numeric and categorical variables against the target variable
c = Counter(LendingClub0711.bad_loans)
plt.bar(range(len(list(c.keys()))),list(c.values()), align='center')
plt.xticks(range(len(list(c.keys()))),list(c.keys()))
plt.show()

##Multivariate visualization

##Plots against the business problem, what is the impact of the bad loans?  What would happen if
##the loans were removed?  Recovery fees, etc.

##Proportion plots (hard to figure out how to do in Python)

##ANOVA testing

##Other t-testing

##Principal Components Analysis
pca = PCA(n_components=2)
pca.fit(X_pca,y)
pca.explained_variance_

##
