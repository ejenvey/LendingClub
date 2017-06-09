#Lending Club Analysis

#Eric Jenvey

#The following code is the first in a series of files that processes, visualizes, and models publicly available data from
#a peer-to-peer lending company called Lending Club.  The purpose of the analysis is to predict the probability that a loan goes
#into default, using only the data we know about the loan and the borrower at the time the loan is requested.

#This specific file imports the data, combines the various years of data, then
#cleans and processes the data to be ready for exploration and analysis. This data can
#be found at https://www.lendingclub.com/info/download-data.action. For brevity,
#code below reads from the local directory that contains this code file.

#I would like to give credit to Andrew Bruce for some of the general framework of this analysis, his post about this dataset +
#analysis can be found at https://turi.com/learn/gallery/notebooks/predict-loan-default.html

#-----------------------------------Data Preprocessing--------------------------

#Importing the libraries needed for this analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from collections import Counter
import seaborn as sns
from matplotlib.mlab import PCA
import statsmodels.api as sm
import pylab as pl
import zipfile
import urllib2
import csv

#---------Import the data------------

#Import data from 2007-2016
LendingClub0711 = pd.read_csv("LoanStats_2007-11.csv")
LendingClub1213 = pd.read_csv("LoanStats_2012-13.csv")
LendingClub14 = pd.read_csv("LoanStats_2014.csv")
LendingClub15 = pd.read_csv("LoanStats_2015.csv")
LendingClubQ1 = pd.read_csv("LoanStats_2016Q1.csv")
LendingClubQ2 = pd.read_csv("LoanStats_2016Q2.csv")
LendingClubQ3 = pd.read_csv("LoanStats_2016Q3.csv")
LendingClubQ4 = pd.read_csv("LoanStats_2016Q4.csv")

#Combine the data into one Data Frame
frames = [LendingClub0711, LendingClub1213, LendingClub14, LendingClub15, LendingClubQ1, LendingClubQ2, LendingClubQ3, LendingClubQ4]
LendingClub = pd.concat(frames)

#Remove the singular frames from session
del LendingClub0711, LendingClub1213, LendingClub14, LendingClub15, LendingClubQ1, LendingClubQ2, LendingClubQ3, LendingClubQ4


#Subset the data to ONLY those loans that are no longer active, as we must know
#the final outcome of the loan in order to train a model that will predict
#a loan's outcome. We'll create a flag based on loan_status to do so
LendingClub['inactive'] = LendingClub.apply(lambda row: 1 if row['loan_status'] == 'Fully Paid' or row['loan_status'] == 'Charged Off' or row['loan_status'] == 'Default' else 0, axis=1)
LendingClub = LendingClub[LendingClub['inactive']==1]

#Write the full dataset out to CSV for future reference
LendingClub.to_csv("LoanStats_all.csv",index=False)


#---------Data Cleaning------------

#Remove null columns or columns with only one unique value
noInfoColumns = []

for i in range(len(LendingClub.columns)):
    if len(LendingClub.iloc[:,i].dropna().unique()) <= 1:
        noInfoColumns.append(i)

infoColumns = [item for item in range(len(LendingClub)) if item not in noInfoColumns and item <111]

LendingClub = LendingClub.iloc[:,infoColumns]

#Remaining Columns with null values
nulls = LendingClub.isnull().sum().sort_values(ascending=False)[:50]
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls

#Remove columns with very high (i.e., ~>90%) occurrences of NULL
LendingClub = LendingClub.drop(['dti_joint', 'annual_inc_joint','desc'], axis=1)

#Handle missing values in other columns - UPDATE TO INTERPOLATE
#data = train.select_dtypes(include=[np.number]).interpolate().dropna() 

#Drop any variables that, after consulting the data dictionary, are either unknown at loan origination
#or are not relevant to the problem
LendingClub = LendingClub.drop(['initial_list_status', 'installment',
'last_pymnt_amnt', 'recoveries', 'title', 'total_pymnt','total_pymnt_inv','total_rec_int',
'total_rec_late_fee','total_rec_prncp','zip_code'],axis=1)

#for now, drop any non-encodable non-numeric variables
#future idea: find if there is a certain time period that has greater occurrence
#of loan defaults than others, could create a coded feature for that
LendingClub = LendingClub.drop(['issue_d', 'last_credit_pull_d', 'last_pymnt_d', 
                                        'earliest_cr_line'], axis=1)

#Data type conversions
LendingClub['term'] = LendingClub['term'].astype('category')
LendingClub['grade'] = LendingClub['grade'].astype('category')
LendingClub['sub_grade'] = LendingClub['sub_grade'].astype('category')
LendingClub['home_ownership'] = LendingClub['home_ownership'].astype('category')
LendingClub['verification_status'] = LendingClub['verification_status'].astype('category')
LendingClub['loan_status'] = LendingClub['loan_status'].astype('category')
LendingClub['purpose'] = LendingClub['purpose'].astype('category')
LendingClub['addr_state'] = LendingClub['addr_state'].astype('category')
LendingClub['acc_now_delinq'] = LendingClub['acc_now_delinq'].astype('category')
LendingClub['addr_state'] = LendingClub['addr_state'].astype('category')
LendingClub['delinq_amnt'] = LendingClub['delinq_amnt'].astype('category')
LendingClub['tax_liens'] = LendingClub['tax_liens'].astype('category')
LendingClub['emp_title'] = LendingClub['emp_title'].astype('category')
LendingClub['loan_amnt'] = LendingClub['loan_amnt'].astype('float64')
LendingClub['annual_inc'] = LendingClub['annual_inc'].astype('float64')
LendingClub['dti'] = LendingClub['dti'].astype('float64')

#Employment length takes a bit more processing to convert to a float. Will 
#convert 10+ years to 10, <1 year to 1, and n/a to 0
LendingClub['emp_length'] = LendingClub['emp_length'].apply(lambda x: x.replace('years', ''))
LendingClub['emp_length'] = LendingClub['emp_length'].apply(lambda x: x.replace('year', ''))
LendingClub['emp_length'] = LendingClub['emp_length'].apply(lambda x: x.replace('<', ''))
LendingClub['emp_length'] = LendingClub['emp_length'].apply(lambda x: x.replace('+', ''))
LendingClub['emp_length'] = LendingClub['emp_length'].apply(lambda x: x.replace('n/a', '0'))
LendingClub['emp_length'] = LendingClub['emp_length'].astype('float64')

#Both interest rate and revolving credit utilization will drop the % and convert to 
#a float
LendingClub['int_rate'] = LendingClub['int_rate'].astype('string')
LendingClub['int_rate'] = LendingClub['int_rate'].apply(lambda x: x.strip('%'))
LendingClub['int_rate'] = LendingClub['int_rate'].astype('float64')

LendingClub['revol_util'] = LendingClub['revol_util'].astype('string')
LendingClub['revol_util'] = LendingClub['revol_util'].apply(lambda x: x.strip('%'))
LendingClub['revol_util'] = LendingClub['revol_util'].astype('float64')

#Create the dependent variable column, bad_loans, where 1 is a Charged Off or Defaulted Loan, and
#1 is a good loan, being paid off
LendingClub['bad_loans'] = LendingClub.apply(lambda row: 1 if row['loan_status'] == 'Charged Off' or row['loan_status'] == 'Default' else 0, axis=1)

#Convert target to Category datatype
LendingClub['bad_loans'] = LendingClub['bad_loans'].astype('category')

##one-hot encoding
#In each of these encodings, we'll convert the one variable to n dummy variables
#where n represents the number of distinct values for the original categorical variable
#then, we will take n-1 of those variables on into the dataframe

#Term
dummy_term = pd.get_dummies(LendingClub['term'], prefix='term')
LendingClub = pd.merge(LendingClub, pd.DataFrame(dummy_term.ix[:,1:]), left_index=True, right_index=True)
LendingClub = LendingClub.drop(['term'], axis=1)

#Grade
dummy_grade = pd.get_dummies(LendingClub['grade'], prefix='grade')
LendingClub = pd.merge(LendingClub, pd.DataFrame(dummy_grade.ix[:,1:]), left_index=True, right_index=True)
LendingClub = LendingClub.drop(['grade'], axis=1)

#Home Ownership
dummy_home = pd.get_dummies(LendingClub['home_ownership'], prefix='home_ownership')
LendingClub = pd.merge(LendingClub, pd.DataFrame(dummy_home.ix[:,1:]), left_index=True, right_index=True)
LendingClub = LendingClub.drop(['home_ownership'], axis=1)

#Purpose
dummy_purpose = pd.get_dummies(LendingClub['purpose'], prefix='purpose')
LendingClub = pd.merge(LendingClub, pd.DataFrame(dummy_purpose.ix[:,1:]), left_index=True, right_index=True)
LendingClub = LendingClub.drop(['purpose'], axis=1)

#Sub grade
dummy_subgrade = pd.get_dummies(LendingClub['sub_grade'], prefix='sub_grade')
LendingClub = pd.merge(LendingClub, pd.DataFrame(dummy_subgrade.ix[:,1:]), left_index=True, right_index=True)
LendingClub = LendingClub.drop(['sub_grade'], axis=1)

##Select a subset of features based on business input
selected_features = LendingClub[['loan_amnt','int_rate', 'annual_inc', 'dti', 'open_acc', 'revol_bal'
                                 , 'revol_util', 'delinq_2yrs', 'inq_last_6mths'
                                 ,'open_acc', 'pub_rec',
                                 'acc_open_past_24mths', 'tot_hi_cred_lim', 'term_ 60 months', 'grade_B', 'grade_C' ,
                                 'grade_D', 'grade_E', 'grade_F', 'grade_G', 'home_ownership_MORTGAGE',
                                 'home_ownership_NONE', 'home_ownership_OTHER', 'home_ownership_OWN', 
                                 'home_ownership_RENT', 'bad_loans']]

#fill nas with 0 for numeric features
selected_features = selected_features.fillna(0)

#Output the cleaned dataset and dataset with subset of features for future reference
LendingClub.to_csv("LoanStats_all_cleaned.csv",index=False)
selected_features.to_csv("LoanStats_keycols_cleaned.csv",index=False)

