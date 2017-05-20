# Lending Club Analysis Data Preprocessing

# Eric Jenvey

source("multiplot.R")

library(dplyr)
library(ggplot2)
library(SDMTools)
library(ROCR)
library(lattice)
library(plyr)
library(reshape2)


# create df, data types and examples of complete entries

# Reading from a locally created file, more pre-processing steps were necessary, but
# leaving those out of the script for now
loans <- read.csv("loanStats.csv")
loans_complete <- loans[complete.cases(loans),]

# recode non-numeric values as factors
loans$id <- as.factor(loans$id)
loans$member_id <- as.factor(loans$member_id)
loans$policy_code <- as.factor(loans$policy_code)
loans$not_compliant <- as.factor(loans$not_compliant)
loans$inactive_loans <- as.factor(loans$inactive_loans)
loans$short_emp <- as.factor(loans$short_emp)
loans$last_delinq_none <- as.factor(loans$last_delinq_none)
loans$last_record_none <- as.factor(loans$last_record_none)
loans$last_major_derog_none <- as.factor(loans$last_major_derog_none)
loans$inactive <- loans$status == "Fully Paid" | loans$status == "Charged Off" | loans$status=="Default"

# create a dataframe of the inactive loans from the complete loans
loans_inactive <- loans[loans$inactive=="TRUE",]
loans_inactive$bad_loans_label <- ifelse(loans_inactive$bad_loans == 1,"Yes","No")
loans_inactive$bad_loans_label <- as.factor(loans_inactive$bad_loans_label)

# create a dataframe of just the bad loans
loans_bad <- loans_inactive[loans_inactive$status=="Charged Off"|loans_inactive$status=="Default",]

# refactoring levels of variables to make them more readable for graphics and other
# modeling output

levels(loans_inactive$purpose) <- c("Car","Credit Card","Debt Cons.", "Home Imp.","House",
                                    "Maj. Purch.", "Medical", "Moving", "Other", "Small Bus.", 
                                    "Vacation", "Wedding")
levels(loans_inactive$emp_length_num) <- c("Under one year","1 year","2 years","3 years","4 years","5 years","6 years","7 years","8 years","9 years","10 years","Over 10 years")
