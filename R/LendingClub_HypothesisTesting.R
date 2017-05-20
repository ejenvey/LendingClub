# Lending Club Analysis Hypothesis Testing

# Eric Jenvey

# Use LendingClub_DataPreprocessing.R to read and clean the data

source("multiplot.R")

library(dplyr)
library(ggplot2)
library(SDMTools)
library(ROCR)
library(lattice)
library(plyr)
library(reshape2)

# ANOVA testing... is there statistical basis to say that all bad_loan rates are not the same across a category

aov <- aov(bad_loans~purpose, data = loans_inactive)
summary(aov)

aov <- aov(bad_loans~home_ownership, data = loans_inactive)
summary(aov)

aov <- aov(bad_loans~sub_grade, data = loans_inactive)
summary(aov)

# t-testing, is there a stat basis to say that the default rates for two groups are different from each other?
loans_bad <- loans_inactive[loans_inactive$status=="Charged Off"|loans_inactive$status=="Default",]
loans_good <- loans_inactive[loans_inactive$status!="Charged Off"&loans_inactive$status!="Default",]

t.test(loans_good$loan_amnt, loans_bad$loan_amnt, alternative = "two.sided", var.equal = TRUE)
t.test(loans_good$annual_inc, loans_bad$annual_inc, alternative = "two.sided", var.equal = TRUE)
t.test(loans_good$dti, loans_bad$dti, alternative = "two.sided", var.equal = TRUE)
t.test(loans_good$payment_inc_ratio, loans_bad$payment_inc_ratio, alternative = "two.sided", var.equal = TRUE)
