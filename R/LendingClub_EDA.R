# Lending Club Analysis Exploratory Data Analysis

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

### Descriptive statistics

summary(loans_inactive$loan_amnt)

summary(loans_inactive$purpose)

# Univariate Plots - barplots & histograms

qplot(loans$loan_amnt, xlab = "Loan Amount ($)",ylab = "Number of Loans", main="Distribution of Loan Amounts")
qplot(loans_inactive$grade,xlab="Lending Club Loan Grade", ylab = "Number of Loans", main="Number of Loans by Grade")
qplot(loans_inactive$status,xlab="Status of Loan", ylab = "Number of Loans", main="Number of Loans by Final Status")
qplot(loans_inactive$int_rate, xlab = "Interest Rate (%)",ylab = "Number of Loans", main="Distribution of Interest Rates")

qplot(loans_inactive$dti, xlab = "Debt to Income Ratio",ylab = "Number of Loans", main="Distribution of DTI", bins = 70)
qplot(loans_inactive$payment_inc_ratio, xlab = "Payment to Income Ratio",ylab = "Number of Loans", main="Distribution of Payment to Income")
qplot(loans_inactive$home_ownership, xlab = "Home Ownership Catergory",ylab = "Number of Loans", main="Distribution of Howeownership")

# Kernel Density Plot
d <- density(loans_inactive$int_rate) # returns the density data for a variable
plot(d) # plots the results

# Univarate plots - box and whisker plots (ability to see distribution of the continuous variables/outliers)

bwplot(~loan_amnt, data = loans_inactive, main = "Box Plots of Loan Amounts", xlab = "Loan Amount ($)")
bwplot(~int_rate, data = loans_inactive, main = "Box Plots of Interest Rates", xlab = "Interest Rates (%)")
bwplot(~dti, data = loans_inactive, main = "Box Plots of DTI", xlab = "DTI")
bwplot(~payment_inc_ratio, data = loans_inactive, main = "Box Plots of Payment to Inc Ratio", xlab = "Payment to Inc Ratio")
bwplot(~annual_inc, data = loans_inactive, main = "Box Plots of Annual Income", xlab = "Annual Income ($)")

# multivariate visualizations

bwplot(term~int_rate, data = loans, main = "Loan Interest Rates by Term", xlab = "Interest Rate (%)")
bwplot(term~annual_inc, data = loans, main = "Borrower Income by Term", xlab = "Annual Income ($)")

bwplot(purpose~int_rate, data = loans, main = "Loan Interest Rates by Loan Purpose", xlab = "Interest Rate (%)")
bwplot(purpose~loan_amnt, data = loans, main = "Borrower Income by Loan Purpose", xlab = "Annual Income ($)")

bwplot(bad_loans~int_rate, data = loans, main = "Loan Interest Rates by Loan Status", xlab = "Interest Rate (%)")
bwplot(bad_loans~loan_amnt, data = loans, main = "Borrower Income by Loan Status", xlab = "Annual Income ($)")

# What does the bottom line look like, where are we trying to affect change? $ spend on Recovering bad loans 

# what does the cost of bad loans look like: in loan value and recovery cost?
lbls_recoveries <- sprintf("Recoveries - $%s", prettyNum(sum(loans_bad$recoveries),big.mark=","))

lbls_loanAmount = sprintf("Bad Loans - $%s", prettyNum(sum(loans_bad$loan_amnt),big.mark=","))

ds_loanAmount <- data.frame(labels=lbls_loanAmount,values=c(319288500))

ds_recoveries <- data.frame(labels=lbls_recoveries,values=c(12919336))

pie_loanAmount <- ggplot(ds_loanAmount, aes(x=factor(1),fill = labels)) +
  geom_bar(width = 1)
p1 <- pie_loanAmount  + coord_polar(theta = "y")+ labs(title="Total Loan Amount for Bad Loans", x=" ",y=" ") + theme(axis.text=element_blank(),axis.ticks=element_blank(),legend.title=element_blank())

pie_recoveries <- ggplot(ds_recoveries, aes(x=factor(1),fill = labels)) +
  geom_bar(width = 1)
p2 <- pie_recoveries + coord_polar(theta = "y") + labs(title="Total Recovery Fees for Bad Loans", x=" ",y=" ") + theme(axis.text=element_blank(),axis.ticks=element_blank(),legend.title=element_blank())

multiplot(p1,p2,cols=1)

# Proportion of Loans that are in default across different categories
par(mfrow=c(1,1)) 

loans2 <- loans_inactive
loans2$bad_loans <- as.factor(loans2$bad_loans)

# by grade
ggplot(data=loans2, aes(x=loans2$grade,fill=loans2$bad_loans)) + geom_bar(position="fill") + labs(title="Proportion of Loans in Default by Grade",x="Loan Grade (given by The Lending Club)",y="Proportion of Loans in Default") + scale_fill_manual(name="Loans in Default?",breaks=c(1,0),labels=c("Yes","No"),values=c("#00C094","#F8766D"))

# by home ownership status
ggplot(data=loans2, aes(x=loans2$home_ownership,fill=loans2$bad_loans)) + geom_bar(position="fill") + labs(title="Proportion of Loans in Default by Home Ownership Status",x="Home Ownership Status",y="Proportion of Loans in Default") + scale_fill_manual(name="Loans in Default?",breaks=c(1,0),labels=c("Yes","No"),values=c("#00C094","#F8766D"))

# by loan purpose
ggplot(data=loans2, aes(x=loans2$purpose,fill=loans2$bad_loans)) + geom_bar(position="fill") + labs(title="Proportion of Loans in Default by Purpose",x="Loan Grade (given by The Lending Club",y="Proportion of Loans in Default") +scale_fill_manual(name="Loans in Default?",breaks=c(1,0),labels=c("Yes","No"),values=c("#00C094","#F8766D"))+scale_x_discrete(labels=(abbreviate=c("credit_card"="crdt_crd","debt_consolidation"="debt_cnsld","home_improvement"="hm_imprv","major_purchase"="mjr_prch","small_business"="sm. biz","wedding"="wdng")))

# table view of the default % by purpose
t <- table(loans_inactive$purpose,loans_inactive$bad_loans)
d <- data.frame(round(100*(t[,2]/t[,1]),2))
d <- cbind(rownames(d),d)
names(d) <- c("Purpose","Percent of Bad Loans")
rownames(d) <- NULL
d

#Principal Components Analysis

# create factors out of loanee characteristics
loans_inactive$addr_state <- as.factor(loans_inactive$addr_state)
loans_inactive$home_ownership <- as.factor(loans_inactive$home_ownership)
loans_inactive$is_inc_v <- as.factor(loans_inactive$is_inc_v)
loans_inactive$emp_length_num <- as.factor(loans_inactive$emp_length_num)
loans_inactive$short_emp <- as.factor(loans_inactive$short_emp)

# PCA of variables in dataset (non-predictive, i.e. to see movements between variables)

loans.pca <- princomp(na.omit(loans_inactive[,c("loan_amnt", "installment", "int_rate", 
                                                "annual_inc", "payment_inc_ratio", "dti")]), cor = T)
summary(loans.pca)

screeplot(loans.pca, main = "Variance for PC of Metrics")

barplot(loans.pca$loadings[,1])
barplot(loans.pca$loadings[,2])

biplot(loans.pca)

# biplot is hard to read because of skew from annual inc

# remove outliers

loanbox <- boxplot(loans_inactive[, "annual_inc"])
in_loans <- loans_inactive[(loans_inactive[, "annual_inc"] < loanbox$stats[5] & loans_inactive[, "annual_inc"] > loanbox$stats[1]),]

# redo PCA
in.loans.pca <- princomp(na.omit(in_loans[,c("loan_amnt", "installment", "int_rate", 
                                             "annual_inc", "payment_inc_ratio", "dti")]), cor = T)
summary(in.loans.pca)

screeplot(in.loans.pca, main = "Variance for PC of Metrics")

barplot(in.loans.pca$loadings[,1])
barplot(in.loans.pca$loadings[,2])

biplot(loans.pca)

# PCA of predictors

reg.loans.pca <- princomp(na.omit(loans_inactive[,c("loan_amnt", "int_rate", "annual_inc", "payment_inc_ratio", "dti")]), cor = T )
summary(reg.loans.pca)

# Biplot of PCA vectors
biplot(reg.loans.pca)

# plot of variance of PCA analysis
screeplot(reg.loans.pca, main = "Variance for PC of Metrics")

barplot(reg.loans.pca$loadings[,1])
barplot(reg.loans.pca$loadings[,2])
