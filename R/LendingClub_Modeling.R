# Lending Club Analysis Model Building and Validation

# Eric Jenvey

# Use LendingClub_DataPreprocessing.R to read and clean the data

# LendingClub_EDA and LendingClub_HypothesisTesting store the code to generate
# visualizations, other EDA, and univariate/multivariate statistical tests on the data

set.seed(123)

# split loans into training and test sets for model, subset the test set loans by bad loans

sample <- sample.int(122607,size=98085)
trainSet <- loans_inactive[sample,]
testSet <- loans_inactive[-sample,]
loans_bad_test <- testSet[testSet$status=="Charged Off"|testSet$status=="Default",]

s <- as.matrix(c(nrow(trainSet[trainSet$bad_loans==0,]),nrow(testSet[testSet$bad_loans==0,])))
s <- rbind(s,nrow(trainSet[trainSet$bad_loans==1,]),nrow(testSet[testSet$bad_loans==1,]))
s <- cbind(s,c("TrainingSet","TestingSet","TrainingSet","TestingSet"),c("No","No","Yes","Yes"))
s <- data.frame(s)
names(s) <- c("NumRows","Set","Loan_Default_Indicator")
s$NumRows <- as.numeric(as.character(s$NumRows))
g <- ggplot(s, aes(x=Set,y=NumRows,fill=Loan_Default_Indicator))
g + geom_bar(stat="identity") + labs(x="Data Split", y="Number of Loans",title="Number of Loans by Data Split") + scale_fill_manual(name="Loan in Default?",breaks=c("Yes","No"),labels=c("Yes","No"),values=c("#00C094","#F8766D"))

# logit model

loansModel <- glm(bad_loans ~ grade + sub_grade_num + emp_length_num + home_ownership + dti + purpose 
                  + payment_inc_ratio + delinq_2yrs + inq_last_6mths + open_acc 
                  + pub_rec + revol_util + total_rec_late_fee, family = binomial, data=trainSet,na.action = na.omit) 

par(mfrow = c(2,2))
plot(loansModel,which =1)
plot(loansModel,which =2)
plot(loansModel,which =3)
plot(loansModel,which =5)

par(mfrow = c(1,1))

summary(loansModel)

# log transformed data model

logLoans <- loans_inactive

logLoans[, c("loan_amnt", "annual_inc", "dti", "payment_inc_ratio", "revol_util", "total_rec_late_fee")] <- log(loans_inactive[, c("loan_amnt", "annual_inc", "dti", "payment_inc_ratio", "revol_util", "total_rec_late_fee")] + .0001)
logLoans$bad_loans <- loans_inactive$bad_loans

log.trainSet <- logLoans[sample,]
log.testSet <- logLoans[-sample,]
log_bad_test <- log.testSet[log.testSet$status=="Charged Off"|log.testSet$status=="Default",]


loansModel2 <- glm(bad_loans ~ grade + sub_grade_num + emp_length_num + home_ownership + dti + purpose 
                   + payment_inc_ratio + open_acc + loan_amnt + annual_inc + revol_util + total_rec_late_fee, family = binomial, data=log.trainSet,na.action = na.omit) 

par(mfrow = c(2,2))
plot(loansModel2,labels.id = NULL)
par(mfrow = c(1,1))

summary(loansModel2)

## Model Validation

### ANOVA test of models

anova(loansModel, loansModel2, test="Chisq")

### Confusion Matrix


fitted.results <- predict(loansModel, newdata=testSet, type="response")
fitted.results <- ifelse(fitted.results > 0.5, 1, 0)
misClasificError <- mean(fitted.results != testSet$bad_loans, na.rm=TRUE)

c1 <- confusion.matrix(testSet$bad_loans,fitted.results)
c1[1:2,1:2]

# 25%**
fitted.results <- predict(loansModel, newdata=testSet, type="response")
fitted.results <- ifelse(fitted.results > 0.25, 1, 0)
misClasificError <- mean(fitted.results != testSet$bad_loans, na.rm=TRUE)

c2 <- confusion.matrix(testSet$bad_loans,fitted.results)
c2[1:2,1:2]

# 75%**
fitted.results <- predict(loansModel, newdata=testSet, type="response")
fitted.results <- ifelse(fitted.results > 0.75, 1, 0)
misClasificError <- mean(fitted.results != testSet$bad_loans, na.rm=TRUE)

c3 <- confusion.matrix(testSet$bad_loans,fitted.results)
c3[1:2,1:2]

# log'd logit
log.results <- predict(loansModel2, newdata=log.testSet, type="response")
log.results <- ifelse(log.results > 0.5, 1, 0)
misClasificError <- mean(log.results != log.testSet$bad_loans, na.rm=TRUE)

c2 <- confusion.matrix(log.testSet$bad_loans,log.results)
c2[1:2,1:2]

# 25%**
log.results <- predict(loansModel2, newdata=log.testSet, type="response")
log.results <- ifelse(fitted.results > 0.25, 1, 0)
misClasificError <- mean(log.results != log.testSet$bad_loans, na.rm=TRUE)

c2 <- confusion.matrix(log.testSet$bad_loans,log.results)
c2[1:2,1:2]

# 75%**
log.results <- predict(loansModel2, newdata=log.testSet, type="response")
log.results <- ifelse(log.results > 0.75, 1, 0)
misClasificError <- mean(log.results != testSet$bad_loans, na.rm=TRUE)

c2 <- confusion.matrix(log.testSet$bad_loans,log.results)
c3[1:2,1:2]

### ROC Curves for both options

m <- matrix(data=c(0,0,1,1),nrow=2,ncol=2,byrow=TRUE)

plot(m,xlim=c(0,1),ylim=c(0,1), xlab="False positive rate",ylab="True positive rate",type="l")
points(.2,.2,pch=19)

fitted.results <- predict(loansModel, newdata=testSet, type="response")
pr <- prediction(fitted.results, testSet$bad_loans)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")

log.results <- predict(loansModel2, newdata=log.testSet, type="response")
lpr <- prediction(log.results, log.testSet$bad_loans)
lprf <- performance(lpr, measure = "tpr", x.measure = "fpr")

plot(prf)
lines(c(0,1),c(0,1))
plot(lprf, add = TRUE, colorize = TRUE)

points(.2,.2,pch=19,col="Forest Green")
points(.05,.2,pch=19,col="Red")

### Conclusion (using Eric's model)

lbls_recoveries <- c(sprintf("Saved Recoveries - $%s", prettyNum(sum(loans_bad$recoveries)*.2,big.mark=",")),sprintf("Remaining Recoveries - $%s", prettyNum(sum(loans_bad$recoveries)*.8,big.mark=",")))

lbls_loanAmount <- c(sprintf("Saved Loan Amounts - $%s", prettyNum(sum(loans_bad$loan_amnt)*.2,big.mark=",")),sprintf("Remaining Bad Loans - $%s", prettyNum(sum(loans_bad$loan_amnt)*.8,big.mark=",")))

ds_loanAmount <- data.frame(labels=lbls_loanAmount,values=c(319288500*.2,319288500*.8))

ds_recoveries <- data.frame(labels=lbls_recoveries,values=c(12919336*.2,12919336*.8))

pie_loanAmount <- ggplot(ds_loanAmount, aes(x=factor(1),y=values,fill = labels)) +
  geom_bar(stat="identity")
p1 <- pie_loanAmount  + coord_polar(theta = "y")+ labs(title="Total Loan Amount", x=" ",y=" ") + theme(axis.text=element_blank(),axis.ticks=element_blank(),legend.title=element_blank())

pie_recoveries <- ggplot(ds_recoveries, aes(x=factor(1),y=values,fill = labels)) +
  geom_bar(stat="identity")
p2 <- pie_recoveries + coord_polar(theta = "y") + labs(title="Total Recovery Fees", x=" ",y=" ") + theme(axis.text=element_blank(),axis.ticks=element_blank(),legend.title=element_blank())

multiplot(p1,p2,cols=1)
