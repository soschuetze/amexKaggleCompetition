setwd("/Users/sinclaireschuetze/Desktop/QAI Competition Project")
library("arrow")

train_data <- read_parquet("train_data.parquet", col_select = NULL, as_data_frame = TRUE)
train_labels <- read.csv("train_labels.csv",header=TRUE)

set.seed(1)
ind.train  = sample(1:nrow(train_data),10000,replace=FALSE) 
train.sample = train_data[ind.train,]

train.sample <- train.sample[ , -which(names(train.sample) %in% c("D_42", "D_49", "D_50", "D_53", "D_56", "S_9", "B_17", "D_66", "D_73", "D_76", "R_9", "D_82", "D_87", "B_29", "D_88", "D_105", "D_106", "R_26", "D_108", "D_110", "D_111", "B_39", "B_42", "D_132", "D_134", "D_135", "D_136", "D_137", "D_138", "D_142"))]
dim(train.sample)

train.sample <- train.sample[, -c(162)]

#noCategorical <- train.sample[,-which(names(train.sample) %in% c("D_64", "S_2", "B_30", "B_31", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"))]
#Characters D_63, D_64
categoricalList <- c("B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_68")
categorical <- train.sample[,which(names(train.sample) %in% c("B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_68"))]
train.sample <- train.sample[,-which(names(train.sample) %in% c("B_14","D_104","D_77", "D_139", "S_24","B_23","B_1","D_141","D_75","D_119","B_37","R_5","D_58","B_13","D_55","B_2"))]


calc_mode <- function(x){
  
  # List the distinct / unique values
  distinct_values <- unique(x)
  
  # Count the occurrence of each distinct value
  distinct_tabulate <- tabulate(match(x, distinct_values))
  
  # Return the value with the highest occurrence
  distinct_values[which.max(distinct_tabulate)]
}

train.sample <- as.data.frame(train.sample)

for(i in 1:ncol(train.sample)){
  name_of_column = colnames(train.sample)[i]
  if(any(colnames(categorical)==name_of_column)){
    next
  } else{
    train.sample[,i][is.na(train.sample[,i])] <- median(train.sample[,i], na.rm = T)
  }
}

for(i in 1:ncol(train.sample)){
  train.sample[,i][is.na(train.sample[,i])] <- calc_mode(train.sample[,i])
}

library(Hmisc)
head(train.sample)
train.sample <- train.sample[,-c(1,2)]
######################## 
#Partitioning Data
########################
set.seed(1)
ind.train  = sample(1:10000,5000,replace=FALSE) 
train.set = train.sample[ind.train,]
test.set = train.sample[-ind.train,]

######################## 
#Tree Methods
########################
library("tree")
#fitting classification tree
fit.tree <- tree(as.factor(target) ~ ., data = train.set)
set.seed(1)
result <- cv.tree(fit.tree, FUN = prune.tree, K=10)
plot(result)
new.tree <- prune.tree(fit.tree, best = 6)
plot(new.tree)
text(new.tree, pretty = 0)

Yhat.tree <- predict(new.tree, type = "class", newdata = test.set[,-145])
tablea <- table(Yhat.tree, test.set$target) 
tablea

totalCorrect = 0
for(i in 1:10){
  totalCorrect <- totalCorrect + tablea[i,i]
  i = i+1
}
totalCorrect
totalValues <- 5000 #known from partitioning
misclassRate <- (totalValues-totalCorrect)/totalValues
misclassRate
1-misclassRate

treeVariables <- train.sample[,which(names(train.sample) %in% c("P_2","B_9","B_3"))]
library(corrplot)
corMatrix <- cor(treeVariables)
corrplot(corMatrix, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)


#Random Forest
library(randomForest)
library(caret)
set.seed(1)
rf <- randomForest(as.factor(target)~., data=train.set, proximity=TRUE)
rf_pred <- predict(rf, type = "class", newdata = test.set[,-145])
CM <- table(rf_pred, train.set$target)
CM
accuracy = (sum(diag(CM)))/sum(CM)
accuracy

#Bagging
# Helper packages
library(ipred)
library(rpart)
library(MASS)
library(TH.data)

bag <- bagging(as.factor(target) ~ ., data = train.set, coob=TRUE)
bag_pred <- predict(bag, type = "class", newdata = test.set[,-145])
CMbag <- table(bag_pred, train.set$target)
CMbag
accuracyBag = (sum(diag(CMbag)))/sum(CMbag)
accuracyBag

######################## 
#SVM
########################
train.set$target = factor(train.set$target, levels = c(0, 1))
library(caTools)
library(e1071)

classifier = svm(formula = target ~ .,
                 data = train.set,
                 type = 'C-classification',
                 kernel = 'linear', probability = TRUE)

y_pred = predict(classifier, newdata = test.set[,-143], probability = TRUE)
head(attr(y_pred, "probabilities"))

cm = table(test.set[, 143], y_pred)

accuracySVM = (sum(diag(cm)))/sum(cm)
accuracySVM

######################## 
#Testing
########################
test_data <- read.csv("test_data.csv",header=TRUE)


######################## 
#VIF - does not need to be run again
########################

library("car")
lm <- glm(target~., data = noCategorical)
vif(lm, threshold = 10)

vif1DF <- noCategorical[,-which(names(noCategorical) %in% c("B_14"))]
lmVIF1 <- glm(target~., data = vif1DF)
vif(lmVIF1, threshold = 10)

vif2DF <- vif1DF[,-which(names(vif1DF) %in% c("D_104"))]
lmVIF2 <- glm(target~., data = vif2DF)
vif(lmVIF2, threshold = 10)

vif3DF <- vif2DF[,-which(names(vif2DF) %in% c("D_77"))]
lmVIF3 <- glm(target~., data = vif3DF)
vif(lmVIF3, threshold = 10)

vif4DF <- vif3DF[,-which(names(vif3DF) %in% c("D_139"))]
lmVIF4 <- glm(target~., data = vif4DF)
vif(lmVIF4, threshold = 10)

vif5DF <- vif4DF[,-which(names(vif4DF) %in% c("S_24"))]
lmVIF5 <- glm(target~., data = vif5DF)
vif(lmVIF5, threshold = 10)

vif6DF <- vif5DF[,-which(names(vif5DF) %in% c("B_23"))]
lmVIF6 <- glm(target~., data = vif6DF)
vif(lmVIF6, threshold = 10)

vif7DF <- vif6DF[,-which(names(vif6DF) %in% c("B_1"))]
lmVIF7 <- glm(target~., data = vif7DF)
vif(lmVIF7, threshold = 10)

vif8DF <- vif7DF[,-which(names(vif7DF) %in% c("D_141"))]
lmVIF8 <- glm(target~., data = vif8DF)
vif(lmVIF8, threshold = 10)

vif9DF <- vif8DF[,-which(names(vif8DF) %in% c("D_75"))]
lmVIF9 <- glm(target~., data = vif9DF)
vif(lmVIF9, threshold = 10)

vif10DF <- vif9DF[,-which(names(vif9DF) %in% c("D_119"))]
lmVIF10 <- glm(target~., data = vif10DF)
vif(lmVIF10, threshold = 10)

vif11DF <- vif10DF[,-which(names(vif10DF) %in% c("B_37"))]
lmVIF11 <- glm(target~., data = vif11DF)
vif(lmVIF11, threshold = 10)

vif12DF <- vif11DF[,-which(names(vif11DF) %in% c("R_5"))]
lmVIF12 <- glm(target~., data = vif12DF)
vif(lmVIF12, threshold = 10)

vif13DF <- vif12DF[,-which(names(vif12DF) %in% c("D_58"))]
lmVIF13 <- glm(target~., data = vif13DF)
vif(lmVIF13, threshold = 10)

vif14DF <- vif13DF[,-which(names(vif13DF) %in% c("B_13"))]
lmVIF14 <- glm(target~., data = vif14DF)
vif(lmVIF14, threshold = 10)

vif15DF <- vif14DF[,-which(names(vif14DF) %in% c("D_55"))]
lmVIF15 <- glm(target~., data = vif15DF)
vif(lmVIF15, threshold = 10)

vif16DF <- vif15DF[,-which(names(vif15DF) %in% c("B_2"))]
lmVIF16 <- glm(target~., data = vif16DF)
vif(lmVIF16, threshold = 10)
###############################


