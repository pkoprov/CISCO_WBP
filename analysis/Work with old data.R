library(glmnet)
library(tree)
library(dplyr)
library(caret)
library(rpart)
library(randomForest)



ur_df = read.csv("./5_7_2022/df_UR5_all.csv")

# ur_labels = unique(ur_df[1])[[1]]

# df is used in all functions
df = ur_df


train_test_split = function(df){
  ur_labels = unique(df[1])[[1]]
  # define df with only target class data
  target =  ur_labels[1]
  target_df = df[which(df[1]==target),]
  
  
  ind = sample(1:nrow(target_df), 22) # indices for train/test split of target df
  train_target_df = target_df[ind,] 
  test_target_df = target_df[-ind,]
  
  # because there are more non-target data we need to resample it to match number
  # of samples of target df
  false_df = df[which(df[1]!=target),]
  ind = sample(1:nrow(false_df), nrow(train_target_df))
  train_false_df = false_df[ind,]
  
  # label non-target data as "false"
  train_false_df[,1] = "false"
  test_false_df = false_df[-ind,]
  test_false_df[,1] = "false"
  
  # unite target and non-target data in one training and one testing dataframes
  train_data = rbind(train_target_df,train_false_df )
  test_data = rbind(test_target_df,test_false_df )
  
  len = nrow(train_target_df)
  return(list(train_data, test_data, len))
}


result = train_test_split(df)
tr1 = result[1]
test1 = result[2]
len1 = result[3]

# define df with only target class data
target =  ur_labels[1]
target_df = df[which(df[1]==target),]


ind = sample(1:nrow(target_df), 22) # indices for train/test split of target df
train_target_df = target_df[ind,] 
test_target_df = target_df[-ind,]

# because there are more non-target data we need to resample it to match number
# of samples of target df
false_df = df[which(df[1]!=target),]
ind = sample(1:nrow(false_df), nrow(train_target_df))
train_false_df = false_df[ind,]

# label non-target data as "false"
train_false_df[,1] = "false"
test_false_df = false_df[-ind,]
test_false_df[,1] = "false"

# unite target and non-target data in one training and one testing dataframes
train_data = rbind(train_target_df,train_false_df )
test_data = rbind(test_target_df,test_false_df )

len = nrow(train_target_df)


## PCA

v = svd(train_data[1:len,-1])$v
PCS = train_data[,-1]*v

x = data.matrix(PCS)
y = c(rep(1,len), rep(0,len))
grid=10^seq(-10,0,length=30)

library(glmnet)
cv.fit = cv.glmnet(x,y, alpha = 1, family = "binomial", type.measure = "class", lambda = grid, nfolds = nrow (x),grouped = FALSE)

plot(cv.fit)

lasso_coef = coef(cv.fit,s = cv.fit$lambda.min)
betas = which(lasso_coef != 0)


train_PCS = PCS[,betas]
test_PCS = (test_data[,-1]*v)[,betas]

log.mod = glm(y ~ ., data = train_PCS, family = "binomial")

train_pred = predict(log.mod, train_PCS, type = "response")

plot(train_pred, main = target, ylim = c(0,1))
points(train_pred[train_data[,1]!= "false"],col="red", pch = 20)

test_pred = predict(log.mod, test_PCS, type = "response")

points(test_pred[test_data[,1]!= "false"],col="red", pch = 0, cex=1.5)
points(test_pred[test_data[,1]== "false"], pch = 2, cex = 1.5)
# abline(0,b=0, col='green')
legend(0, 0.2, legend=c("Target train", "False train", "Target test", "False test"),
       col=c("red", "black","red", "black"), pch = c(20,1,0,2), cex=c(1,1,1,1))


pred = c()
for (j in test_pred){
  if (j > 0.9)
  {pred = append(pred,target)}
  else {pred = append(pred, "false")}
}

library(caret)
confusionMatrix(table(pred, test_data[,1]))


## Raw data

# Decision Tree

tree.mod = rpart(X0~.,data = train_data, method = "class", minsplit=5, xval=10)
plot(tree.mod, uniform = T)
text(tree.mod)

pred = predict(tree.mod, test_data, type = "class")

confusionMatrix(table(pred, test_data[,1]))

# Random forest

bag.mod = randomForest(as.factor(X0)~.,data = train_data, importance=T)
bag.mod
plot(bag.mod)
varImpPlot(bag.mod)
pred = predict(bag.mod, test_data)

confusionMatrix(table(pred, test_data[,1]))


## LASSO
x = model.matrix(X0~.,train_data)[,-1]
y = train_data[,1]
grid=10^seq(-10,0,length=30)


lasso.mod = cv.glmnet(x,y, alpha = 1, family = "binomial", type.measure = "class", lambda = grid,nfolds = 10)
plot(lasso.mod)

bestlam.r=lasso.mod$lambda.min
test_x = model.matrix(X0~.,test_data)[,-1]
test_y = test_data[,1]
pred = predict(lasso.mod, s=bestlam.r, newx = test_x)
confusion.glmnet(lasso.mod, newx = test_x, newy = test_y)

# SVM

library(e1071)

svm.mod = svm(as.factor(X0) ~ ., data = train_data, type = 'nu-classification',
              kernel = 'linear')

pred = predict(svm.mod, test_data)
confusionMatrix(table(pred, test_data[,1]))

### anomaly detection


## OCSVM

library(kernlab)

par(mfrow=c(2,2))

for (i in 1:length(ur_labels)){
  target = ur_labels[i]
  
  target_df = df[which(df[1]==target),]
  
  ind = sample(1:nrow(target_df), 22)
  train_target_df = target_df[ind,]
  test_target_df = target_df[-ind,]
  
  false_df = df[which(df[1]!=target),]
  ind = sample(1:nrow(false_df), nrow(train_target_df))
  train_false_df = false_df[ind,]
  train_false_df[,1] = "false"
  test_false_df = false_df[-ind,]
  test_false_df[,1] = "false"
  
  train_data = rbind(train_target_df,train_false_df )
  test_data = rbind(test_target_df,test_false_df )
  
  
  model_svm = ksvm(X0~., train_data[1:22,], type="one-svc")
  train_scores = predict(model_svm, train_data[1:22,], type = "decision")
  test_scores = predict(model_svm, test_data, type = "decision")
  
  plot(test_scores, main = target, cex = 2)
  target_ind = which(test_data[,1] == target)
  points(target_ind,test_scores[target_ind], col="red", pch = 20, cex = 2)
  
  points(seq(1:22),train_scores[1:22], col="blue", pch = 20, cex = 2)
  cutoff = quantile(c(test_scores[target_ind],train_scores), 0.05)
  abline(cutoff,0, col="red")
  
  scores = predict(model_svm, df, type = "decision")
  pred = c()
  for (j in scores){
    if (j > cutoff)
    {pred = append(pred,target)}
    else {pred = append(pred, "false")}
  }
  true = rep("false", nrow(df))
  true[which(df[,1] == target)] = target
  print(confusionMatrix(table(pred, true)))
}



############ Isolated forest ############ 

library(isotree)
par(mfrow=c(2,2))
set.seed(123)

for (i in 1:length(ur_labels)){
  target = ur_labels[i]
  
  target_df = df[which(df[1]==target),]
  
  ind = sample(1:nrow(target_df), 22)
  train_target_df = target_df[ind,]
  test_target_df = target_df[-ind,]
  
  false_df = df[which(df[1]!=target),]
  ind = sample(1:nrow(false_df), nrow(train_target_df))
  train_false_df = false_df[ind,]
  train_false_df[,1] = "false"
  test_false_df = false_df[-ind,]
  test_false_df[,1] = "false"
  
  train_data = rbind(train_target_df,train_false_df )
  test_data = rbind(test_target_df,test_false_df )
  
  
  model_isoforest = isolation.forest(train_data[1:22,], ntrees =1000)
  train_scores = predict(model_isoforest, train_data[1:22,], type = "score")
  test_scores = predict(model_isoforest, test_data, type = "score")
  
  plot(test_scores, main = target, cex = 2)
  target_ind = which(test_data[,1] == target)
  points(target_ind,test_scores[target_ind], col="red", pch = 20, cex = 2)
  
  points(seq(1:22),train_scores[1:22], col="blue", pch = 20, cex = 2)
  cutoff = quantile(c(test_scores[target_ind],train_scores), 0.95)
  cutoff2 = quantile(c(test_scores[target_ind],train_scores), 0.75)
  # cutoff = mean(c(test_scores[target_ind],train_scores))+2*sd(c(test_scores[target_ind],train_scores))
  abline(cutoff,0, col="red")
  if (i %in% c(2)){abline(cutoff2,0, col="red", lty = 2)}
  
  scores = predict(model_isoforest, df, type = "score")
  pred = c()
  for (j in scores){
    if (j < cutoff)
    {pred = append(pred,target)}
    else {pred = append(pred, "false")}
  }
  true = rep("false", nrow(df))
  true[which(df[,1] == target)] = target
  print(confusionMatrix(table(pred, true)))
}


######################## LOF ########################


library(DescTools)

par(mfrow=c(2,2))
set.seed(123)

for (i in 1:length(ur_labels)){
  target = ur_labels[i]
  
  target_df = df[which(df[1]==target),]
  train_ind = row.names(target_df)[1:22]
  test_ind = row.names(target_df)[-train_ind]
  
  
  scores = c()
  for (l in 23:35){
    LOF_scores = LOF(df[c(train_ind[1:22], l),1], 20)
    scores = append(scores, LOF_scores[23])
  }
  plot(LOF_scores[1:22], main = target, cex = 2, ylim = c(min(LOF_scores, scores), max(LOF_scores, scores)))
  points(scores[1:8],col="red", pch = 20, cex = 2)
  
  points(scores[9:length(scores)],col="blue", pch = 20, cex = 2)
  
  
  
  plot(LOF_scores, main = target, cex = 2)
  target_ind = which(test_data[,1] == target)
  points(target_ind,test_scores[target_ind], col="red", pch = 20, cex = 2)
  
  points(seq(1:22),train_scores[1:22], col="blue", pch = 20, cex = 2)
  cutoff = quantile(c(test_scores[target_ind],train_scores[1:22]), 0.90)
  cutoff2 = quantile(c(test_scores[target_ind],train_scores[1:22]), 0.75)
  # cutoff = mean(c(test_scores[target_ind],train_scores))+2*sd(c(test_scores[target_ind],train_scores))
  abline(cutoff,0, col="red")
  if (i %in% c(2)){abline(cutoff2,0, col="red", lty = 2)}
  
  scores = LOF(df[,-1], 22)
  pred = c()
  for (j in scores){
    if (j < cutoff2)
    {pred = append(pred,target)}
    else {pred = append(pred, "false")}
  }
  true = rep("false", nrow(df))
  true[which(df[,1] == target)] = target
  print(confusionMatrix(table(pred, true)))
}
