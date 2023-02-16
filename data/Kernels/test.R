library(glmnet)
library(tree)
library(dplyr)
library(caret)
library(rpart)
library(randomForest)


ur_df = read.csv("2023_02_07/merged.csv")
ur_labels = unique(ur_df[1])

df = ur_df

target = ur_labels[[1]][1]

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



## PCA

v = svd(train_data[1:22,-1])$v
PCS = train_data[,-1]

x = data.matrix(PCS)
y = c(rep(1,22), rep(0,22))
grid=10^seq(-10,0,length=30)

library(glmnet)
cv.fit = cv.glmnet(x,y, alpha = 1, family = "binomial", type.measure = "class", lambda = grid, nfolds = nrow (x),grouped = FALSE)

# plot(cv.fit)

lasso_coef = coef(cv.fit,s = cv.fit$lambda.min)
betas = which(lasso_coef != 0)


train_PCS = PCS[,betas]
test_PCS = (test_data[,-1]*t(v))[,betas]

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

tree.mod = rpart(X~.,data = train_data, method = "class", minsplit=5, xval=10)
plot(tree.mod, uniform = T)
text(tree.mod)

pred = predict(tree.mod, test_data, type = "class")

confusionMatrix(table(pred, test_data[,1]))

# Random forest

bag.mod = randomForest(as.factor(X)~.,data = train_data, importance=T)
bag.mod
plot(bag.mod)
varImpPlot(bag.mod)
pred = predict(bag.mod, test_data)

confusionMatrix(table(pred, test_data[,1]))


## LASSO
x = model.matrix(X~.,train_data)[,-1]
y = train_data[,1]
grid=10^seq(-10,0,length=30)


lasso.mod = cv.glmnet(x,y, alpha = 1, family = "binomial", type.measure = "class", lambda = grid,nfolds = 10)
plot(lasso.mod)

bestlam.r=lasso.mod$lambda.min
test_x = model.matrix(X~.,test_data)[,-1]
test_y = test_data[,1]
pred = predict(lasso.mod, s=bestlam.r, newx = test_x)
confusion.glmnet(lasso.mod, newx = test_x, newy = test_y)

# SVM

library(e1071)

svm.mod = svm(as.factor(X) ~ ., data = train_data, type = 'nu-classification',
              kernel = 'linear')

pred = predict(svm.mod, test_data)
confusionMatrix(table(pred, test_data[,1]))

### anomaly detection


## OCSVM

library(kernlab)

par(mfrow=c(2,2))

for (i in 1:length(ur_labels[[1]])){
  target = ur_labels[[1]][i]
  
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
  
  
  model_svm = ksvm(X~., train_data[1:22,], type="one-svc")
  train_scores = predict(model_svm, train_data, type = "decision")
  test_scores = predict(model_svm, test_data, type = "decision")
  
  plot(test_scores, main = target, cex = 2, ylim=c(-1.7,0.1))
  target_ind = which(test_data[,1] == target)
  points(target_ind,test_scores[target_ind], col="red", pch = 20, cex = 2)
  
  points(seq(1:22), train_scores[1:22], col="blue", pch = 20, cex = 2)
  cutoff = -0.5
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

for (i in 1:length(ur_labels[[1]])){
  target = ur_labels[[1]][i]
  
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
  cutoff = 0.5
  abline(cutoff,0, col="red")
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

