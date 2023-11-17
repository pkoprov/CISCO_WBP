library(tree)
library(caret)




df = read.csv("UR_merged.csv")


labels = unique(df[1])
n = 22

target = labels[[1]][4]

target_df = df[which(df[1]==target),]

set.seed(123)
ind = sample(1:nrow(target_df), n)
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



## Raw data

# Decision Tree
library(rpart)

start = proc.time()
tree.mod = rpart(asset~.,data = train_data, method = "class", minsplit=5, xval=10)
print(proc.time()-start)
plot(tree.mod, uniform = T)
text(tree.mod)


pred = predict(tree.mod, test_data, type = "class")

confusionMatrix(table(pred, test_data[,1]), mode = "everything")

# Random forest
library(randomForest)

start = proc.time()
bag.mod = randomForest(as.factor(asset)~.,data = train_data, importance=T, ntree = 150)
print(proc.time()-start)
bag.mod
plot(bag.mod)
varImpPlot(bag.mod)
pred = predict(bag.mod, test_data)

confusionMatrix(table(pred, test_data[,1]), mode = "everything")


## LASSO
library(glmnet)

x = model.matrix(asset~.,train_data)[,-1]
y = train_data[,1]
grid=10^seq(-10,0,length=30)

start = proc.time()
lasso.mod = cv.glmnet(x,y, alpha = 1, family = "binomial", type.measure = "class", lambda = grid,nfolds = nrow (x),grouped = FALSE)
print(proc.time()-start)
plot(lasso.mod)
lasso.mod

bestlam.r=lasso.mod$lambda.min
test_x = model.matrix(asset~.,test_data)[,-1]
test_y = test_data[,1]
pred = predict(lasso.mod, s=bestlam.r, newx = test_x)
CM = confusion.glmnet(lasso.mod, newx = test_x, newy = test_y)
print(CM)
precision = CM[1]/(CM[1]+CM[3])
recall = CM[1]/(CM[1]+CM[2])
f1 = 2*precision*recall/(precision+recall)
print(f1)

# SVM

library(e1071)
set.seed(123)
start = proc.time()
svm.mod = svm(as.factor(asset) ~ ., data = train_data, type = 'nu-classification',
              kernel = 'linear')
print(proc.time()-start)
pred = predict(svm.mod, test_data)
confusionMatrix(table(pred, test_data[,1]), mode = "everything")


### anomaly detection


## OCSVM

library(kernlab)

par(mfrow=c(3,5))
set.seed(123)

for (i in 1:length(labels[[1]])){
  target = labels[[1]][i]
  
  target_df = df[which(df[1]==target),]
  
  ind = sample(1:nrow(target_df), n)
  train_data = target_df[ind,]
  test_data = rbind(target_df[-ind,], df[which(df[1]!=target),])

  start = proc.time()
  model_svm = ksvm(asset~., train_data[1:n,], type="one-svc")
  print(proc.time()-start)
  
  train_scores = predict(model_svm, train_data[1:n,], type = "decision")
  test_scores = predict(model_svm, test_data, type = "decision")
  
  plot(test_scores, main = target, cex = 2, ylim = c(-1.75, 0.1))
  target_ind = which(test_data[,1] == target)
  points(target_ind,test_scores[target_ind], col="red", pch = 20, cex = 2)
  
  points(seq(1:n),train_scores[1:n], col="blue", pch = 20, cex = 2)
  cutoff = sort(test_scores[target_ind],decreasing = TRUE)[nrow(target_df)-n-1]
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
  print(confusionMatrix(table(pred, true), mode = "everything"))
}



############ Isolated forest ############ 

library(isotree)
par(mfrow=c(3,5))
set.seed(123)

for (i in 1:length(labels[[1]])){
  target = labels[[1]][i]
  
  target_df = df[which(df[1]==target),]
  
  ind = sample(1:nrow(target_df), n)
  train_data = target_df[ind,]
  test_data = rbind(target_df[-ind,], df[which(df[1]!=target),])
  
  start = proc.time()
  model_isoforest = isolation.forest(train_data[1:n,], ntrees =10000)
  print(proc.time()-start)
  train_scores = predict(model_isoforest, train_data[1:n,], type = "score")
  test_scores = predict(model_isoforest, test_data, type = "score")
  
  plot(test_scores, main = target, cex = 2, ylim =c(0.46, 0.64))
  target_ind = which(test_data[,1] == target)
  points(target_ind,test_scores[target_ind], col="red", pch = 20, cex = 2)
  points(seq(1:n),train_scores[1:n], col="blue", pch = 20, cex = 2)
  
  test_cutoff = sort(test_scores[target_ind])[nrow(target_df)-n-1]
  train_cutoff = sort(train_scores)[n-1]
  cutoff = ifelse(test_cutoff > train_cutoff, test_cutoff, train_cutoff)
  
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
  print(confusionMatrix(table(pred, true), mode = "everything"))
}

##################################################################################

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


