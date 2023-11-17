ur_df = read.csv("./5_7_2022/df_UR5_all.csv")

# ur_labels = unique(ur_df[1])[[1]]

# df is used in all functions
df = ur_df


train_test_split = function(df, label){
  ur_labels = unique(df[1])[[1]]
  # define df with only target class data
  target =  ur_labels[label]
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


library(glmnet)

for (i in 1:4){
  
  # split data
  
  result = train_test_split(df,i)
  train_data = result[[1]]
  test_data = result[[2]]
  len = result[[3]]
  target = train_data[1,1]
  
  # Calculate PCS
  v = svd(train_data[1:len,-1])$v
  PCS = train_data[,-1]*v
  
  x = data.matrix(PCS)
  y = c(rep(1,len), rep(0,len))
  grid=10^seq(-10,0,length=30)
  
  # Apply LASSO with LOOCV
  cv.fit = cv.glmnet(x,y, alpha = 0.5, family = "binomial", type.measure = "class", lambda = grid, nfolds = nrow (x),grouped = FALSE)
  
  par(mfrow=c(2,1))
  plot(cv.fit)
  
  # reduce number of variables to those that are not 0
  lasso_coef = coef(cv.fit,s = cv.fit$lambda.min)
  betas = which(lasso_coef != 0)
  
  
  train_PCS = PCS[,betas]
  test_PCS = (test_data[,-1]*v)[,betas]
  
  # train logit regression
  log.mod = glm(y ~ ., data = train_PCS, family = "binomial")
  
  train_pred = predict(log.mod, train_PCS, type = "response")
  
  plot(train_pred, main = target, ylim = c(0,1))
  points(train_pred[train_data[,1]!= "false"],col="red", pch = 20)
  
  test_pred = predict(log.mod, test_PCS, type = "response")
  
  points(test_pred[test_data[,1]!= "false"],col="red", pch = 0, cex=1.5)
  points(test_pred[test_data[,1]== "false"], pch = 2, cex = 1.5)

  legend(42, 1.1, legend=c("Target train", "False train", "Target test", "False test"),
         col=c("red", "black","red", "black"), pch = c(20,1,0,2), cex=c(1,1,1,1))
  
  
  pred = c()
  for (j in test_pred){
    if (j > 0.9)
    {pred = append(pred,target)}
    else {pred = append(pred, "false")}
  }
  
  library(caret)
  confusionMatrix(table(pred, test_data[,1]))
  
}