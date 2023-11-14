# libraries needed
library(caret)
library(dplyr)
library(randomForest) # RF
library(e1071) # SVM
library(ROSE) # ROSE
library(DMwR) # SMOTE
library(pROC) # ROC 


# features selected >= 5000 times
features.var <- unlist(selected_500_features$features.name)
#features.var <- unlist(feature$feature.name)
# set permutation number
permnum <- 100
svmlsmoteroc.auc <-matrix(0,nrow=permnum,ncol=3)
svmlroseroc.auc <-matrix(0,nrow=permnum,ncol=3)
svmpsmoteroc.auc <-matrix(0,nrow=permnum,ncol=3)
svmproseroc.auc <-matrix(0,nrow=permnum,ncol=3)
svmrsmoteroc.auc <-matrix(0,nrow=permnum,ncol=3)
svmrroseroc.auc <-matrix(0,nrow=permnum,ncol=3)
svmssmoteroc.auc <-matrix(0,nrow=permnum,ncol=3)
svmsroseroc.auc <-matrix(0,nrow=permnum,ncol=3)
et_svmlsmoteroc.auc <-matrix(0,nrow=permnum,ncol=3)
et_svmlroseroc.auc <-matrix(0,nrow=permnum,ncol=3)
et_svmpsmoteroc.auc <-matrix(0,nrow=permnum,ncol=3)
et_svmproseroc.auc <-matrix(0,nrow=permnum,ncol=3)
et_svmrsmoteroc.auc <-matrix(0,nrow=permnum,ncol=3)
et_svmrroseroc.auc <-matrix(0,nrow=permnum,ncol=3)
et_svmssmoteroc.auc <-matrix(0,nrow=permnum,ncol=3)
et_svmsroseroc.auc <-matrix(0,nrow=permnum,ncol=3)

# discovery cohort 
discovery <- data.frame(discovery)
discovery_select <- select(discovery, features.var, converter)
discovery_select$converter <- as.factor(discovery_select$converter)

# external test set
external <- data.frame(external)
external_select <- select (external, features.var, converter)
external_select$converter <- as.factor(external_select$converter)
et_x <- subset(external_select, select=-converter)
et_y <- external_select$converter

# set variable for feature weight of svm linear model
wsmote=0
wrose=0

for (i in 1:permnum){
  set.seed(i)
  trainIndex<- createDataPartition(discovery_select$converter, p=0.75, list=FALSE)
  train <-discovery_select[trainIndex,]
  test <-discovery_select[-trainIndex,]
  
  ex_trainIndex<- createDataPartition(external_select$converter, p=0.75, list=FALSE)
  ex_train <-external_select[ex_trainIndex,]
  ex_test <-external_select[-ex_trainIndex,]
  
  
  # SMOTE
  smote <- SMOTE(converter ~ ., data=train, perc.over=100, perc.under=200)
  table(smote$converter)
  
  ex_smote <- SMOTE(converter ~ ., data=ex_train, perc.over=100, perc.under=200)
  
  # ROSE
  rose <- ROSE(converter~., data=train, N=500, seed=111)$data
  table(rose$converter)
  
  ex_rose <- ROSE(converter~., data=ex_train, N=500, seed=111)$data
  
  x <- subset(test, select=-converter)
  y <- test$converter
  

  # SVM smote
  svmlsmote <-svm(converter~., data=smote, kernel="linear", probability=TRUE,cross=10)
  svmpsmote <-svm(converter~., data=smote, kernel="polynomial", probability=TRUE,cross=10)
  svmrsmote <-svm(converter~., data=smote, kernel="radial", probability=TRUE,cross=10)
  svmssmote <-svm(converter~., data=smote, kernel="sigmoid", probability=TRUE,cross=10)
  
  svmlsmotepred <-predict(svmlsmote,x, decision.values=TRUE, probability=TRUE)
  svmpsmotepred <-predict(svmpsmote,x, decision.values=TRUE, probability=TRUE)
  svmrsmotepred <-predict(svmrsmote,x, decision.values=TRUE, probability=TRUE)
  svmssmotepred <-predict(svmssmote,x, decision.values=TRUE, probability=TRUE)
  
  svmlsmotepred.prob <- attr(svmlsmotepred,"probabilities")[,2] 
  svmpsmotepred.prob <- attr(svmpsmotepred,"probabilities")[,2]
  svmrsmotepred.prob <- attr(svmrsmotepred,"probabilities")[,2] 
  svmssmotepred.prob <- attr(svmssmotepred,"probabilities")[,2]
  
  caret::confusionMatrix(y,svmlsmotepred,positive="1")
  caret::confusionMatrix(y,svmpsmotepred,positive="1")
  caret::confusionMatrix(y,svmrsmotepred,positive="1")
  caret::confusionMatrix(y,svmssmotepred,positive="1")
  
  svmlsmoteroc=roc(test$converter,svmlsmotepred.prob) #smote linear
  svmpsmoteroc=roc(test$converter,svmpsmotepred.prob) #smote poly
  svmrsmoteroc=roc(test$converter,svmrsmotepred.prob) #smote radial
  svmssmoteroc=roc(test$converter,svmssmotepred.prob) #smote sigmoid
  
  svmlsmoteroc.auc[i,] <- as.vector(ci.auc(svmlsmoteroc))
  svmpsmoteroc.auc[i,] <- as.vector(ci.auc(svmpsmoteroc))
  svmrsmoteroc.auc[i,] <- as.vector(ci.auc(svmrsmoteroc))
  svmssmoteroc.auc[i,] <- as.vector(ci.auc(svmssmoteroc))
  
  
  wsmote <- wsmote + t(svmlsmote$coefs) %*% svmlsmote$SV
  
  ## SVM ROSE
  svmlrose <-svm(converter~., data=rose, kernel="linear", probability=TRUE,cross=10)
  svmprose <-svm(converter~., data=rose, kernel="polynomial", probability=TRUE,cross=10)
  svmrrose <-svm(converter~., data=rose, kernel="radial", probability=TRUE,cross=10)
  svmsrose <-svm(converter~., data=rose, kernel="sigmoid", probability=TRUE,cross=10)
  
  svmlrosepred <-predict(svmlrose,x, decision.values=TRUE, probability=TRUE)
  svmprosepred <-predict(svmprose,x, decision.values=TRUE, probability=TRUE)
  svmrrosepred <-predict(svmrrose,x, decision.values=TRUE, probability=TRUE)
  svmsrosepred <-predict(svmsrose,x, decision.values=TRUE, probability=TRUE)
  
  svmlrosepred.prob <- attr(svmlrosepred,"probabilities")[,2] # 1-prob
  svmprosepred.prob <- attr(svmprosepred,"probabilities")[,2]
  svmrrosepred.prob <- attr(svmrrosepred,"probabilities")[,2] 
  svmsrosepred.prob <- attr(svmsrosepred,"probabilities")[,2]
  
  caret::confusionMatrix(y,svmlrosepred,positive="1")
  caret::confusionMatrix(y,svmprosepred,positive="1")
  caret::confusionMatrix(y,svmrrosepred,positive="1")
  caret::confusionMatrix(y,svmsrosepred,positive="1")
  
  svmlroseroc=roc(test$converter,svmlrosepred.prob) #rose linear
  svmproseroc=roc(test$converter,svmprosepred.prob) #rose poly
  svmrroseroc=roc(test$converter,svmrrosepred.prob) #rose radial
  svmsroseroc=roc(test$converter,svmsrosepred.prob) #rose sigmoid
  
  svmlroseroc.auc[i,] <- as.vector(ci.auc(svmlroseroc))
  svmproseroc.auc[i,] <- as.vector(ci.auc(svmproseroc))
  svmrroseroc.auc[i,] <- as.vector(ci.auc(svmrroseroc))
  svmsroseroc.auc[i,] <- as.vector(ci.auc(svmsroseroc))
  
  wrose <- wrose+t(svmlrose$coefs) %*% svmlrose$SV
  
  
  # external test
  
  # SVM smote
  ex_svmlsmote <-svm(converter~., data=ex_smote, kernel="linear", probability=TRUE,cross=10)
  ex_svmpsmote <-svm(converter~., data=ex_smote, kernel="polynomial", probability=TRUE,cross=10)
  ex_svmrsmote <-svm(converter~., data=ex_smote, kernel="radial", probability=TRUE,cross=10)
  ex_svmssmote <-svm(converter~., data=ex_smote, kernel="sigmoid", probability=TRUE,cross=10)
  
  # svm smote
  et_svmlsmotepred <-predict( ex_svmlsmote,et_x, decision.values=TRUE, probability=TRUE)
  et_svmpsmotepred <-predict( ex_svmpsmote,et_x, decision.values=TRUE, probability=TRUE)
  et_svmrsmotepred <-predict( ex_svmrsmote,et_x, decision.values=TRUE, probability=TRUE)
  et_svmssmotepred <-predict( ex_svmssmote,et_x, decision.values=TRUE, probability=TRUE)
  
  et_svmlsmotepred.prob <- attr(et_svmlsmotepred,"probabilities")[,2] # 1-prob
  et_svmpsmotepred.prob <- attr(et_svmpsmotepred,"probabilities")[,2]
  et_svmrsmotepred.prob <- attr(et_svmrsmotepred,"probabilities")[,2] 
  et_svmssmotepred.prob <- attr(et_svmssmotepred,"probabilities")[,2]
  
  caret::confusionMatrix(et_y,et_svmlsmotepred,positive="1")
  caret::confusionMatrix(et_y,et_svmpsmotepred,positive="1")
  caret::confusionMatrix(et_y,et_svmrsmotepred,positive="1")
  caret::confusionMatrix(et_y,et_svmssmotepred,positive="1")
  
  et_svmlsmoteroc=roc(external_select$converter,et_svmlsmotepred.prob) # linear
  et_svmpsmoteroc=roc(external_select$converter,et_svmpsmotepred.prob) # poly
  et_svmrsmoteroc=roc(external_select$converter,et_svmrsmotepred.prob) # radial
  et_svmssmoteroc=roc(external_select$converter,et_svmssmotepred.prob) # sigmoid
  
  et_svmlsmoteroc.auc[i,] <- as.vector(ci.auc(et_svmlsmoteroc))
  et_svmpsmoteroc.auc[i,] <- as.vector(ci.auc(et_svmpsmoteroc))
  et_svmrsmoteroc.auc[i,] <- as.vector(ci.auc(et_svmrsmoteroc))
  et_svmssmoteroc.auc[i,] <- as.vector(ci.auc(et_svmssmoteroc))
  
  ## SVM ROSE
  et_svmlrose <-svm(converter~., data=ex_rose, kernel="linear", probability=TRUE,cross=10)
  et_svmprose <-svm(converter~., data=ex_rose, kernel="polynomial", probability=TRUE,cross=10)
  et_svmrrose <-svm(converter~., data=ex_rose, kernel="radial", probability=TRUE,cross=10)
  et_svmsrose <-svm(converter~., data=ex_rose, kernel="sigmoid", probability=TRUE,cross=10)
  
  # svm rose
  et_svmlrosepred <-predict(et_svmlrose,et_x, decision.values=TRUE, probability=TRUE)
  et_svmprosepred <-predict(et_svmprose,et_x, decision.values=TRUE, probability=TRUE)
  et_svmrrosepred <-predict(et_svmrrose,et_x, decision.values=TRUE, probability=TRUE)
  et_svmsrosepred <-predict(et_svmsrose,et_x, decision.values=TRUE, probability=TRUE)
  
  et_svmlrosepred.prob <- attr(et_svmlrosepred,"probabilities")[,2] # 1-prob
  et_svmprosepred.prob <- attr(et_svmprosepred,"probabilities")[,2]
  et_svmrrosepred.prob <- attr(et_svmrrosepred,"probabilities")[,2] 
  et_svmsrosepred.prob <- attr(et_svmsrosepred,"probabilities")[,2]
  
  caret::confusionMatrix(et_y,et_svmlrosepred,positive="1")
  caret::confusionMatrix(et_y,et_svmprosepred,positive="1")
  caret::confusionMatrix(et_y,et_svmrrosepred,positive="1")
  caret::confusionMatrix(et_y,et_svmsrosepred,positive="1")
  
  et_svmlroseroc=roc(external_select$converter,et_svmlrosepred.prob) # linear
  et_svmproseroc=roc(external_select$converter,et_svmprosepred.prob) # poly
  et_svmrroseroc=roc(external_select$converter,et_svmrrosepred.prob) # radial
  et_svmsroseroc=roc(external_select$converter,et_svmsrosepred.prob) # sigmoid
  
  et_svmlroseroc.auc[i,] <- as.vector(ci.auc(et_svmlroseroc))
  et_svmproseroc.auc[i,] <- as.vector(ci.auc(et_svmproseroc))
  et_svmrroseroc.auc[i,] <- as.vector(ci.auc(et_svmrroseroc))
  et_svmsroseroc.auc[i,] <- as.vector(ci.auc(et_svmsroseroc))
}

