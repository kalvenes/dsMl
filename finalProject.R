#
library(caret)
library(gbm)
library(e1071)
library(randomForest)
#
set.seed(10000)
#
# Read data from file --- assumed to be donwloaded and in local directory
#
pmlData <- read.csv("pml-training.csv",header=TRUE)
#keepCols <- c(1,2,3,4,5,6,7,8,9,10,11,37,38,39,40,41,42,43,44,45,46,47,48,49,60,61,62,63,64,65,66,67,68,84,85,86,
#              102,113,114,115,116,117,118,119,120,121,122,123,124,140,151,152,153,154,155,156,157,158,159,160)
#
# Keep only columns that have data. Also, drop columns related to subject identity, start times, etc.
#
keepCols <- c(2,37,38,39,40,41,42,43,44,45,46,47,48,49,60,61,62,63,64,65,66,67,68,84,85,86,
              102,113,114,115,116,117,118,119,120,121,122,123,124,140,151,152,153,154,155,156,157,158,159,160)
pmlDataRedux <- pmlData[,keepCols]
#
# Split data set into train (60%), test (20%), and validate (20%)
#
inTrain <- createDataPartition(y=pmlDataRedux$classe,p=0.6,list=FALSE)
training <- pmlDataRedux[inTrain,] # 60% of sample for training
testingValidating <- pmlDataRedux[-inTrain,]
inTest <- createDataPartition(y=testingValidating$classe,p=0.5,list=FALSE)
testing <- testingValidating[inTest,] # 20% of sample for testing
validating <- testingValidating[-inTest,] # 20% of sample for validation
#
# With 54 variables, dimensionality reduction might be necessary. It will
# speed up convergence and eliminate correlation in the predictors (for
# regression models).
#
transPCA <- preProcess(training[,-54],method="pca",thresh=0.9)
trainPCA <- predict(transPCA,training)
testPCA <- predict(transPCA,testing)
validPCA <- predict(transPCA,validating)
#
# Fit three models:
# - Random forest
# - Linear discriminant analysis
# - Support vector machine (with radial kernel)
#
modFitLda <- train(classe~.,method="lda",data=trainPCA)
modFitSvm <- svm(classe~.,trainPCA,model="radial")
modFitRf <- train(classe~.,method="rf",data=trainPCA,ntree=100)
#
# Predict outcome on test data for each model
#
predictLda <- predict(modFitLda,testPCA)
predictSvm <- predict(modFitSvm, testPCA)
predictRf <- predict(modFitRf,testPCA)
#
# Display accuracy of each prediction on test data
#
confusionMatrix(testing$classe,predictLda)$overall
confusionMatrix(testing$classe,predictSvm)$overall
confusionMatrix(testing$classe,predictRf)$overall
#
# Collate predictions from all models to create a random forest ensemble model
#
stackedData <- data.frame(predictSvm,predictRf,classe=testing$classe)
modFitStacked <- train(classe~.,method="rf",data=stackedData,ntree=100)
#
# Use validation sample to estimate ensemble model accuracy
#
validationStacked <- predict(modFitStacked,validPCA)
confusionMatrix(validating$classe,validationStacked)$overall
#
# Read data for 20 test cases for forecasting
#
pmlForecastData <- read.csv("pml-testing.csv")
pmlForecastData <- pmlForecastData[,keepCols]
pmlFCPCA <- predict(transPCA,pmlForecastData)
colnames(pmlFCPCA)[2] <- "classe" # rename column to match name in previously trained models
#
# Match up factor levels for SVM model
#
ufactor <- levels(trainPCA$user_name)
cfactor <- levels(trainPCA$classe)
levels(pmlFCPCA$user_name) <- ufactor
levels(pmlFCPCA$classe) <- cfactor
#
# Forecast each of the individual models, then collate and forecast the ensemble model
#
predictTestSvm <- predict(modFitSvm, pmlFCPCA)
predictTestRf <- predict(modFitRf,pmlFCPCA)
#
stackedTestData <- data.frame(predictTestSvm,predictTestRf,classe=pmlFCPCA$classe)
#
# Re-name columns to fit training model (to avoid confusion)
#
colnames(stackedTestData) <- colnames(stackedData)
#
forecastTest <- predict(modFitStacked,stackedTestData)
