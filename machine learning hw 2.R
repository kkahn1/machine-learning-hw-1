# Machine Learning Homework 2

############################
library(caret)
library(mlr)
library(ROCR)
library(parallelMap)
library(parallel)
library(dplyr)
# reading in file and reproduce
hwdf<-read.csv("https://raw.githubusercontent.com/kkahn1/machine-learning-hw-1/main/edtg_hw1_table4.csv")
set.seed(9986000)
keep<-with(hwdf,c("dstar","x1","x2","left","nat","right","shr_trans","diversity","num_base",
                  "lpop","polity1","polity1sqr","ethnic","ethn2","EAP","ECA","LAC","NA.","SAS","SSA",
                  "lelev","tropics","landlock"))
hwdf<-hwdf%>%
  select(all_of(keep))
hwdf$dstar<-as.factor(hwdf$dstar)
hwdf<-na.omit(hwdf)
# so i can do two class summary
levels(hwdf$dstar)<-c("active","end")
# reproduce
reproduce<-glm(dstar~x1+x2+left+nat+right+shr_trans+diversity+num_base+
                 lpop+polity1+polity1sqr+ethnic+ethn2+EAP+ECA+LAC+NA.+SAS+SSA+
                 lelev+tropics+landlock, family=binomial(link="logit"),data=hwdf)
summary(reproduce)
# split data
partition<-createDataPartition(hwdf$dstar,p=.7,list=FALSE)
train<-hwdf[partition,]
test<-hwdf[-partition,]

######################
set.seed(9986000)
# logit
trainlogit <- train
testlogit <- test
levels(trainlogit$dstar)<-c(0,1)
levels(testlogit$dstar)<-c(0,1)
logittraintask <- makeClassifTask(data=trainlogit,target="dstar", positive = 1)
logittesttask <- makeClassifTask(data = testlogit, target = "dstar", positive = 1)
logLearn <- makeLearner("classif.logreg", predict.type = "prob")
lrLearn <- makeWeightedClassesWrapper(logLearn, wcw.weight = 10)
logModel <- train(lrLearn, logittraintask)
predLogit <- data.frame(predict(logModel, logittesttask))
confusionmatrixLogit <- table(predLogit$truth, predLogit$response)
prop.table(confusionmatrixLogit, 1) 

#####################
set.seed(9986000)
# SVM
# set task
svmtraintask <- makeClassifTask(data=train,target="dstar", positive = "end")
svmtesttask <- makeClassifTask(data = test, target = "dstar", positive = "end")
# set learner
# give .01 weight to the positive class of end
svmlearner <- makeLearner("classif.svm", predict.type = "prob")
wcw.learn <- makeWeightedClassesWrapper(svmlearner, wcw.weight = 10)
# parameter space
svmParamSpace <- makeParamSet(
  makeDiscreteParam("kernel", values = c("polynomial", "radial", "sigmoid")),
  makeIntegerParam("degree", lower = 1, upper = 3),
  makeNumericParam("cost", lower=7, upper=10),
  makeNumericParam("gamma", lower=5, 8))
# set search function
svmrandSearch <- makeTuneControlRandom(maxit = 15)
# set cross validation
cvForTuning = makeResampleDesc("CV", iters = 2, stratify = TRUE)
# tuning
parallelStart(mode="multicore",cpus=parallel::detectCores())
tunedSvmPars <- tuneParams(wcw.learn, task = svmtraintask,
                           resampling = cvForTuning,
                           par.set = svmParamSpace,
                           control = svmrandSearch,
                           measures = ber) 
save(tunedSvmPars, file = "mlhw2_objects.rdata")
parallelStop()
# 
tunedSvmPars
# training with tuned hyperparameters
tunedSvm <- setHyperPars(wcw.learn, par.vals = tunedSvmPars$x)
tunedSvmModel <- train(tunedSvm, svmtraintask)
svmPredict <- data.frame(predict(tunedSvmModel, svmtesttask))
confusionmatrixsvm <- table(svmPredict$truth, svmPredict$response)
prop.table(confusionmatrixsvm, 1) 


#################
# random forest
set.seed(9986000)
# set task
foresttraintask <- makeClassifTask(data=train, target="dstar", positive = "end")
foresttesttask <- makeClassifTask(data=test, target="dstar", positive = "end")
# create learner
flearner <- makeLearner("classif.randomForest", predict.type = "prob")
forestLearner <- makeWeightedClassesWrapper(flearner, wcw.weight = 10)
# set parameter space
forestParamSpace <- makeParamSet(                        
  makeIntegerParam("ntree", lower = 200, upper = 200),
  makeIntegerParam("mtry", lower = 15, upper = 19),
  makeIntegerParam("nodesize", lower = 3, upper = 5),
  makeIntegerParam("maxnodes", lower = 15, upper = 50))
# set random search
randSearchForest<- makeTuneControlRandom(maxit = 50L)
# cv is same 
forestCV <- makeResampleDesc("CV", iters = 5L, stratify=TRUE)
parallelStart(mode="multicore",cpus=parallel::detectCores())
tunedForestPars <- tuneParams(forestLearner, task = foresttraintask,     
                              resampling = forestCV,     
                              par.set = forestParamSpace,   
                              control = randSearchForest,
                              measures = ber)
save(tunedForestPars, file = "mlhw2_objects2.rdata")
parallelStop()
tunedForestPars
tunedForest <- setHyperPars(forestLearner, par.vals = tunedForestPars$x)
tunedForestModel <- train(tunedForest, foresttraintask)
# predict
forestPredict <- data.frame(predict(tunedForestModel, foresttesttask))
confusionmatrixForest <- table(forestPredict$truth, forestPredict$response)
prop.table(confusionmatrixForest, 1) 

#############
# neural network
set.seed(9986000)
# set task
nntraintask <- makeClassifTask(data=train, target="dstar", positive = "end")
nntesttask <- makeClassifTask(data=test, target="dstar", positive = "end")
# create learner
nnlearner <- makeLearner("classif.nnet", predict.type = "prob")
nnLearner <- makeWeightedClassesWrapper(nnlearner, wcw.weight = 10)
getParamSet("classif.nnet")
# set parameter space
nnParamSpace <- makeParamSet(                        
  makeIntegerParam("size", lower = 1, upper = 10),
  makeNumericParam("decay", lower = .01, upper = 2),
  makeIntegerParam("maxit", lower = 500, upper = 500))
# set random search
randSearchnn<- makeTuneControlRandom(maxit = 30L)
# cv is same 
nnCV <- makeResampleDesc("CV", iters = 5L, stratify=TRUE)
parallelStart(mode="multicore",cpus=parallel::detectCores())
tunedNNPars <- tuneParams(nnLearner, task = nntraintask,     
                              resampling = nnCV,     
                              par.set = nnParamSpace,   
                              control = randSearchnn,
                          measures = ber)
save(tunedNNPars, file = "mlhw2_objects3.rdata")
parallelStop()
tunedNNPars
tunedNN <- setHyperPars(nnLearner, par.vals = tunedNNPars$x)
tunedNNModel <- train(tunedNN, nntraintask)
# predict
nnPredict <- data.frame(predict(tunedNNModel, nntesttask))
confusionmatrixnn <- table(nnPredict$truth, nnPredict$response)
prop.table(confusionmatrixnn, 1) 


# predictions for aupr

rocr.logit <- prediction(predLogit$prob.1, testlogit$dstar)
rocr.rf <- prediction(forestPredict$prob.end, test$dstar)
rocr.nn <- prediction(nnPredict$prob.end, test$dstar)
prauLogit<-ROCR::performance(rocr.logit,"aucpr")
prauRF<-ROCR::performance(rocr.rf,"aucpr")
prauNN<-ROCR::performance(rocr.nn,"aucpr")
prauLogit@y.values # .196691
prauRF@y.values # .08953794
prauNN@y.values #  .1519851

# Recall-Precision curve  
par(mfrow=c(2,2))
plot(ROCR::performance(rocr.logit,"prec","rec"), main = "Logit AUPR: 0.196691")
plot(ROCR::performance(rocr.rf,"prec","rec"), main = "RF AUPR: 0.08953794")
plot(ROCR::performance(rocr.nn,"prec","rec"), main = "NN AUPR: 0.1519851")


#######################
# redo pipeline removing variables
logitdata <- hwdf
levels(logitdata$dstar)<-c(0,1)

#keep<-with(hwdf,c("dstar","x1","x2","left","nat","right","shr_trans","diversity","num_base",
                  #"lpop","polity1","polity1sqr","ethnic","ethn2","EAP","ECA","LAC","NA.","SAS","SSA",
                  #"lelev","tropics","landlock"))

set.seed(9986000)
# remove duration
no.x <- makeClassifTask(data = logitdata%>%select(-c("x1","x2")), target = "dstar", positive = 1 )
# stays the same
#logLearn <- makeLearner("classif.logreg", predict.type = "prob")
#lrLearn <- makeWeightedClassesWrapper(logLearn, wcw.weight = 10)
logModel.x <- train(lrLearn, no.x)
predLogit.x <- data.frame(predict(logModel.x, no.x))

no.orientation <- makeClassifTask(data = logitdata%>%select(-c("left","nat","right")), target = "dstar", positive = 1 )
logModel.orientation <- train(lrLearn, no.orientation)
predLogit.orientation <- data.frame(predict(logModel.orientation, no.orientation))

no.trnsn <- makeClassifTask(data = logitdata%>%select(-c("shr_trans")), target = "dstar", positive = 1 )
logModel.trnsn <- train(lrLearn, no.trnsn)
predLogit.trnsn <- data.frame(predict(logModel.trnsn, no.trnsn))

no.div <- makeClassifTask(data = logitdata%>%select(-c("diversity")), target = "dstar", positive = 1 )
logModel.div <- train(lrLearn, no.div)
predLogit.div <- data.frame(predict(logModel.div, no.div))
cmLogit.div <- table(predLogit.div$truth, predLogit.div$response)
prop.table(cmLogit.div, 1)

no.base <- makeClassifTask(data = logitdata%>%select(-c("num_base")), target = "dstar", positive = 1 )
logModel.base <- train(lrLearn, no.base)
predLogit.base <- data.frame(predict(logModel.base, no.base))

no.lpop <- makeClassifTask(data = logitdata%>%select(-c("lpop")), target = "dstar", positive = 1 )
logModel.lpop <- train(lrLearn, no.lpop)
predLogit.lpop <- data.frame(predict(logModel.lpop, no.lpop))

no.polity <- makeClassifTask(data = logitdata%>%select(-c("polity1","polity1sqr")), target = "dstar", positive = 1 )
logModel.polity <- train(lrLearn, no.polity)
predLogit.polity <- data.frame(predict(logModel.polity, no.polity))

no.ethnic <- makeClassifTask(data = logitdata%>%select(-c("ethnic","ethn2")), target = "dstar", positive = 1 )
logModel.ethnic <- train(lrLearn, no.ethnic)
predLogit.ethnic <- data.frame(predict(logModel.ethnic, no.ethnic))

no.region <- makeClassifTask(data = logitdata%>%select(-c("EAP","ECA","LAC","NA.","SAS","SSA")), target = "dstar", positive = 1 )
logModel.region <- train(lrLearn, no.region)
predLogit.region <- data.frame(predict(logModel.region, no.region))
cmLogit.region <- table(predLogit.region$truth, predLogit.region$response)
prop.table(cmLogit.region, 1)

no.land <- makeClassifTask(data = logitdata%>%select(-c("lelev","tropics","landlock")), target = "dstar", positive = 1 )
logModel.land <- train(lrLearn, no.land)
predLogit.land <- data.frame(predict(logModel.land, no.land))
all <- makeClassifTask(data = logitdata, target = "dstar", positive = 1 )
logModel.all <- train(lrLearn, all)
predLogit.all <- data.frame(predict(logModel.all, all))

pr <- c(prauc(logitdata$dstar,predLogit.all$prob.1,"1"),prauc(logitdata$dstar,predLogit.x$prob.1,"1"),
        prauc(logitdata$dstar,predLogit.orientation$prob.1,"1"),prauc(logitdata$dstar,predLogit.trnsn$prob.1,"1"),
        prauc(logitdata$dstar,predLogit.div$prob.1,"1"),prauc(logitdata$dstar,predLogit.base$prob.1,"1"),
        prauc(logitdata$dstar,predLogit.lpop$prob.1,"1"),prauc(logitdata$dstar,predLogit.polity$prob.1,"1"),
        prauc(logitdata$dstar,predLogit.ethnic$prob.1,"1"),prauc(logitdata$dstar,predLogit.region$prob.1,"1"),
        prauc(logitdata$dstar,predLogit.land$prob.1,"1"))
vars <- c("all","duration","orientation","share_transnat","attack_diversity","number_bases","log pop","polity",
          "ethnic","region","land")
######
# nnet pipeline
# neural network
set.seed(9986000)
# predict
nnPredict.all <- data.frame(predict(train(tunedNN,makeClassifTask(data=hwdf, target="dstar", positive = "end")), newdata = hwdf))
nnPredict.x <- data.frame(predict(train(tunedNN, makeClassifTask(data=hwdf%>%select(-c("x1","x2")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("x1","x2"))))
nnPredict.orientation <- data.frame(predict(train(tunedNN, makeClassifTask(data=hwdf%>%select(-c("left","nat","right")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("left","nat","right"))))
nnPredict.trnsn <- data.frame(predict(train(tunedNN, makeClassifTask(data=hwdf%>%select(-c("shr_trans")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("shr_trans"))))
nnPredict.div <- data.frame(predict(train(tunedNN, makeClassifTask(data=hwdf%>%select(-c("diversity")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("diversity"))))
nnPredict.base <- data.frame(predict(train(tunedNN, makeClassifTask(data=hwdf%>%select(-c("num_base")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("num_base"))))
nnPredict.pop <- data.frame(predict(train(tunedNN, makeClassifTask(data=hwdf%>%select(-c("lpop")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("lpop"))))
nnPredict.polity <- data.frame(predict(train(tunedNN, makeClassifTask(data=hwdf%>%select(-c("polity1","polity1sqr")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("polity1","polity1sqr"))))
nnPredict.ethnic <- data.frame(predict(train(tunedNN, makeClassifTask(data=hwdf%>%select(-c("ethnic","ethn2")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("ethnic","ethn2"))))
nnPredict.region <- data.frame(predict(train(tunedNN, makeClassifTask(data=hwdf%>%select(-c("EAP","ECA","LAC","NA.","SAS","SSA")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("EAP","ECA","LAC","NA.","SAS","SSA"))))
nnPredict.land <- data.frame(predict(train(tunedNN, makeClassifTask(data=hwdf%>%select(-c("lelev","tropics","landlock")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("lelev","tropics","landlock"))))

prnn <- c(prauc(hwdf$dstar,nnPredict.all$prob.end,"end"),prauc(hwdf$dstar,nnPredict.x$prob.end,"end"),
        prauc(hwdf$dstar,nnPredict.orientation$prob.end,"end"),prauc(hwdf$dstar,nnPredict.trnsn$prob.end,"end"),
        prauc(hwdf$dstar,nnPredict.div$prob.end,"end"),prauc(hwdf$dstar,nnPredict.base$prob.end,"end"),
        prauc(hwdf$dstar,nnPredict.pop$prob.end,"end"),prauc(hwdf$dstar,nnPredict.polity$prob.end,"end"),
        prauc(hwdf$dstar,nnPredict.ethnic$prob.end,"end"),prauc(hwdf$dstar,nnPredict.region$prob.end,"end"),
        prauc(hwdf$dstar,nnPredict.land$prob.end,"end"))

# rf
rfPredict.all <- data.frame(predict(train(tunedForest,makeClassifTask(data=hwdf, target="dstar", positive = "end")), newdata = hwdf))
rfPredict.x <- data.frame(predict(train(tunedForest, makeClassifTask(data=hwdf%>%select(-c("x1","x2")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("x1","x2"))))
rfPredict.orientation <- data.frame(predict(train(tunedForest, makeClassifTask(data=hwdf%>%select(-c("left","nat","right")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("left","nat","right"))))
rfPredict.trnsn <- data.frame(predict(train(tunedForest, makeClassifTask(data=hwdf%>%select(-c("shr_trans")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("shr_trans"))))
rfPredict.div <- data.frame(predict(train(tunedForest, makeClassifTask(data=hwdf%>%select(-c("diversity")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("diversity"))))
rfPredict.base <- data.frame(predict(train(tunedForest, makeClassifTask(data=hwdf%>%select(-c("num_base")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("num_base"))))
rfPredict.pop <- data.frame(predict(train(tunedForest, makeClassifTask(data=hwdf%>%select(-c("lpop")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("lpop"))))
rfPredict.polity <- data.frame(predict(train(tunedForest, makeClassifTask(data=hwdf%>%select(-c("polity1","polity1sqr")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("polity1","polity1sqr"))))
rfPredict.ethnic <- data.frame(predict(train(tunedForest, makeClassifTask(data=hwdf%>%select(-c("ethnic","ethn2")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("ethnic","ethn2"))))
rfPredict.region <- data.frame(predict(train(tunedForest, makeClassifTask(data=hwdf%>%select(-c("EAP","ECA","LAC","NA.","SAS","SSA")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("EAP","ECA","LAC","NA.","SAS","SSA"))))
rfPredict.land <- data.frame(predict(train(tunedForest, makeClassifTask(data=hwdf%>%select(-c("lelev","tropics","landlock")), target="dstar", positive = "end")), newdata = hwdf%>%select(-c("lelev","tropics","landlock"))))

prff <- c(prauc(hwdf$dstar,rfPredict.all$prob.end,"end"),prauc(hwdf$dstar,rfPredict.x$prob.end,"end"),
          prauc(hwdf$dstar,rfPredict.orientation$prob.end,"end"),prauc(hwdf$dstar,rfPredict.trnsn$prob.end,"end"),
          prauc(hwdf$dstar,rfPredict.div$prob.end,"end"),prauc(hwdf$dstar,rfPredict.base$prob.end,"end"),
          prauc(hwdf$dstar,rfPredict.pop$prob.end,"end"),prauc(hwdf$dstar,rfPredict.polity$prob.end,"end"),
          prauc(hwdf$dstar,rfPredict.ethnic$prob.end,"end"),prauc(hwdf$dstar,rfPredict.region$prob.end,"end"),
          prauc(hwdf$dstar,rfPredict.land$prob.end,"end"))

variables <- data.frame(vars,pr,prnn,prff)
variables$pr <- round(variables$pr, 4)
variables$prnn <- round(variables$prnn, 4)
variables$prff <- round(variables$prff, 4)
xtable(variables)
