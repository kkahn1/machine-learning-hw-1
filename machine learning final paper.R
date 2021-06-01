# packages needed: dplyr,randomForestSRC, survival
# loaded throughout


library(dplyr)
edtg_groupyear <- read.csv("https://raw.githubusercontent.com/kkahn1/machine-learning-hws/main/EDTG_Replication%20Data.csv")
edtg_landvars_select <- edtg_groupyear %>%
  dplyr::select(gid,y_start,end,duration,num_base,EAP,ECA,LAC,MENA,NA.,SAS,SSA,
         ethnic,language,religion,elevation,tropics,landlock,polity,mul_bases,
         pop,poppermillion,gdp,gdppercapita,tradeopenness2010usd,logelvation,
         GovernmentspendingGDP2010us,competitor,end)
# aggregating data to cross sectional 
# remove(mean) 
edtg_landvars <- edtg_landvars_select %>%
  group_by(gid) %>%
  summarise(across(y_start, min,na.rm=TRUE),
            across(c(end,duration,EAP,ECA,LAC,MENA,NA.,SAS,SSA,landlock,mul_bases), max,na.rm=TRUE),
            across(c(num_base,ethnic,language,religion,elevation,tropics,polity,pop,
                     gdp,gdppercapita,tradeopenness2010usd,
                     GovernmentspendingGDP2010us,competitor),mean,na.rm=TRUE))
edtg_landvars <- as.data.frame(edtg_landvars)
edtg_landvars <- edtg_landvars %>% mutate_all(~replace(., is.nan(.), NA))
edtg_landvars$gid <- NULL

# test and train
seed <- 9986000
set.seed(seed)
partition <- sample(1:nrow(edtg_landvars),nrow(edtg_landvars)*.7)
traindf <- edtg_landvars[partition,]
testdf <- edtg_landvars[-partition,]

library(randomForestSRC)
set.seed(seed)
makeSurvTask()



rfTune <- randomForestSRC::tune(Surv(time = duration, event = end)~., data = traindf,
                                mtryStart = 5, nodesizeTry = c(1:15), ntreeTry = 400, 
                                sampsize = function(x){(x * .632)}, nsplit = 10,
                                maxIter = 100, trace = TRUE)
save(rfTune, file = "rfsrc_mlfinal.rdata")

rfsrcTrained <- randomForestSRC::rfsrc(Surv(time = duration, event = end)~., data = traindf,
                                       ntree = 500, mtry = 15, nodesize = 1, 
                                       forest = TRUE, block.size = 10, importance = TRUE) 
rfsrcPreds <- randomForestSRC::predict.rfsrc(rfsrcTrained, newdata = testdf, importance = TRUE)
rfsrcPredError <- rfsrcPreds$err.rate
rfsrcPredImp <- rfsrcPreds$importance
rfsrcTrained$err.rate
rfsrcPreds$p


# plot training data
plot.rfsrc(rfsrcTrained, verbose = TRUE,plots.one.page = FALSE)
plot.survival.rfsrc(rfsrcTrained,plots.one.page = TRUE)
# plot test data
plot.rfsrc(rfsrcPreds,verbose = TRUE,plots.one.page = FALSE)
plot.survival.rfsrc(rfsrcPreds)

library(ggRandomForests)
plot(gg_vimp(rfsrcTrained))














# categorical and continuous predictors
# predicting end, not duration
# rare event... gradient boost
# categories are integer dummies

# preparing dataset for gradient boosted tree
xgbdata <- edtg_groupyear %>%
  dplyr::select(gid,y_start,end,duration,num_base,EAP,ECA,LAC,MENA,NA.,SAS,SSA,
                ethnic,language,religion,elevation,tropics,landlock,polity,mul_bases,
                pop,poppermillion,gdp,gdppercapita,tradeopenness2010usd,logelvation,
                GovernmentspendingGDP2010us,competitor,end) %>%
  mutate(across(c(EAP,ECA,LAC,NA.,SSA,SAS,landlock,MENA),factor))
str(xgbdata)
set.seed(seed)
library(caret)
xgbdata$end <- factor(xgbdata$end)

xgbpartition <- createDataPartition(xgbdata$end, p=.7,list=FALSE)
xgbtraindf <- xgbdata[xgbpartition,]
xgbtestdf <- xgbdata[-xgbpartition,]

imputeMethod <- imputeLearner("regr.rpart")
imputeMethodDummy <- imputeLearner("classif.rpart")
set.seed(seed)
imputingtrain <- impute(xgbtraindf, target = "end", classes = list(numeric = imputeMethod, integer = imputeMethod, factor = imputeMethodDummy))
dfTrainImputed <- imputingtrain$data %>% select(-c(y_start,gid))
dfTrainImputed <- mutate_at(dfTrainImputed, .vars = vars(-end), .funs = as.numeric)
reimputingtest <- reimpute(xgbtestdf, imputingtrain$desc) %>% select(-c(y_start,gid))
dfTestImputed <- mutate_at(reimputingtest, .vars = vars(-end), .funs = as.numeric)

library(rms)



# create task
xgbTrainTask<-makeClassifTask(data=dfTrainImputed,target="end",positive=1)
xgbTestTask<-makeClassifTask(data=dfTestImputed,target="end")

# make learner
xgbLearner<-makeLearner("classif.xgboost", predict.type="prob")
xgbParamSpace <- makeParamSet(
  makeNumericParam("eta", lower = 0, upper = 1),
  makeNumericParam("gamma", lower = 0, upper = 5),
  makeIntegerParam("max_depth", lower = 1, upper = 5),
  makeNumericParam("min_child_weight", lower = 1, upper = 10),
  makeNumericParam("subsample", lower = 0.5, upper = 1),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
  makeIntegerParam("nrounds", lower = 100, upper = 100),
  makeDiscreteParam("eval_metric", values = c("auc", "error")))
randSearch <- makeTuneControlRandom(maxit = 500)
cvForTuning <- makeResampleDesc("CV", iters = 5)
# tune
xgbTuned <- tuneParams(xgbLearner, task = xgbTrainTask,
                     resampling = cvForTuning,
                     par.set = xgbParamSpace,
                     control = randSearch)
save(xgbTuned, file = "xgboost_mlfinal.rdata")
xgbSetParams <- setHyperPars(xgbLearner, par.vals = xgbTuned$x)
xgbTrained <- mlr::train(xgbSetParams, xgbTrainTask)
xgbModelData <- getLearnerModel(xgbTrained)
ggplot(xgbModelData$evaluation_log, aes(iter, train_auc)) +
  geom_line() +
  geom_point()
listMeasures(xgbTrainTask)

xgbPredProb<-predict(xgbTrained,xgbTestTask,type="prob")
library(ROCR)
rocrPred <- prediction(xgbPredProb$data$prob.1,xgbPredProb$data$truth)
rocrPerf <- ROCR::performance(rocrPred, "prec", "rec")
rocraucpr<- ROCR::performance(rocrPred, "aucpr")
plot(rocrPerf, colorize = TRUE)
rocraucpr@y.values
ROCR::performance(rocrPred, "tpr")@y.values

xgbconfmat <- table(xgbPredProb$data$truth, xgbPredProb$data$response)
prop.table(xgbconfmat,1) 



