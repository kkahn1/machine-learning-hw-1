library(dplyr)
library(tidyverse)
library(mlr)
library(parallel)
library(parallelMap)
library(ggplot2)
library(mboost)
hw3df <- read.csv("https://raw.githubusercontent.com/kkahn1/machine-learning-hws/main/Huber%20JCR%20Replication%20Crossnational%20Data.csv")
# variables to keep
df_select <- hw3df %>%
  select("cowcode","iyear","LegislaturePctgreenlag","domestic_civilian_attacksproportionlag1","cllag","prlag","GDPlag1log","cinc.ylag1",
                "Populationlag1log","war_anylag","ythbullag","muslim_maj","coldwar","domestic_total_attackslag")
# rename variables
df_select <- df_select %>%
  rename(leg_pct=LegislaturePctgreenlag, civ_atks=domestic_civilian_attacksproportionlag1, gdp=GDPlag1log,
         cinc=cinc.ylag1, pop=Populationlag1log, war=war_anylag, total_atks=domestic_total_attackslag, youth=ythbullag)

# part 1
seed <- 9987000
set.seed(seed)
outendog3 <- lm(leg_pct ~ civ_atks + cllag + prlag + gdp + cinc + pop + war + youth + 
                  muslim_maj + coldwar + total_atks, data=df_select)
summary(outendog3)
library(stargazer)
stargazer(outendog3)

 #############
##### some data cleaning for imputation
# remove NAs on the target variable
df_select <- df_select %>%
  filter(is.na(leg_pct) == FALSE)
# factor dummies for imputation. otherwise it will throw in some 2s among the 0s and 1s
df_select[,c("coldwar","war","muslim_maj")] <- lapply(df_select[,c("coldwar","war","muslim_maj")], factor)

set.seed(seed)
# train test split
partition <- sample(1:nrow(df_select),nrow(df_select)*.7)
train <- df_select[partition,]
test <- df_select[-partition,]

# impute train data
imputeMethod <- imputeLearner("regr.rpart")
imputeMethodDummy <- imputeLearner("classif.rpart")
trainImputed <- impute(train, target = "leg_pct", 
                       classes = list(numeric = imputeMethod, integer = imputeMethod, factor = imputeMethodDummy))
summary(trainImputed$data)
testImputed <- reimpute(test, trainImputed$desc)
# remove country year from data
trainImputedFinal <- trainImputed$data
trainImputedFinal <- trainImputedFinal %>%
  select(-c("cowcode","iyear"))
testImputedFinal <- testImputed
testImputedFinal <- testImputedFinal %>%
  select(-c("cowcode","iyear"))



##########################
# linear
trainedLinear <- lm(leg_pct ~ civ_atks + cllag + prlag + gdp + cinc + pop + war + youth + 
                      muslim_maj + coldwar + total_atks, data = trainImputedFinal)
olsPred <- predict(trainedLinear,newdata = testImputedFinal)
olsRes <- olsPred - testImputedFinal$leg_pct
olsRMSE <- sqrt(mean(olsRes^2))
olsPredDf <-
  as_tibble(predict(trainedLinear,newdata = testImputedFinal)) %>%
  bind_cols(testImputedFinal) %>%
  mutate(error = value - leg_pct,
         error_sq = error ^ 2,
         error_abs = abs(error),
         sum_abs_error = sum(error_abs),
         mae = sum_abs_error / n(),
         sum_sq_error = sum(error_sq),
         rmse = sqrt(sum_sq_error / n()))
cat("ols: ", unique(predLinear$mae), "\n") #imputing is slightly better than removing NAs. 6 vs 8

library(MLmetrics)
RMSE(olsPred,testImputedFinal$leg_pct)
MAE(olsPred,testImputedFinal$leg_pct)
R2_Score(olsPred,testImputedFinal$leg_pct)



#######################
# plotting predictors against target
train2 <- train
train2$coldwar <- as.numeric(levels(train$coldwar))[train$coldwar]
train2$war_anylag <- as.numeric(levels(train$war_anylag))[train$war_anylag]
train2$muslim_maj <- as.numeric(levels(train$muslim_maj))[train$muslim_maj]
trainUntidy <- gather(train2, key = "variable", value = "value", -c(LegislaturePctgreenlag))
ggplot(trainUntidy, aes(value, LegislaturePctgreenlag)) +
  facet_wrap(~ variable, scale = "free_x") +
  geom_point() +
  geom_smooth() +
  theme_bw()
#####################

# gam
set.seed(seed)
gamTask <- makeRegrTask(data = trainImputedFinal, target = "leg_pct")
gamLearner <- makeLearner("regr.gamboost")
gamFeatSelControl <- makeFeatSelControlSequential(method = "sfbs")
cv <- makeResampleDesc("CV", iters = 5)
gamFeatSelWrapper <- makeFeatSelWrapper(learner = gamLearner, resampling = cv, control = gamFeatSelControl)
parallelStart(mode="multicore",cpus=parallel::detectCores())
gamTrained <- mlr::train(gamFeatSelWrapper, gamTask)
save(gamTrained, file = "mlhw3_gam.rdata")
parallelStop()

gamData <- getLearnerModel(gamTrained, more.unwrap = TRUE)
summary(getLearnerModel(gamTrained))

gamPred <- predict(gamData, testImputedFinal, type = "response")
gamPred


par(mfrow = c(3,3))
plot(gamData$fitted(), resid(gamData))
plot(gamData, type = "l")
qqnorm(resid(gamData))
qqline(resid(gamData))
summary(gamData)
names(gamData)

gamRes <- gamPred - testImputedFinal$leg_pct
gamRMSE <- sqrt(mean(gamRes^2))
gamRMSE
RMSE(gamPred,testImputedFinal$leg_pct)
MAE(gamPred,testImputedFinal$leg_pct)
R2_Score(gamPred,testImputedFinal$leg_pct)

gamPredDf <-
  as_tibble(predict(gamData, testImputedFinal, type = "response")) %>%
  bind_cols(testImputedFinal) %>%
  mutate(error = V1 - leg_pct,
         error_sq = error ^ 2,
         error_abs = abs(error),
         sum_abs_error = sum(error_abs),
         mae = sum_abs_error / n(),
         sum_sq_error = sum(error_sq),
         rmse = sqrt(sum_sq_error / n()))


ggplot(predictOls, aes(x = infantmortality, y = value)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  theme_bw() +
  labs(title = "OLS predicted vs. truth")
##################
# random forest
set.seed(seed)
rfTask <- makeRegrTask(data = trainImputedFinal, target = "leg_pct")
rfLearner <- makeLearner("regr.randomForest")
rfParamSpace <-
  makeParamSet(makeIntegerParam("ntree", lower = 120, upper = 120),
               makeIntegerParam("mtry", lower = 3, upper = 11),
               makeIntegerParam("nodesize", lower = 3, upper = 15),
               makeIntegerParam("maxnodes", lower = 20, upper = 70))
# search 200 combos of hyperparameters
rfRandSearch <- makeTuneControlRandom(maxit = 100)
parallelStart(mode="multicore",cpus=parallel::detectCores())
rfTunedParams <- tuneParams(rfLearner, task = rfTask,
                            resampling = cv,
                            par.set = rfParamSpace,
                            control = rfRandSearch)
save(rfTunedParams, file = "mlhw3_rf.rdata")
parallelStop()
rfTunedParams

rfTuned <- setHyperPars(rfLearner, par.vals = rfTunedParams$x)
rfTrained <- mlr::train(rfTuned, rfTask)
rfModelData <- getLearnerModel(rfTrained)
plot(rfModelData)
rfPreds <- predict(rfTrained, newdata = testImputedFinal)
rfPredsData <- rfPreds$data

rfRes <- rfPredsData$response - rfPredsData$truth
rfRMSE <- sqrt(mean(rfRes^2))
rfRMSE
RMSE(rfPredsData$response,rfPredsData$truth)
MAE(rfPredsData$response,rfPredsData$truth)
R2_Score(rfPredsData$response,rfPredsData$truth)

rfPredDf <-
  as_tibble(predict(rfTrained, newdata = testImputedFinal, type = "response")) %>%
  bind_cols(testImputedFinal) %>%
  mutate(error = response - leg_pct,
         error_sq = error ^ 2,
         error_abs = abs(error),
         sum_abs_error = sum(error_abs),
         mae = sum_abs_error / n(),
         sum_sq_error = sum(error_sq),
         rmse = sqrt(sum_sq_error / n()))

#####
# metrics
library(xtable)
tests_df <- c("", "Linear", "GAM", "RF")
rmses_df <- c("RMSE",RMSE(olsPred,testImputedFinal$leg_pct),RMSE(gamPred,testImputedFinal$leg_pct),RMSE(rfPredsData$response,rfPredsData$truth))
maes_df <- c("MAE", MAE(olsPred,testImputedFinal$leg_pct), MAE(gamPred,testImputedFinal$leg_pct), MAE(rfPredsData$response,rfPredsData$truth))
r2s_df <- c("R^2", R2_Score(olsPred,testImputedFinal$leg_pct), R2_Score(gamPred,testImputedFinal$leg_pct), R2_Score(rfPredsData$response,rfPredsData$truth))
xtable(cbind(tests_df, rmses_df, maes_df, r2s_df))

library(gridExtra)
grid.arrange(
ggplot(olsPredDf, aes(x = leg_pct, y = value)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  theme_bw() +
  labs(title = "OLS predicted vs. truth", x = "pct female legislature", y = "response"),

ggplot(gamPredDf, aes(x = leg_pct, y = V1)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  theme_bw() +
  labs(title = "GAM predicted vs. truth", x = "pct female legislature", y = "response"),

ggplot(rfPredDf, aes(x = leg_pct, y = response)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  theme_bw() +
  labs(title = "RF predicted vs. truth", x = "pct female legislature", y = "response")
)
########
# full data
# impute full data
set.seed(seed)
fullImputed_1 <- impute(df_select, target = "leg_pct", 
                      classes = list(numeric = imputeMethod, integer = imputeMethod, factor = imputeMethodDummy))
fullImputed <- fullImputed_1$data
fullImputed <- fullImputed %>%
  select(-c("cowcode","iyear"))

fullStandard <- fullImputed %>%
  mutate(civ_atks_st = (civ_atks-mean(civ_atks))/sd(civ_atks)) %>%
  mutate(cllag_st = (cllag-mean(cllag))/sd(cllag)) %>%
  mutate(prlag_st = (prlag-mean(prlag))/sd(prlag)) %>%
  mutate(gdp_st = (gdp-mean(gdp))/sd(gdp)) %>%
  mutate(cinc_st = (cinc-mean(cinc))/sd(cinc)) %>%
  mutate(pop_st = (pop-mean(pop))/sd(pop)) %>%
  mutate(youth_st = (youth-mean(youth))/sd(youth)) %>%
  mutate(total_atks_st = (total_atks-mean(total_atks))/sd(total_atks))
fullStandard <- fullStandard %>%
  select(leg_pct, civ_atks_st, cllag_st, prlag_st, gdp_st, cinc_st, pop_st, war, youth_st, 
           muslim_maj, coldwar, total_atks_st)
library(relaimpo)
calc.relimp(olsFull)

olsFullStandard <- lm(leg_pct ~ civ_atks_st + cllag_st + prlag_st + gdp_st + cinc_st + pop_st + war + youth_st + 
                     muslim_maj + coldwar + total_atks_st, data = fullStandard)
olsFull <- lm(leg_pct ~ civ_atks + cllag + prlag + gdp + cinc + pop + war + youth + 
                muslim_maj + coldwar + total_atks, data = fullImputed)
olsNoCiv <- lm(leg_pct ~ cllag + prlag + gdp + cinc + pop + war + youth + 
                muslim_maj + coldwar + total_atks, data = fullImputed)
olsNoCl <- lm(leg_pct ~ civ_atks + prlag + gdp + cinc + pop + war + youth + 
                muslim_maj + coldwar + total_atks, data = fullImputed)
olsNoPr <- lm(leg_pct ~ civ_atks + cllag + gdp + cinc + pop + war + youth + 
                muslim_maj + coldwar + total_atks, data = fullImputed)
olsNoGdp <- lm(leg_pct ~ civ_atks + cllag + prlag + cinc + pop + war + youth + 
                muslim_maj + coldwar + total_atks, data = fullImputed)
olsNoCinc <- lm(leg_pct ~ civ_atks + cllag + prlag + gdp + pop + war + youth + 
                muslim_maj + coldwar + total_atks, data = fullImputed)
olsNoPop <- lm(leg_pct ~ civ_atks + cllag + prlag + gdp + cinc + war + youth + 
                muslim_maj + coldwar + total_atks, data = fullImputed)
olsNoWar <- lm(leg_pct ~ civ_atks + cllag + prlag + gdp + cinc + pop + youth + 
                muslim_maj + coldwar + total_atks, data = fullImputed)
olsNoYouth <- lm(leg_pct ~ civ_atks + cllag + prlag + gdp + cinc + pop + war + 
                muslim_maj + coldwar + total_atks, data = fullImputed)
olsNoMus <- lm(leg_pct ~ civ_atks + cllag + prlag + gdp + cinc + pop + war + youth + 
                coldwar + total_atks, data = fullImputed)
olsNoCw <- lm(leg_pct ~ civ_atks + cllag + prlag + gdp + cinc + pop + war + youth + 
                muslim_maj + total_atks, data = fullImputed)
olsNoTot <- lm(leg_pct ~ civ_atks + cllag + prlag + gdp + cinc + pop + war + youth + 
                muslim_maj + coldwar, data = fullImputed)
Rsqs <- c(summary(olsFull)$r.squared, summary(olsNoCiv)$r.squared, summary(olsNoCl)$r.squared, summary(olsNoPr)$r.squared,
summary(olsNoGdp)$r.squared, summary(olsNoCinc)$r.squared, summary(olsNoPop)$r.squared, summary(olsNoWar)$r.squared,
summary(olsNoYouth)$r.squared, summary(olsNoMus)$r.squared, summary(olsNoCw)$r.squared, summary(olsNoTot)$r.squared)
names.rsqs <- c("Full","Civilian Atks", "Civil Liberties", "Political Rts", "GDP", "CINC", "Pop", "War",
          "Youth", "Muslim Majority", "Cold War", "Total Atks")
rsqDf <- as.data.frame(cbind(names.rsqs, Rsqs))
rsqDf$Rsqs <- as.numeric(rsqDf$Rsqs)
rsqDf <- rsqDf %>%
  mutate_if(is.numeric, round, digits = 4)
xtable(rsqDf, digits = c(0,0,4))

library(caret)
str(varImp(olsFull))
summary(olsFull)
fullLinTask <- makeRegrTask(data = fullStandard, target = "leg_pct")
fullFilterVals <- generateFilterValuesData(fullLinTask, method = "randomForest_importance")
plotFilterValues(fullFilterVals) + theme_bw()

# gam
set.seed(seed)
gamTaskFull <- makeRegrTask(data = fullImputed, target = "leg_pct")
parallelStart(mode="multicore",cpus=parallel::detectCores())
gamTrainedFull <- mlr::train(gamFeatSelWrapper, gamTaskFull)
save(gamTrainedFull, file = "mlhw3_gam_full.rdata")
parallelStop()
gamVarimp <- varimp(getLearnerModel(gamTrainedFull, more.unwrap = TRUE), percent = TRUE)
plot(gamVarimp)
summary(getLearnerModel(gamTrainedFull, more.unwrap = TRUE))

par(mfrow=c(3,3))
gamFullData <- getLearnerModel(gamTrainedFull, more.unwrap = TRUE)
gamFullData
plot(gamFullData, type = "b")

par(mfrow=c(1,1))


# random forest
set.seed(seed)
rfTaskFull <- makeRegrTask(data = fullImputed, target = "leg_pct")
parallelStart(mode="multicore",cpus=parallel::detectCores())
rfTunedParamsFull <- tuneParams(rfLearner, task = rfTaskFull,
                            resampling = cv,
                            par.set = rfParamSpace,
                            control = rfRandSearch)
save(rfTunedParamsFull, file = "mlhw3_rf_full.rdata")
parallelStop()
rfTunedParamsFull

rfTunedFull <- setHyperPars(rfLearner, par.vals = rfTunedParamsFull$x)
rfTrainedFull <- mlr::train(rfTunedFull, rfTaskFull)
rfModelDataFull <- getLearnerModel(rfTrainedFull)
plot(rfModelDataFull)

rfFeatures <-
  getFeatureImportance(rfTrainedFull, type = 2)$res %>%
  mutate(model = "randomForestFeat")
ggplot(rfFeatures, aes(x = reorder(variable, importance), y = importance)) +
  geom_bar(stat = "identity") +
  facet_wrap(~ model, scales = "free_x") +
  coord_flip() +
  theme_bw()



library(stargazer)
stargazer(outendog3, olsFullStandard)


