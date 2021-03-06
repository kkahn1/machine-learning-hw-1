
library(caret)
library(DMwR)
library(PRROC)
library(tidyverse)
library(stargazer)
# reproducing
hwdf<-read.csv("https://raw.githubusercontent.com/kkahn1/machine-learning-hw-1/main/edtg_hw1_table4.csv")
set.seed(9986000)
keep<-with(hwdf,c("dstar","x1","x2","left","nat","right","shr_trans","diversity","num_base",
                  "lpop","polity1","polity1sqr","ethnic","ethn2","EAP","ECA","LAC","NA.","SAS","SSA",
                  "lelev","tropics","landlock","total_atks"))
hwdf<-hwdf%>%
  select(all_of(keep))
hwdf$dstar<-as.factor(hwdf$dstar)
hwdf<-na.omit(hwdf)
# so i can do two class summary
levels(hwdf$dstar)<-c("active","end")

# this is to make sure i can reproduce it
reproduce<-glm(dstar~x1+x2+left+nat+right+shr_trans+diversity+num_base+
                  lpop+polity1+polity1sqr+ethnic+ethn2+EAP+ECA+LAC+NA.+SAS+SSA+
                  lelev+tropics+landlock, family=binomial(link="logit"),data=hwdf)
summary(reproduce)
# split data
partition<-createDataPartition(hwdf$dstar,p=.7,list=FALSE)
train<-hwdf[partition,]
test<-hwdf[-partition,]
# to make sure caret version gives very similar results as glm
reproducetrain<-glm(dstar~x1+x2+left+nat+right+shr_trans+diversity+num_base+
                      lpop+polity1+polity1sqr+ethnic+ethn2+EAP+ECA+LAC+NA.+SAS+SSA+
                      lelev+tropics+landlock, family=binomial(link="logit"),data=train)
summary(reproducetrain)
# it does
reproduce_train<-caret::train(dstar~x1+x2+left+nat+right+shr_trans+diversity+num_base+
                         lpop+polity1+polity1sqr+ethnic+ethn2+EAP+ECA+LAC+NA.+SAS+SSA+
                         lelev+tropics+landlock,data=train, method="glm", family="binomial"(link="logit"))
summary(reproduce_train)
# for cross validation
# get combined metrics
# auc: discriminate between positive and negative
# sensitivity: true positive. predicting positive when outcome is positive
# specificity: true negative.
# precision: true positives out of all positives
# recall: sensitivity
# but treats active as positive and releveling inverts results and makes test incorrect later
# so think true negatives/(true neg+false neg) for precision
# look at specificity for recall
MySummary  <- function(data, lev = NULL, model = NULL){
  a1 <- defaultSummary(data, lev, model)
  b1 <- twoClassSummary(data, lev, model)
  c1 <- prSummary(data, lev, model)
  out <- c(a1, b1, c1)
  out}
# first cross validate the original model
# using synthetic resampling or else it will have problems bc it is so rare
ctrl<-trainControl(method ="repeatedcv",repeats=5,number=5,savePredictions=TRUE,classProbs=TRUE,summaryFunction=MySummary,sampling="smote")
set.seed(9986000)
reproduce_train<-caret::train(dstar~x1+x2+left+nat+right+shr_trans+diversity+num_base+
                                lpop+polity1+polity1sqr+ethnic+ethn2+EAP+ECA+LAC+NA.+SAS+SSA+
                                lelev+tropics+landlock,data=train, method="glm", family="binomial"(link="logit"),
                                trControl=ctrl)
reproduce_train$results 
confusionMatrix(reproduce_train,positive="end")
#precision 
1.8/(1.8+18.8)

# model without regional vars
set.seed(9986000)
noreg<-caret::train(dstar~x1+x2+left+nat+right+shr_trans+diversity+num_base+
                                lpop+polity1+polity1sqr+ethnic+ethn2+
                                lelev+tropics+landlock,data=train, method="glm", family="binomial"(link="logit"),
                              na.action=na.exclude,trControl=ctrl)
noreg$results
confusionMatrix(noreg)
1.8/(1.8+17.9) # precision a little better, specificity worse
# include regionals but no polity sq or ethn sq
set.seed(9986000)
nope2<-caret::train(dstar~x1+x2+left+nat+right+shr_trans+diversity+num_base+
                                lpop+polity1+ethnic+EAP+ECA+LAC+NA.+SAS+SSA+
                                lelev+tropics+landlock,data=train, method="glm", family="binomial"(link="logit"),
                              na.action=na.exclude,trControl=ctrl)
nope2$results
confusionMatrix(nope2)
# remove only polity2
set.seed(9986000)
nop2<-caret::train(dstar~x1+x2+left+nat+right+shr_trans+diversity+num_base+
                      lpop+polity1+ethnic+ethn2+EAP+ECA+LAC+NA.+SAS+SSA+
                      lelev+tropics+landlock,data=train, method="glm", family="binomial"(link="logit"),
                    na.action=na.exclude,trControl=ctrl)
nop2$results
confusionMatrix(nop2)
# no orientation, polity sq, or regional
set.seed(9986000)
noor<-caret::train(dstar~x1+x2+shr_trans+diversity+num_base+
                     lpop+polity1+ethnic+ethn2+
                     lelev+tropics+landlock,data=train, method="glm", family="binomial"(link="logit"),
                   na.action=na.exclude,trControl=ctrl)
noor$results # specificity is a lot worse
confusionMatrix(noor)
1.5/(1.5+15.5)
set.seed(9986000)
attacks<-caret::train(dstar~x1+x2+shr_trans+diversity+num_base+log(total_atks+1)+
                                         lpop+polity1+polity1sqr+ethnic+ethn2+
                                         lelev+tropics+landlock,data=train, method="glm", family="binomial"(link="logit"),
                                       na.action=na.exclude,trControl=ctrl)
attacks$results
confusionMatrix(attacks)
1.6/(16.5+1.6)

# final model is without regional variables (noreg)
predorig<-predict(reproduce_train,test,"raw")
confusionMatrix(data=predorig,reference=test$dstar,positive="end")
prednoreg<-predict(noreg,test,"raw")
confusionMatrix(data=prednoreg,reference=test$dstar,positive="end")

#new specification
newglm<-glm(dstar~x1+x2+left+nat+right+shr_trans+diversity+num_base+
                 lpop+polity1+polity1sqr+ethnic+ethn2+
                 lelev+tropics+landlock, family=binomial(link="logit"),data=hwdf)
summary(newglm)
stargazer(reproduce,newglm,font.size="small")

proborig<-predict(reproduce_train,test,"prob")
probnoreg<-predict(noreg,test,"prob")
plot(pr.curve(scores.class0=proborig$end,scores.class1=proborig$active,curve=TRUE))
plot(pr.curve(scores.class0=probnoreg$end,scores.class1=probnoreg$active,curve=TRUE))


