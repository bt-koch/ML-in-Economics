# =============================================================================.
# Machine Learning in Economics
# =============================================================================.
# Building an early warning system for fiscal stress. Comparing the classical
# econometric approach of using logit regression with a decision tree model,
# i.e. a random forest
# =============================================================================.

# Initialization --------------------------------------------------------------.
rm(list = ls()); gc()
start_time <- Sys.time()

# # set seed (please uncomment depending on your R Version)
# # if using R 3.6 or later:
# set.seed(1000, sample.kind = "Rounding")
# # if using R 3.5 or earlier:
# # set.seed(1)

# package management ----------------------------------------------------------.
if(!require(glmnet)) install.packages("glmnet")
if(!require(randomForest)) install.packages("randomForest")
if(!require(ROCR)) install.packages("ROCR")

library(glmnet)
library(randomForest)
library(ROCR)

# objects ---------------------------------------------------------------------.

results.lasso <- data.frame(
  model = character(),
  year = integer(),
  weight = numeric(),
  tpr = numeric(),
  tnr = numeric(),
  avg = numeric(),
  auc = numeric()
)

cutoffs.lasso <- data.frame(
  model = character(),
  year = integer(),
  weight = numeric(),
  cutoff = numeric()
)

yhat.lasso <- data.frame()

weights <- c(1, 1.5, 2)

# functions -------------------------------------------------------------------.


# =============================================================================.
# 1. Data pre-processing ----
# =============================================================================.

# -----------------------------------------------------------------------------.
# 1.1 Download Data ----
# -----------------------------------------------------------------------------.

url <- "https://raw.githubusercontent.com/bt-koch/ML-in-Economics/main/data/data-ecb-wp-2408.csv"
data <- read.csv(url, check.names = F)
rm(url)


# -----------------------------------------------------------------------------.
# 1.2 Data Manipulation ----
# -----------------------------------------------------------------------------.

# factorize dependent variable(s)
data$crisis_next_period <- as.factor(data$crisis_next_period)
data$crisis_next_year <- as.factor(data$crisis_next_year)
data$crisis_first_year <- as.factor(data$crisis_first_year)

# rename column
names(data)[names(data) == ""] <- "country.id"


# =============================================================================.
# 2. Train models ----
# =============================================================================.

for(i in 2007:max(data$year)){
# for(i in 2007:2007){

  # ---------------------------------------------------------------------------.
  # 2.1 Prepare training ----
  # ---------------------------------------------------------------------------.
  
  # train and test period
  train.period <- min(data$year):(i-1)
  test.period <- i
  
  # train and test set
  train.set <- data[data$year %in% train.period,]
  test.set <- data[data$year == test.period,]
  
  # create set of explanatory variables
  drop <- c("country.id", "country", "year", "crisis_next_year", "crisis_next_period",
            "crisis_first_year")
  
  # GDP as measure of development of country
  x.train.GDP <- as.matrix(
    train.set[, -which(names(train.set) %in% c(drop, "developed"))]
  )
  x.test.GDP <- as.matrix(
    test.set[, -which(names(train.set) %in% c(drop, "developed"))]
  )
  
  # dummy as measure of development of country
  x.train.DUMMY <- as.matrix(
    train.set[, -which(names(train.set) %in% c(drop, "GDP_per_cap"))]
  )
  x.test.DUMMY <- as.matrix(
    test.set[, -which(names(test.set) %in% c(drop, "GDP_per_cap"))]
  )
  
  # response
  y.train <- train.set$crisis_next_period
  y.test <- test.set$crisis_next_period

  # clean up
  rm(drop)
  
  # ---------------------------------------------------------------------------.
  # 2.2 Logit with LASSO Penalisation ----
  # ---------------------------------------------------------------------------.
  
  for(dev.measure in c("GDP", "DUMMY")){
    
    # get data ----------------------------------------------------------------.
    x.train <- get(paste0("x.train.", dev.measure))
    x.test <- get(paste0("x.test.", dev.measure))
    
    # train model -------------------------------------------------------------.
    # fit model
    lasso.fit <- cv.glmnet(x.train, y.train, family = "binomial", nfolds = 5,
                           type.measure = "auc", standardize = TRUE)
    
    # get predicted probability on train set
    lasso.response.train <- predict(lasso.fit, newx = x.train, s = "lambda.min",
                                    type = "response", standardize = TRUE)
    
    # get optimal threshold
    lasso.pred <- prediction(lasso.response.train, y.train)
    lasso.sens <- performance(lasso.pred, measure = "sens", x.measure = "cutoff")
    lasso.spec <- performance(lasso.pred, measure = "spec", x.measure = "cutoff")
    
    for(weight in weights){
      sens <- lasso.sens@y.values[[1]]
      spec <- lasso.spec@y.values[[1]]
      max.sum <- which.max(weight*sens+spec)
      lasso.cutoff <- lasso.sens@x.values[[1]][max.sum]
      
      temp <- data.frame(
        model = paste0("logit.lasso.", dev.measure),
        year = i,
        weight = weight,
        cutoff = lasso.cutoff
      )
      
      cutoffs.lasso <- rbind(cutoffs.lasso, temp)
      
    }
    
    # test model --------------------------------------------------------------.
    lasso.response.test <- predict(lasso.fit, newx = x.test, s = "lambda.min",
                                   type = "response", standardize = TRUE)
    
    colnames(lasso.response.test) <- "response"
    lasso.response.test <- as.data.frame(lasso.response.test)
    
    # calculate area under curve
    lasso.asses <- assess.glmnet(lasso.fit, newx = x.test, newy = y.test,
                                 family = "binomial", standardize = TRUE)
    lasso.auc <- lasso.asses$auc
    
    # calculate true positive rate and true negative rate with different weights
    # on sensitivity and specificity
    for(w in weights){
      cutoff.temp <- cutoffs.lasso[
        cutoffs.lasso$weight == w
        & cutoffs.lasso$model == paste0("logit.lasso.", dev.measure)
        & cutoffs.lasso$year == i,
        ]$cutoff
    
      # assign class conditional on best cutoff for corresponding weight
      lasso.yhat.temp <- ifelse(lasso.response.test$response > cutoff.temp, 1, 0)
      
      # confusion matrix
      lasso.tp.temp <- sum(lasso.yhat.temp == 1 & y.test == 1)
      lasso.fp.temp <- sum(lasso.yhat.temp == 1 & y.test == 0)
      lasso.tn.temp <- sum(lasso.yhat.temp == 0 & y.test == 0)
      lasso.fn.temp <- sum(lasso.yhat.temp == 0 & y.test == 1)
      
      # calculate rates
      lasso.tpr.temp <- lasso.tp.temp/(lasso.tp.temp+lasso.fn.temp)
      lasso.tnr.temp <- lasso.tn.temp/(lasso.tn.temp+lasso.fp.temp)
      lasso.avg.temp <- 0.5*(lasso.tpr.temp+lasso.tnr.temp)
      
      # save results 
      temp <- data.frame(
        model = paste0("logit.lasso.", dev.measure),
        year = i,
        weight = w,
        tpr = lasso.tpr.temp,
        tnr = lasso.tnr.temp,
        avg = lasso.avg.temp,
        auc = lasso.auc
      )
      
      results.lasso <- rbind(results.lasso, temp)
      
    } # end of loop over weights
  } # end of loop over dev.measure
  
  # ---------------------------------------------------------------------------.
  # 2.3 Random Forest ----
  # ---------------------------------------------------------------------------.
  
  # rffit.GDP <- randomForest(x.GDP, y, ntree = 10)
  
} # end of loop over years

message("weight = 1:")
# print(round(mean(results.lasso[results.lasso$weight==1,]$auc),2))
print(round(mean(results.lasso[results.lasso$weight==1,]$tpr),2))
print(round(mean(results.lasso[results.lasso$weight==1,]$tnr),2))
print(round(mean(results.lasso[results.lasso$weight==1,]$avg),2))

message("weight = 1.5:")
# print(round(mean(results.lasso[results.lasso$weight==1.5,]$auc),2))
print(round(mean(results.lasso[results.lasso$weight==1.5,]$tpr),2))
print(round(mean(results.lasso[results.lasso$weight==1.5,]$tnr),2))
print(round(mean(results.lasso[results.lasso$weight==1.5,]$avg),2))

message("weight = 2:")
# print(round(mean(results.lasso[results.lasso$weight==2,]$auc),2))
print(round(mean(results.lasso[results.lasso$weight==2,]$tpr),2))
print(round(mean(results.lasso[results.lasso$weight==2,]$tnr),2))
print(round(mean(results.lasso[results.lasso$weight==2,]$avg),2))