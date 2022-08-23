# =============================================================================.
# Machine Learning in Economics
# =============================================================================.
# Building an early warning system for fiscal stress. Comparing the classical
# econometric approach of using logit regression with a decision tree model,
# i.e. a random forest
# =============================================================================.

# Initialization ----
# -----------------------------------------------------------------------------.
rm(list = ls()); gc()
start_time <- Sys.time()

# set seed (please uncomment depending on your R Version)
# if using R 3.6 or later:
set.seed(1999, sample.kind = "Rounding")
# if using R 3.5 or earlier:
# set.seed(1999)

# package management ----
# -----------------------------------------------------------------------------.
if(!require(glmnet)) install.packages("glmnet")
if(!require(randomForest)) install.packages("randomForest")
if(!require(ROCR)) install.packages("ROCR")
if(!require(iml)) install.packages("iml")
if(!require(pdp)) install.packages("pdp")
if(!require(data.table)) install.packages("data.table")

library(glmnet)
library(randomForest)
library(ROCR)
library(iml)
library(pdp)
library(data.table)

# objects ----
# -----------------------------------------------------------------------------.

means.table <- data.frame(
  variable = character(),
  all_periods = numeric(),
  tranq_periods = numeric(),
  stress_periods = numeric(),
  p_value = numeric(),
  significant = logical()
)

results.lasso <- data.frame(
  model = character(),
  year = integer(),
  weight = numeric(),
  prop.pos = numeric(),
  prop.neg = numeric(),
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

results.rf <- data.frame(
  model = character(),
  year = integer(),
  weight = numeric(),
  prop.pos = numeric(),
  prop.neg = numeric(),
  avg = numeric(),
  auc = numeric()
)

cutoffs.rf <- data.frame(
  model = character(),
  year = integer(),
  weight = numeric(),
  cutoff = numeric()
)

weights <- c(1, 1.5, 2)

list.export <- list()

shapley.values <- data.frame(
  feature = character(),
  phi = numeric()
)

# functions ----
# -----------------------------------------------------------------------------.

pred <- function(model, newdata){
  res <- as.data.frame(predict(model, newdata, type = "prob"))
  return(res[2])
}


# =============================================================================.
# 1. Data pre-processing ----
# =============================================================================.

# -----------------------------------------------------------------------------.
# 1.1 Download Data ----
# -----------------------------------------------------------------------------.
cat("\nLoad and prepare data...")

url <- "https://raw.githubusercontent.com/bt-koch/ML-in-Economics/main/data/data-ecb-wp-2408.csv"
data <- read.csv(url, check.names = F)
rm(url)

# -----------------------------------------------------------------------------.
# 1.2 Data Manipulation ----
# -----------------------------------------------------------------------------.

# factorize dummy variable(s)
data$crisis_next_period <- factor(x = data$crisis_next_period, levels = c(0,1), labels = c(0,1))
data$crisis_next_year <- factor(data$crisis_next_year, levels = c(0,1), labels = c(0,1))
data$crisis_first_year <- factor(data$crisis_first_year, levels = c(0,1), labels = c(0,1))

# rename column
names(data)[names(data) == ""] <- "country.id"

list.export[["data"]] <- data

# read csv for variable names
varnames <- read.csv("data/varnames.csv", sep = ";")

list.export[["varnames"]] <- varnames

# =============================================================================.
# 2. Exploratory Data Analysis ----
# =============================================================================.
cat("\nPerform exploratory data analysis...")

# -----------------------------------------------------------------------------.
# 2.1 Table: means of variables with significance ----
# -----------------------------------------------------------------------------.

drop <- c("country.id", "country", "year", "crisis_next_year", "crisis_next_period",
          "crisis_first_year", "developed")

variables <- names(data[, -which(names(data) %in% drop)])

for(var in variables){
  
  all_periods <- mean(data[[var]])
  tranq_periods <- mean(data[data$crisis_next_period == 0,][[var]])
  stress_periods <- mean(data[data$crisis_next_period == 1,][[var]])
  wilcox_test <- wilcox.test(x = data[data$crisis_next_period == 0,][[var]],
                             y = data[data$crisis_next_period == 1,][[var]])
  p_value <- wilcox_test$p.value
  significant <- ifelse(p_value < 0.05, TRUE, FALSE)
  
  temp <- data.frame(
    variable = var,
    all_periods = all_periods,
    tranq_periods = tranq_periods,
    stress_periods = stress_periods,
    p_value = p_value,
    significant = significant
  )
  
  means.table <- rbind(means.table, temp)
  
}

list.export[["means.table"]] <- means.table

# -----------------------------------------------------------------------------.
# 2.2 pairwise correlations ----
# -----------------------------------------------------------------------------.

variables.df <- data[, which(names(data) %in% variables)]

setnames(variables.df,
         old = varnames$variable,
         new = varnames$name)

corr.matrix <- cor(variables.df)

list.export[["corr.matrix"]] <- corr.matrix


# =============================================================================.
# 3. Train models ----
# =============================================================================.
cat("\nTrain models:")

for(i in 2007:max(data$year)){

  # ---------------------------------------------------------------------------.
  # 3.1 Prepare training ----
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
    test.set[, -which(names(test.set) %in% c(drop, "developed"))]
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
  # 3.2 Logit with LASSO Penalisation ----
  # ---------------------------------------------------------------------------.
  cat("\ntrain logit lasso for", i)

  for(dev.measure in c("GDP", "DUMMY")){

    # get data ----------------------------------------------------------------.
    x.train <- get(paste0("x.train.", dev.measure))
    x.test <- get(paste0("x.test.", dev.measure))

    # drop variables for logit because of high pairwise correlation
    drop <- c("dyn_prod_dol", "dyn_fix_cap_form", "overvaluation")

    x.train <- x.train[, -which(colnames(x.train) %in% drop)]
    x.test <- x.test[, -which(colnames(x.test) %in% drop)]
    
    # add interaction effects if binary variable is used for economic development
    if(dev.measure == "DUMMY"){
      f <- as.formula( ~ developed*.)
      x.train <- model.matrix(f, as.data.frame(x.train))[,-1]
      x.test <- model.matrix(f, as.data.frame(x.test))[,-1] 
    }

    # train model -------------------------------------------------------------.
    # fit model
    lasso.fit <- cv.glmnet(x.train, y.train, family = "binomial", nfolds = 5,
                           type.measure = "auc", standardize = TRUE)
    lasso.lambda.min <- lasso.fit$lambda.min

    # get predicted probability on train set
    lasso.response.train <- predict(lasso.fit, newx = x.train, s = "lambda.1se",
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
    lasso.response.test <- predict(lasso.fit, newx = x.test, s = "lambda.1se",
                                   type = "response", standardize = TRUE)

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
      lasso.yhat.temp <- ifelse(lasso.response.test > cutoff.temp, 1, 0)

      y.test.num <- as.numeric(as.character(y.test))

      # confusion matrix
      lasso.tp.temp <- sum(lasso.yhat.temp == 1 & y.test.num == 1)
      lasso.fp.temp <- sum(lasso.yhat.temp == 1 & y.test.num == 0)
      lasso.tn.temp <- sum(lasso.yhat.temp == 0 & y.test.num == 0)
      lasso.fn.temp <- sum(lasso.yhat.temp == 0 & y.test.num == 1)

      # calculate rates
      lasso.prop.positives <- lasso.tp.temp/sum(y.test.num == 1)
      lasso.prop.negatives <- lasso.tn.temp/sum(y.test.num == 0)
      lasso.avg.temp <- 0.5*(lasso.prop.positives+lasso.prop.negatives)

      # save results
      temp <- data.frame(
        model = paste0("logit.lasso.", dev.measure),
        year = i,
        weight = w,
        prop.pos = lasso.prop.positives,
        prop.neg = lasso.prop.negatives,
        avg = lasso.avg.temp,
        auc = lasso.auc
      )

      results.lasso <- rbind(results.lasso, temp)

    } # end of loop over weights
  } # end of loop over dev.measure
  
  # ---------------------------------------------------------------------------.
  # 3.3 Random Forest ----
  # ---------------------------------------------------------------------------.
  cat("\ntrain random forest for", i)

  for(dev.measure in c("GDP", "DUMMY")){

    # get data ----------------------------------------------------------------.
    x.train <- get(paste0("x.train.", dev.measure))
    x.test <- get(paste0("x.test.", dev.measure))

    # train model -------------------------------------------------------------.
    # fit model
    rf.fit <- randomForest(x = x.train, y = y.train, ntree = 10000)

    # get predicted probability on train set
    rf.response.train <- predict(rf.fit, newx = x.train, type = "prob")
    rf.response.train <- rf.response.train[,2]

    # get optimal threshold
    rf.pred <- prediction(rf.response.train, y.train)
    rf.sens <- performance(rf.pred, measure = "sens", x.measure = "cutoff")
    rf.spec <- performance(rf.pred, measure = "spec", x.measure = "cutoff")

    for(weight in weights){
      sens <- rf.sens@y.values[[1]]
      spec <- rf.spec@y.values[[1]]
      max.sum <- which.max(weight*sens+spec)
      rf.cutoff <- rf.sens@x.values[[1]][max.sum]

      temp <- data.frame(
        model = paste0("rf.", dev.measure),
        year = i,
        weight = weight,
        cutoff = rf.cutoff
      )

      cutoffs.rf <- rbind(cutoffs.rf, temp)

    }


    # test model --------------------------------------------------------------.
    rf.response.test <- predict(rf.fit, newdata = x.test, type = "prob")
    rf.response.test <- rf.response.test[,2]

    # calculate area under curve
    rf.asses <- prediction(rf.response.test, y.test)
    rf.auc <- performance(rf.asses, measure = "auc")@y.values[[1]]

    # calculate true positive rate and true negative rate with different weights
    # on sensitivity and specificity
    for(w in weights){
      cutoff.temp <- cutoffs.rf[
        cutoffs.rf$weight == w
        & cutoffs.rf$model == paste0("rf.", dev.measure)
        & cutoffs.rf$year == i,
      ]$cutoff

      # assign class conditional on best cutoff for corresponding weight
      rf.yhat.temp <- ifelse(rf.response.test > cutoff.temp, 1, 0)

      y.test.num <- as.numeric(as.character(y.test))

      # confusion matrix
      rf.tp.temp <- sum(rf.yhat.temp == 1 & y.test.num == 1)
      rf.fp.temp <- sum(rf.yhat.temp == 1 & y.test.num == 0)
      rf.tn.temp <- sum(rf.yhat.temp == 0 & y.test.num == 0)
      rf.fn.temp <- sum(rf.yhat.temp == 0 & y.test.num == 1)

      # calculate rates
      rf.prop.positives.temp <- rf.tp.temp/sum(y.test.num == 1)
      rf.prop.negatives.temp <- rf.tn.temp/sum(y.test.num == 0)
      rf.avg.temp <- 0.5*(rf.prop.positives.temp+rf.prop.negatives.temp)

      temp <- data.frame(
        model = paste0("rf.", dev.measure),
        year = i,
        weight = w,
        prop.pos = rf.prop.positives.temp,
        prop.neg = rf.prop.negatives.temp,
        avg = rf.avg.temp,
        auc = rf.auc
      )

      results.rf <- rbind(results.rf, temp)

    } # end of loop over weights
    
    # save some objects for evaluation ----------------------------------------.
    if(i == max(data$year) & dev.measure == "GDP"){
      
      x.train.eval <- x.train
      list.export[["x.train.eval"]] <- x.train.eval
      rf.fit.eval <- rf.fit
      list.export[["rf.fit.eval"]] <- rf.fit.eval
      rf.response.train.eval <- rf.response.train
      list.export[["rf.response.train.eval"]] <- rf.response.train.eval
      
      # x.train.df <- as.data.frame(x.train)
      # 
      # predictor.rf.eval <- Predictor$new(
      #   model = rf.fit.eval,
      #   data = x.train.df,
      #   y = rf.response.train,
      #   predict.function = pred,
      #   class = "classification"
      # )
      # 
      # # calculate shapley values for each observation
      # for(obs in 1:nrow(x.train.df)){
      # 
      #   cat("\rcalculate shapley values for observation", obs, "of",
      #       nrow(x.train.df), "observations")
      #   flush.console()
      # 
      #   x.interest <- x.train.df[obs,]
      #   temp <- Shapley$new(predictor.rf.eval, x.interest = x.interest)
      # 
      #   temp <- data.frame(
      #     feature = temp$results$feature,
      #     phi = temp$results$phi
      #   )
      # 
      #   shapley.values <- rbind(shapley.values, temp)
      # 
      # }
      # 
      # # calculate shapley values as averages of absolute values of shapley values
      # # of preductors in each observation
      # shapley.values.raw <- shapley.values
      # shapley.values$phi <- abs(shapley.values$phi)
      # shapley.values <- aggregate(shapley.values$phi,
      #                             by = list(shapley.values$feature),
      #                             FUN = mean)
      # 
      # list.export[["shapley.values"]] <- shapley.values
      # 
      # 
      # # calculate partail dependence
      # partial.net_lending <- partial(rf.fit.eval, train = x.train.eval,
      #                                pred.var = "net_lending", plot = F,
      #                                type = "classification", which.class = 2)
      # partial.ca_balance <- partial(rf.fit.eval, train = x.train.eval,
      #                               pred.var = "ca_balance", plot = F,
      #                               type = "classification", which.class = 2)
      # 
      # list.export[["partial.net_lending"]] <- partial.net_lending
      # list.export[["partial.ca_balance"]]  <- partial.ca_balance
      # 
      # # calculate accumulated local effects
      # ale.net_lending <- FeatureEffect$new(predictor.rf.eval,
      #                                      feature = "net_lending")
      # ale.ca_balance  <- FeatureEffect$new(predictor.rf.eval,
      #                                      feature = "ca_balance")
      # 
      # list.export[["ale.net_lending"]] <- ale.net_lending
      # list.export[["ale.ca_balance"]]  <- ale.ca_balance
      
    }
    
  } # end of loop over dev.measure
  
} # end of loop over years


# =============================================================================.
# 4. get relevant results and save rmd input ----
# =============================================================================.
cat("\nFinalize data and save some objects for RMD report...")

# -----------------------------------------------------------------------------.
# 4.1 calculate average prediction metrics ----
# -----------------------------------------------------------------------------.
results.avg <- rbind(results.lasso, results.rf)
results.avg <- results.avg[results.avg$weight == 1.5,]

# aggregate over years by model
results.avg <- aggregate(results.avg[c("prop.pos", "prop.neg", "auc")],
                         by = list(results.avg$model), FUN = mean)
names(results.avg)[names(results.avg) == "Group.1"] <- "model"

# calculate mean
results.avg$avg <- rowMeans(results.avg[c("prop.pos", "prop.neg")])

list.export[["results.avg"]] <- results.avg


# -----------------------------------------------------------------------------.
# 4.2 calculate interpretation metrics ----
# -----------------------------------------------------------------------------.

x.train.eval.df <- as.data.frame(x.train.eval)
list.export[["x.train.eval.df"]] <- x.train.eval.df

# generate predictor object
predictor.rf.eval <- Predictor$new(
  model = rf.fit.eval,
  data = x.train.eval.df,
  y = rf.response.train.eval,
  predict.function = pred,
  class = "classification"
)

# 4.2.1 shapley values --------------------------------------------------------.
# calculate shapley values for each observation
for(obs in 1:nrow(x.train.eval.df)){

  cat("\rcalculate shapley values for observation", obs, "of",
      nrow(x.train.eval.df), "observations")
  flush.console()
  
  x.interest <- x.train.eval.df[obs,]
  temp <- Shapley$new(predictor.rf.eval, x.interest = x.interest)
  
  temp <- data.frame(
    feature = temp$results$feature,
    phi = temp$results$phi
  )
  
  shapley.values <- rbind(shapley.values, temp)
  
}

# calculate shapley values as averages of absolute values of shapley values
# of preductors in each observation
shapley.values.raw <- shapley.values
shapley.values$phi <- abs(shapley.values$phi)
shapley.values <- aggregate(shapley.values$phi,
                            by = list(shapley.values$feature),
                            FUN = mean)

list.export[["shapley.values"]] <- shapley.values

# 4.2.2 partial dependence ----------------------------------------------------.
cat("\ncalculate partial dependence")
# calculate partial dependence
partial.net_lending <- partial(rf.fit.eval, train = x.train.eval,
                               pred.var = "net_lending", plot = F,
                               type = "classification", which.class = 2)
partial.ca_balance <- partial(rf.fit.eval, train = x.train.eval,
                              pred.var = "ca_balance", plot = F,
                              type = "classification", which.class = 2)
partial.diff_unempl <- partial(rf.fit.eval, train = x.train.eval,
                               pred.var = "diff_unempl", plot = F,
                               type = "classification", which.class = 2)

# export 
list.export[["partial.net_lending"]] <- partial.net_lending
list.export[["partial.ca_balance"]]  <- partial.ca_balance
list.export[["partial.diff_unempl"]]  <- partial.diff_unempl

# 4.2.3 accumulated local effects ---------------------------------------------.
# calculate accumulated local effects
ale.net_lending <- FeatureEffect$new(predictor.rf.eval,
                                     feature = "net_lending")
ale.ca_balance  <- FeatureEffect$new(predictor.rf.eval,
                                     feature = "ca_balance")
ale.ca_balance  <- FeatureEffect$new(predictor.rf.eval,
                                     feature = "diff_unempl")

list.export[["ale.net_lending"]] <- ale.net_lending
list.export[["ale.ca_balance"]]  <- ale.ca_balance
list.export[["ale.diff_unempl"]]  <- ale.diff_unempl

# -----------------------------------------------------------------------------.
# 4.3 save relevant data for rmd report ----
# -----------------------------------------------------------------------------.
save(list.export, file = "data/input.RData")

cat("\nRuntime:")
Sys.time() - start_time

