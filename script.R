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
# ROCR? no need rn

library(glmnet)
library(randomForest)
library(ROCR)

# objects ---------------------------------------------------------------------.

# results <- data.frame(
#   model = character(),
#   year = integer(),
#   accur_stress = numeric(),
#   accur_tranq = numeric(),
#   avg = numeric(),
#   auroc = numeric()
# )

results.lasso <- data.frame(
  model = character(),
  year = integer(),
  tpr = numeric(),
  tnr = numeric(),
  avg = numeric(),
  auc = numeric()
)

# functions -------------------------------------------------------------------.
# opt.cut <- function(perf, pred){
#   cut.ind <- mapply(FUN=function(x, y, p){
#     d <- (x-0)^2 + (y-1)^2
#     ind <- which(d == min(d))
#     c(sensitivity = y[[ind]],
#       specificity = 1-x[[ind]],
#       cutoff = p[[ind]])
#   }, perf@x.values, perf@y.values, pred@cutoffs)
# }


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

# # subset explanatory variables
# drop <- c("country.id", "country", "year", "crisis_next_year", "crisis_next_period",
#           "crisis_first_year", "developed")
# data.st <- data[, -which(names(data) %in% drop)]
# 
# # standardize explanatory variables (mean = 0, sd = 1)
# data.st <- scale(data.st)
# 
# # checks
# stopifnot(round(apply(data.st, 2, mean), 6) == 0)
# stopifnot(round(apply(data.st, 2, sd), 6) == 1)
# 
# # add other variables again
# add <- data[, which(names(data) %in% drop)]
# data.st <- cbind(data.st, add)
# 
# # clean up
# rm(drop, add)

# =============================================================================.
# 2. Train models ----
# =============================================================================.

# train.period <- min(data$year):2009

test <- c()
test2 <- c()

test.cutoff <- c()

for(i in 2007:max(data$year)){
# for(add.year in 2006:2006){
  
  # ---------------------------------------------------------------------------.
  # 2.1 Prepare training ----
  # ---------------------------------------------------------------------------.
  
  train.period <- min(data$year):(i-1)
  test.period <- i
  
  # print(train.period)
  # print(i)
  
  # train and test set
  train.set <- data[data$year %in% train.period,]
  test.set <- data[data$year == test.period,]
  
  # train.set.st <- data.st[data.st$year %in% train.period,]
  # test.set.st <- data.st[data.st$year == test.period,]
  
  # explanatory variables
  drop <- c("country.id", "country", "year", "crisis_next_year", "crisis_next_period",
            "crisis_first_year")
  
  x.train.GDP <- as.matrix(
    train.set[, -which(names(train.set) %in% c(drop, "developed"))]
  )
  x.test.GDP <- as.matrix(
    test.set[, -which(names(train.set) %in% c(drop, "developed"))]
  )
  
  # x.train.GDP.st <- as.matrix(
  #   train.set.st[, -which(names(train.set.st) %in% c(drop, "developed"))]
  # )
  # x.test.GDP.st <- as.matrix(
  #   test.set.st[, -which(names(train.set.st) %in% c(drop, "developed"))]
  # )
  
  # x.train.DUMMY <- as.matrix(
  #   train.set[, -which(names(train.set) %in% c(drop, "GDP_per_cap"))]
  # )
  # x.train.DUMMY.st <- as.matrix(
  #   train.set.st[, -which(names(train.set.st) %in% c(drop, "GDP_per_cap"))]
  # )
  
  # response
  # y.train <- as.factor(train.set$crisis_next_period)
  # y.test <- as.factor(test.set$crisis_next_period)
  # --> factorize before so we are sure to have same factor in both sets!!
  y.train <- train.set$crisis_next_period
  y.test <- test.set$crisis_next_period

  # print(train.period)
  
  # ---------------------------------------------------------------------------.
  # 2.2 Logit with LASSO Penalisation ----
  # ---------------------------------------------------------------------------.
  
  # ---------------------------------------------------------------------------.
  # 2.2.1 Logit LASSO with GDP as measure for developement of country ----
  # ---------------------------------------------------------------------------.
  
  # train model ---------------------------------------------------------------.
  # fit model
  lasso.fit.GDP <- cv.glmnet(x.train.GDP, y.train, family = "binomial",
                             type.measure = "auc", nfolds = 5, standardize = TRUE)
  
  lasso.response.train.GDP <- predict(lasso.fit.GDP, newx = x.train.GDP, s = "lambda.min",
                                      standardize = TRUE, type = "response")
  
  # get optimal cutoff
  lasso.pred.GDP <- prediction(lasso.response.train.GDP, y.train)
  lasso.sens.GDP <- performance(lasso.pred.GDP, measure="sens", x.measure = "cutoff")
  lasso.spec.GDP <- performance(lasso.pred.GDP, measure="spec", x.measure = "cutoff")
  
  best.sum <- which.max(1*lasso.sens.GDP@y.values[[1]]+lasso.spec.GDP@y.values[[1]])
  lasso.cutoff.GDP <- lasso.sens.GDP@x.values[[1]][best.sum]

  # both.eq <- which.min(abs(lasso.sens.GDP@y.values[[1]]-lasso.spec.GDP@y.values[[1]]))
  # closest <- lasso.sens.GDP@x.values[[1]][both.eq]
  # 
  # plot(lasso.sens.GDP, type = "l", col = "red",xlab = "", ylab = "")
  # par(new=T)
  # plot(lasso.spec.GDP, type = "l", col = "blue", xlab = "", ylab = "")
  # abline(v = max.sum, col = "black", lty = 3)
  # abline(v = closest, col = "black", lty = 3)
  # abline(v = .1, col = "green")

  # test model ----------------------------------------------------------------.
  
  lasso.response.test.GDP <- predict(lasso.fit.GDP, newx = x.test.GDP, s = "lambda.min",
                                     standardize = TRUE, type = "response")
  lasso.yhat.GDP <- ifelse(lasso.response.test.GDP > lasso.cutoff.GDP, 1, 0)
  
  # confusion matrix
  lasso.tp.GDP <- sum(lasso.yhat.GDP == 1 & y.test == 1)
  lasso.fp.GDP <- sum(lasso.yhat.GDP == 1 & y.test == 0)
  lasso.tn.GDP <- sum(lasso.yhat.GDP == 0 & y.test == 0)
  lasso.fn.GDP <- sum(lasso.yhat.GDP == 0 & y.test == 1)
  
  lasso.tpr.GDP <- lasso.tp.GDP/(lasso.tp.GDP+lasso.fn.GDP)
  lasso.tnr.GDP <- lasso.tn.GDP/(lasso.tn.GDP+lasso.fp.GDP)
  lasso.avg.GDP <- 0.5*(lasso.tpr.GDP+lasso.tnr.GDP)
  
  message(i+2)
  print(paste("tpr:", round(lasso.tpr.GDP, 2)))
  print(paste("tnr:", round(lasso.tnr.GDP, 2)))
  print(paste("avg:", round(lasso.avg.GDP, 2)))
  
  # calculate area under curve
  lasso.asses.GDP <- assess.glmnet(lasso.fit.GDP, newx = x.test.GDP, newy = y.test,
                                   family = "binomial", standardize = TRUE)
  lasso.auc.GDP <- lasso.asses.GDP$auc
  print(paste("auc:", round(lasso.auc.GDP, 2)))
  
  # save results --------------------------------------------------------------.
  
  temp <- data.frame(
    model = "lasso.GDP",
    year = i,
    tpr = lasso.tpr.GDP,
    tnr = lasso.tnr.GDP,
    avg = lasso.avg.GDP,
    auc = lasso.auc.GDP
  )
  
  results.lasso <- rbind(results.lasso, temp)
  

  
  
  
  # ---------------------------------------------------------------------------.
  # 2.3 Random Forest ----
  # ---------------------------------------------------------------------------.
  
  # rffit.GDP <- randomForest(x.GDP, y, ntree = 10)
  
}
