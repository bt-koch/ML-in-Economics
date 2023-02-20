# Machine Learning in Economics

The goal of this project is to build an early warning system for fiscal stress episodes using economic data. At the same time, the traditional econometric approach of using logistic regression is compared to a more modern approach of using a random forest algorithm.

## Data

The data used for this project covers annual frequency data for 43 countries, defined as 24 advanced economies and 19 emerging economies by the International Monetary Fund, for the years 1992-2018. It includes explanatory variables that can be classified to macroeconomic and global economy variables, financial variables, fiscal variables, variables about competitiveness and domestic demand as well as labor market variables. The dependent variable for a fiscal stress episode is a binary variable, which is equal to 1 in the case of a fiscal stress event and 0 otherwise. The fiscal stress variable is lagged (crisis_next_year lagged by 1 year, crisis_next_period lagged by 2 years, crisis_first_year lagged by 1 year and only first year of a fiscal stress episode coded as 1).

The data is provided by the author of the original paper and was downloaded from [figshare.com](https://figshare.com/articles/dataset/dataframe_csv/11593899).

## Models

This project compares the traditional econometric approach of logit regression (with least absolute shrinkage and selection operator, LASSO) to an implementation of random forest to build an early warning system signalling risk of fiscal stress.

## Results

The model based on the random forest algorithm achieved an average of sensitivity and specificity of 77-79%, therefore outperforming the model based logit LASSO with an average of sensitivity and specificity of 71-73%. However, according to the paper of [Barbara Jarmulska (2020)](https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2408~aa6b05aed7.en.pdf), the prediction accuracy drops by around 10\% when trying to predict the first occurrence of a fiscal stress episode. However, the proposed models can still be helpful, since the objective of an early warning model is not to forecast a fiscal crisis, but rather to warn from a heightened level of vulnerability. Therefore, it is still worth to further develop these models.

## About

This project attempts to replicate the paper [Random forest versus logit models: which offers better early warning of fiscal stress?](https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2408~aa6b05aed7.en.pdf) by Barbara Jarmulska (ECB Working Paper Series No 2408 / May 2020) and was created as a part of the Lecture "Machine Learning in Economics" at the University of Bern.
