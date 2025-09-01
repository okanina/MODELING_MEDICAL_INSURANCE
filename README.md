# Medical Insurance Payouts.


<img src="templates/static/images/insurance_claim_form.jpg" width=1000>


## Project Overview

 The challenge that face insurance industries is to charge each customer an appropriate premium for the risk they present. The ability to predict the correct claim amount has a significant impact on insurer's management decisions and financial statements. Therefore, the prediction errors can adversely affect the insurer's pricing, potentially hurting its profitability. The errors can also lead to insufficient reserves and jeopardize the  insurer's solvency.

 The prediction risk is a serious concern to both the insurer and regulator.
 
 The aim of this project is to develop a very robust model that will help identify high risk customers in the follwing year by predicting future healthcare expenditures from past data with minimal errors to no errors.

 ## Data Description

* Data Source - [https://www.kaggle.com/datasets/sureshgupta/health-insurance-data-set?resource=download](https://www.kaggle.com/datasets/sureshgupta/health-insurance-data-set?resource=download).

* It has 15000 cases with 13 features. 7 categorical and 6 numeric values.

<mark>Dataset features<mark>:

age : Age of the policyholder (Numeric)

sex: Gender of policyholder (Categoric)

weight: Weight of the policyholder (Numeric)

bmi: Measure of body fat based on height and weight (Numeric)

smoker: Indicates policyholder is a smoker or a non-smoker (non-smoker=0;smoker=1) (Categoric)

claim: The amount claimed by the policyholder (Numeric)

bloodpressure: Bloodpressure reading of policyholder (Numeric)

diabetes: Indicates policyholder suffers from diabetes or not (non-diabetic=0; diabetic=1) (Categoric)

regular_ex: A policyholder regularly excercises or not (no-excercise=0; excercise=1) (Categoric)

job_title: Job profile of the policyholder (Categoric)

city: The city in which the policyholder resides (Categoric)

hereditary_diseases: A policyholder suffering from a hereditary diseases or not (Categoric)

* The dataset has a total of 1352 missing values for feature Age and bmi. Which is roughly 2% for bmi feature and 6% for age feature.  

* The dataset also has 1096 duplicates.

### Data Limitation

It is not clear as to where/when/how this data was collected. As a result there are limitations this brings on the analysis.

There seems to be an issue of data intergrity on this dataset. The amount of claim for diabetes customers in a variable diabetes does not match the amount claimed by diabetes customers on hereditary_diseases variable.

## Methods and Algorithms.

### Code and Resources used

* Editor Used: Visual Studio Code and jupyternotebook for [exploratory data analysis](research/exploratory_data_analysis.ipynb).
* Python Version: 3.9.13

### Python Packages Used

* General Purpose: sys, os, pathlib, dill, logging, exceptions.
* Data Manipulation: pandas, numpy
* Data Visualization: seaborn, matplotlib, python
* Machine Learning: scikit-learn

### Data Preprocessing

* Removed duplicates values using pandas.
* Imputed missing values using SimpleImputer. 
* Handled Categorical values using OneHotEncoder
* Scaled numeric values using standardScaler and numpy for better performance.

### Data Split
I used 80-20% train-test split data.

### Machine Learning Techniques using scikit-learn and hyperparameters used:

a) RandomForestRegressor : max_features, max_depth, n_estimators.

b) DecisionTreeRegressor : criterion, splitter, max_depth.
    
c) XGBoosrRegressor : learning_rate, n_estimators, max_depth.

d) AdaboostRegressor : n_estimators, learning_rate, random_state.

e) Ridge : alpha.

f) Lasso

g) LinearRegression

* I used GridSearchCV to select a best performing model with optimal hyperparameters.

## Project Analysis

## Evaluation and Result

Many candidate models were fitted into the training set. The best model that proved to be generalizing well on unseen data is Model DecisionTreeRegressor with a test score of 96% accuracy.

## Conclusion and Future Work

Please note, in principle insurance companies may use a lot more indicators than the ones used in this dataset.

Should I acquire dataset of good quality in relation to this project I will modify maybe utilize cloud services such as AWS S3, dynamoDB/RDS, CodePipeline.

## References
I haven't done any reading to assist with the analysis. My sole focus for this project is on the ml techniques.


  

  


