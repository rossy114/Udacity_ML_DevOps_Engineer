# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project identifies Credit Card Customers that are likely to churn. Included is a python package for an ML project which follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested).

## Files and data description
* data
  * bank_data.csv
* images
  * eda
    * churn_distribution.png
    * customer_age_distribution.png
    * heatmap.png
    * marital_status_distribution.png
    * total_transaction_distribution.png
  * results
    * feature_importance.png
    * logistics_results.png
    * rf_results.png
    * roc_curve_result.png
  * logs
    * churn_library.log
  * models
    * logistic_model.pkl
    * rfc_model.pkl
  * churn_library.py
  * churn_script_logging_and_test.py
  * README.md


The data is stored in a csv file in the data folder. The data is pulled from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
The output plots are stored in the images folder in the eda and results folders.
There is a logs folder where all log outputs are stored in the file churn_library.log.
There is a models folder where the models logistic_model.pkl and rfc_model.pkl are stored in pickle format.
The main python module script is churn_library.py which runs the model for predicting customer churn. This incudes functions for importing data, performing EDA, performing feature transformation on categorical variables, performing feature engineering, training model, producing classification report and producing feature importance plot
There is also a python module script churn_script_logging_and_test.py which is used for testing and logging purposes.




## Running Files
The modules and their versions needed to run this project are contained in the requirements.txt file and these are:
* scikit-learn(0.24.1)
* shap(0.40.0)
* joblib(1.0.1)
* pandas(1.2.4)
* numpy(1.20.1)
* matplotlib(3.3.4)
* seaborn(0.11.2)
* pylint(2.7.4)
* autopep8(1.5.6)

### To begin create virtual environment

* python3 -m venv venv

### Activate virtual environment

* source venv/bin/activate

### Install dependencies

* pip install -r requirements.txt

### Run Main Model Script

* python churn_library.py

### Run Testing Script
To test and log results run:

* pytest

* python churn_script_logging_and_test.py




