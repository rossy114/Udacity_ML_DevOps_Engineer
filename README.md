# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project identifies Credit Card Customers that are likely to churn. Included is a python package for an ML project which follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested).

## Files and data description
The main python module script is churn_library.py which runs the model for predicting customer churn. This incudes functions for importing data, performing EDA, performing feature transformation on categorical variables, performing feature engineering, training model, prodcuing classification report and producing feature importance plot, 
There is also a python module script churn_script_logging_and_tests.py  which is used for unit testing and logging purposes.
There is a logs folder where all log outputs are stored
There is a models folder where models are stored
The data is stored in a csv file in the data folder. The data is pulled from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
The output plots are stored in the images folder in the eda and results folders.


## Running Files
*Create virtual environment

python3 -m venv venv

*Activate virtual environment

source venv/bin/activate

##Install dependencies

pip install -r requirements.txt

*Run Main Model Script

python churn_library.py

*Run Testing Script
To test and log results run:
pytest
python churn_script_logging_and_test.py




