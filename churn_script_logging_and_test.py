'''
Python module that contatins fucntions related a model for predicting customer churn
Author: Ross Fitzgerald
Date:08/06/2022
'''

import os
import logging
from pathlib import Path
import pandas as pd
import pytest
import churn_library as cl


PATH = "./data/bank_data.csv"

cat_lst = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


# Fixtures

@pytest.fixture(scope="module", name='path')
def path():
    """
    Fixture - The test function test_import_data() 
    uses the return of path() as an argument
    """
    yield PATH


@pytest.fixture(name='df')
def df():
    """
    Fixture - The test function test_eda() and test_encoder_helper 
    uses the return of df() as an argument
    """
    yield cl.import_data(PATH)


@pytest.fixture(name='df_encoded')
def df_encoded(df):
    """
    Fixture - The test function test_perform_feature_engineering() 
    uses the return of df_encoded() as an argument
    """
    yield cl.encoder_helper(df, cat_lst)


@pytest.fixture(name='train_test_split')
def train_test_split(df_encoded):
    """
    Fixture - The test function train_models() 
    use the return of train_test_split() as an argument
    """
    return cl.perform_feature_engineering(df_encoded)


# Unit Tests
def test_import(path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cl.import_data(path)
        logging.info("SUCCESS: File Imported Successfully")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(df):
    """
    Check if EDA results are saved
    """
    cl.perform_eda(df)

    # Check if each file exists
    path = Path("./images/eda")

    for file in [
        'Churn',
        'Customer_Age',
        'Marital_Status',
        'Total_Trans_Ct',
            'Heatmap']:
        file_path = path.joinpath(f'{file}.png')
        try:
            assert file_path.is_file()
        except AssertionError as err:
            logging.error("ERROR: Graphs not found")
            raise err
    logging.info("SUCCESS: Graphs saved sucessfully")


def test_encoder_helper(df):
    '''
    test encoder helper
    '''

    # Check if dataframe is empty
    assert isinstance(df, pd.DataFrame)
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "ERROR: The dataframe doesn't appear to have rows and columns")
        raise err

    cat_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    df = cl.encoder_helper(df, cat_lst)

    # Check if categorical columns exist in df
    try:
        for col in cat_lst:
            assert col in df.columns
    except AssertionError as err:
        logging.error("ERROR: There are missing Categorical Columns")
        raise err
    logging.info(
        "SUCCESS: The Categorical Columns have been correctly encoded.")

    return df


def test_perform_feature_engineering(df_encoded):
    '''
    test perform_feature_engineering
    '''
    # Check x training sets and y training sets have same number of rows
    # Check x test sets and y test sets have same number of rows
    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
        df_encoded)

    try:
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    except AssertionError as err:
        logging.error("ERROR: The shape of train/test splits are not equal")
        raise err
    logging.info("SUCCESS: The train/test splits are correct.")

    return (X_train, X_test, y_train, y_test)


def test_train_models(train_test_split):
    '''
    test train_models
    '''

    X_train, X_test, y_train, y_test = train_test_split

    # Train model
    cl.train_models(X_train, X_test, y_train, y_test)

    # Check if model were saved after done training
    path = Path("./models")

    models = ['logistic_model.pkl', 'rfc_model.pkl']

    for model_name in models:
        model_path = path.joinpath(model_name)
        try:
            assert model_path.is_file()
        except AssertionError as err:
            logging.error("ERROR: Models are not found")
            raise err
    logging.info("SUCCESS: Models saved successfully")


if __name__ == "__main__":
    df = test_import(PATH)
    test_eda(df)
    encoded_data = test_encoder_helper(df)
    features = test_perform_feature_engineering(encoded_data)
    test_train_models(features)
