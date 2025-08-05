from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder, MinMaxScaler, MaxAbsScaler, RobustScaler, OrdinalEncoder
from sklearn.ensemble import IsolationForest

from xgboost import XGBClassifier

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo # dataset

# Link of the dataset: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

# Steps
# 1. Fetch dataset and study its content and structure
# 2. Determine features and targets
# 3. Split data and prepare for training
# 4. Train a model and evaluate its performance

scaler = StandardScaler()

def fetch_data():

    data_bike_sharing = fetch_ucirepo(id=275)

    #print(f"Features from API: {data_bike_sharing.data.features}")
    #print(f"Targets from API: {data_bike_sharing.data.targets}")

    return data_bike_sharing.data.original

# PROBLEM: predict the amount of rental bikes in a day given the weather and other features.
# Features: weathersit (C) ; season (C) ; workingday (Binary)
# Target: cnt (Continuous)

# IMPORTANT: the dataset is already preprocessed, we can use it directly.

def split_data(df_bike_sharing):

    df_features = df_bike_sharing[['dteday','hr','mnth','yr','weathersit', 'season','temp','windspeed','hum','holiday','workingday']]
    df_target = df_bike_sharing[['cnt']]

    return train_test_split(df_features, df_target, test_size=0.25, random_state=42)

def train_model(X_train, y_train):

    model_name = "xgboost" # (linear regression ; sdg regressor ; xgboost

    if model_name.lower() == "linear regression":
        model = LinearRegression()
    elif model_name.lower() == 'sdg regressor':
        model = SGDRegressor(max_iter = 50, tol=1e-3, random_state=42)
    elif model_name.lower() == 'xgboost':
        model = XGBClassifier(n_estimators=4, max_depth=10, learning_rate=1)

    model.fit(X=X_train, y=y_train)

    return model

def train_sgd_regression_model(X_train, y_train):
    model = SGDRegressor(max_iter = 50, tol=1e-3, random_state=42)
    model.fit(X=X_train, y=y_train)

    return model

def calculate_metrics_predictions(y_predictions, y_test):
    mae = mean_absolute_error(y_test, y_predictions)
    mse = mean_squared_error(y_test, y_predictions)
    r2 = r2_score(y_test, y_predictions)

    print(f"Mean absolute error: {mae}")
    print(f"Mean Squared error: {mse}")
    print(f"R2 score: {r2}")

def clean_outliers(X_train, Y_train):
    print("Handling outliers...")
    iso = IsolationForest(contamination=0.005, random_state=42)
    outliers = iso.fit_predict(X_train)

    # Remove outliers (keep only inliers where prediction == 1)
    X_cleaned = X_train[outliers == 1]
    Y_cleaned = Y_train[outliers == 1]

    return X_cleaned, Y_cleaned

def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame, Y_train: pd.DataFrame):
    # convert datetime column to numeric
    X_train['dteday'] = pd.to_datetime(X_train['dteday'])
    X_test['dteday'] = pd.to_datetime(X_test['dteday'])
    X_train['dteday'] = (X_train['dteday'] - X_train['dteday'].min()).dt.days
    X_test['dteday'] = (X_test['dteday'] - X_test['dteday'].min()).dt.days
    X_train['dteday_num'] = scaler.fit_transform(X_train[['dteday']])
    X_test['dteday_num'] = scaler.transform(X_test[['dteday']])
    X_train = X_train.drop(columns=['dteday'])
    X_test = X_test.drop(columns=['dteday'])

    #Encode binary columns
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    list_binary_columns = ['holiday','workingday']
    encoder.fit(X_train[list_binary_columns])

    X_train[list_binary_columns] = encoder.transform(X_train[list_binary_columns])
    X_test[list_binary_columns] = encoder.transform(X_test[list_binary_columns])

    #Encode categorical columns
    # [yr,mnth,hr,weekday,weathersit,season]
    #Using one-hot encoder
    encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    #list_categorical_columns = X_train.select_dtypes(include='object').columns.to_list()
    list_categorical_columns = ['hr','mnth','yr','weathersit','season']

    print(f'Categorical columns of train/test data: {list_categorical_columns}')

    encoder.fit(X_train[list_categorical_columns])
    ohe_train = encoder.transform(X_train[list_categorical_columns])
    ohe_test = encoder.transform(X_test[list_categorical_columns])

    #Pass the OneHot encoder results into a dataframe, and then concatenate the new columns

    df_ohe_train = pd.DataFrame(
        ohe_train,
        columns = encoder.get_feature_names_out(input_features=list_categorical_columns), # Assign new columns of onehot encoder results
        index = X_train.index
    )

    df_ohe_test = pd.DataFrame(
        ohe_test,
        columns = encoder.get_feature_names_out(input_features=list_categorical_columns),
        index = X_test.index
    )

    X_train = pd.concat([X_train, df_ohe_train], axis=1)
    X_test = pd.concat([X_test, df_ohe_test], axis=1)

    X_train = X_train.drop(columns=list_categorical_columns)
    X_test = X_test.drop(columns=list_categorical_columns)

    #remove outliers
    X_train, Y_train = clean_outliers(X_train, Y_train)

    # feature scaling columns with MinMax scaler

    #min_max_scaler = MinMaxScaler()
    #min_max_scaler = MaxAbsScaler()
    min_max_scaler = RobustScaler()
    df_train_scaled = min_max_scaler.fit_transform(X_train)
    df_test_scaled = min_max_scaler.transform(X_test)

    X_train = pd.DataFrame(
        df_train_scaled,
        columns=X_train.columns.tolist(),
        index = X_train.index
    )

    X_test = pd.DataFrame(
        df_test_scaled,
        columns=X_test.columns.tolist(),
        index = X_test.index
    )

    return X_train, X_test, Y_train

# CODE STARTS  
df_bike_sharing = fetch_data()

print(f"amount of records: {df_bike_sharing.shape}")

# Split the data into features and targets, then split the data
X_train, X_test, y_train, y_test = split_data(df_bike_sharing)
# Train a model
#model = train_linear_regression_model(X_train, y_train)

X_train, X_test, y_train = preprocess_data(X_train, X_test, y_train)

print(X_train.head(3))

model = train_sgd_regression_model(X_train, y_train)
print("Model trained")
# Testing model
y_predictions = model.predict(X_test)
print("Predictions already done")
# calculate metrics
calculate_metrics_predictions(y_predictions= y_predictions, y_test= y_test)