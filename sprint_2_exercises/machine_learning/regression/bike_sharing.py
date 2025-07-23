from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from ucimlrepo import fetch_ucirepo # dataset

# Link of the dataset: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

# Steps
# 1. Fetch dataset and study its content and structure
# 2. Determine features and targets
# 3. Split data and prepare for training
# 4. Train a model and evaluate its performance

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

    df_features = df_bike_sharing[['dteday','hr','weathersit', 'season','temp','windspeed','hum']]
    df_features['dteday'] = pd.to_datetime(df_features['dteday'])   #convert date to continuous
    df_target = df_bike_sharing[['cnt']]

    return train_test_split(df_features, df_target, test_size=0.2, random_state=42)

def train_linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X=X_train, y=y_train)

    return model

def train_sgd_regression_model(X_train, y_train):
    model = SGDRegressor(max_iter = 100, random_state=42)
    model.fit(X=X_train, y=y_train)

    return model

def calculate_metrics_predictions(y_predictions, y_test):
    mae = mean_absolute_error(y_test, y_predictions)
    mse = mean_squared_error(y_test, y_predictions)
    r2 = r2_score(y_test, y_predictions)

    print(f"Mean absolute error: {mae}")
    print(f"Mean Squared error: {mse}")
    print(f"R2 score: {r2}")


df_bike_sharing = fetch_data()

print(f"amount of records: {df_bike_sharing.shape}")

# Split the data into features and targets, then split the data
X_train, X_test, y_train, y_test = split_data(df_bike_sharing)
# Train a model
#model = train_linear_regression_model(X_train, y_train)
model = train_sgd_regression_model(X_train, y_train)
print("Model trained")
# Testing model
y_predictions = model.predict(X_test)
print("Predictions already done")
# calculate metrics
calculate_metrics_predictions(y_predictions= y_predictions, y_test= y_test)