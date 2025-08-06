from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, r2_score

def fetch_data():
    '''
    Return
    X : pd.DataFrame (Features of dataset)
    y : pd.Dataframe (Target of dataset)
    '''
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
    # X = cdc_diabetes_health_indicators.data.features 
    # y = cdc_diabetes_health_indicators.data.targets 

    data = cdc_diabetes_health_indicators.data.original

    X = data[['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
       'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
       'Income']]
    y = data[['Diabetes_binary']]

    return X,y

def short_eda(x_data, y_data):
    '''
    Key points:
    1. Shape, columns and types.
    2. Study of sensitive data: Gender, Income and Education level.
    3. Study of other important features (plot features using matplotlib).
    '''
    print(f'Instances of dataset: {x_data.shape}')
    print()
    #print(x_data.head(5))
    print(f'Columns and Types: \n{x_data.dtypes}' )
    
    #Plot all features to find correlations between features and target
    df_combined = pd.concat([x_data,y_data], axis=1) #join both features and targets to plot them

    #Now compute the correlation matrix
    plt.figure(figsize=(20,10))
    sns.heatmap(df_combined.corr(), annot=True, cmap='coolwarm',fmt='.2f')
    plt.title('Correlation heatmap')
    plt.show()

def split_data(x_data, y_data):
    '''
    Feature selection based on the EDA:
    # Features with more correlation
    # with target: GenHlth; HighBP; HighChol; BMI; DiffWalk; 
    # with other features: [PhysHlth : GenHlth]; [DiffWalk : PhysHlth] ; [DiffWalk : GenHlth] ; [Education : Income] ; [HighBP : Age] ; [MentHlth : PhysHlth] ; [HighChol : HighBP]

    '''

    x_data = x_data[['MentHlth','PhysHlth','GenHlth','HighBP','HighChol','BMI','DiffWalk', 'Age','Sex']]

    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42,shuffle=True)

    print(f'Shape X train: {X_train.shape}')
    print(f'Shape X test: {X_test.shape}')
    print(f'Shape Y train: {Y_train.shape}')
    print(f'Shape Y test: {Y_test.shape}')

    return X_train, Y_train, X_test, Y_test
    
def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    '''
    Although the EDA says all features are numeric, some of them are categorical or binary.
    So we need to encode both binary and not-binary column categories
    Binary: [DiffWalk,Sex,HighBP, HighChol]
    Categorical: [Age, GenHlth,]
    Cardinal: []
    '''
    oh_enc = OneHotEncoder()
    ord_enc = OrdinalEncoder()
    min_max_scaler = MinMaxScaler()

    ord_columns = ['DiffWalk','Sex','HighBP', 'HighChol']
    oh_columns = ['Age', 'GenHlth']
    #card_columns = ['MentHlth', 'PhysHlth', 'BMI']

    #Ordinal first
    X_train[ord_columns] = ord_enc.fit_transform(X_train[ord_columns])
    X_test[ord_columns] = ord_enc.transform(X_test[ord_columns])

    #Categorical columns
    ohenc_train = oh_enc.fit_transform(X_train[oh_columns])
    ohenc_test = oh_enc.transform(X_test[oh_columns])

    df_train_ohe = pd.DataFrame(
        ohenc_train,
        columns=oh_enc.get_feature_names_out(input_features=oh_columns),
        index= X_train.index
    )

    df_test_ohe = pd.DataFrame(
        ohenc_test,
        columns=oh_enc.get_feature_names_out(input_features=oh_columns),
        index= X_test.index
    )

    X_train = X_train.drop(columns=oh_columns, axis=1)
    X_test = X_test.drop(columns=oh_columns, axis=1)

    X_train = pd.concat([X_train,df_train_ohe],axis=1)
    X_test = pd.concat([X_test,df_test_ohe],axis=1)

    # feature scaling with minmax scaler
    minmax_train = min_max_scaler.fit_transform(X_train)
    minmax_test = min_max_scaler.fit_transform(X_test)

    X_train = pd.DataFrame(
        minmax_train,
        columns=X_train.columns.tolist(),
        index=X_train.index
    )

    X_test = pd.DataFrame(
        minmax_test,
        columns=X_test.columns.tolist(),
        index = X_test.index
    )

    return X_train, X_test

def train_model(X_train:pd.DataFrame, Y_train:pd.DataFrame):
    '''
    
    '''
    
    rf_model = RandomForestClassifier(n_estimators=200,criterion='entropy',max_depth=200, min_samples_leaf=2, max_features='sqrt',
                                      n_jobs=-1,random_state=42, class_weight='balanced')
    
    print("start training")
    rf_model.fit(X_train, Y_train)
    print("ends training")
    return rf_model


def show_prediction_metrics(Y_predictions, Y_test):
    '''
    Compare both results (predictions) and the actual results.
    Use different metrics: R2 score, roc_auc_score
    '''
    r2_results = r2_score(y_pred=Y_predictions, y_true=Y_test)
    roc_auc_metric = roc_auc_score(y_true=Y_test, y_score=Y_predictions)

    print(f'Roc AUC score: {roc_auc_metric}')
    print(f'R2 score: {r2_results}')


x_data, y_data = fetch_data()
#short_eda(x_data=x_data, y_data=y_data)

X_train, Y_train, X_test, Y_test = split_data(x_data=x_data, y_data=y_data)

X_train, X_test = preprocess_data(X_train=X_train, X_test=X_test)

model = train_model(X_train, Y_train)

predictions = model.predict(X_test)
print('Predictions done. Check metrics')

show_prediction_metrics(predictions)