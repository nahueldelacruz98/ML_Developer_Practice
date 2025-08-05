from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler

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
    y = data['Diabetes_binary']

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

    X_train, Y_train, X_test, Y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42,shuffle=True)

    return X_train, Y_train, X_test, Y_test
    
def preprocess_data(X_train, X_test):
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
    card_columns = ['MentHlth', 'PhysHlth', 'BMI']

    #Ordinal first
    X_train[ord_columns] = ord_enc.fit_transform(X_train[ord_columns])
    X_test[ord_columns] = ord_enc.transform(X_test[ord_columns])

    #Categorical columns
    X_train[oh_columns] = oh_enc.fit_transform(X_train[oh_columns])
    X_test[oh_columns] = oh_enc.transform(X_test[oh_columns])

    


x_data, y_data = fetch_data()
short_eda(x_data=x_data, y_data=y_data)

X_train, Y_train, X_test, Y_test = split_data(x_data=x_data, y_data=y_data)

