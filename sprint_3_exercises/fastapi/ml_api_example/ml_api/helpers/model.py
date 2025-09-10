import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier


# Load sklearn model
with open("assets/model.pkl", "rb") as f:
    model: MLPClassifier = pickle.load(f)
# Load preprocessor
with open("assets/preprocessor.pkl", "rb") as f:
    preprocessor: ColumnTransformer = pickle.load(f)


def predict(
    gender: str,
    age: int,
    hypertension: int,
    heart_disease: int,
    smoking_history: str,
    bmi: float,
    HbA1c_level: float,
    blood_glucose_level: int,
) -> float:
    # Convert input data to a DataFrame as expected by the preprocessor
    input_array = pd.DataFrame(
        {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "HbA1c_level": HbA1c_level,
            "blood_glucose_level": blood_glucose_level,
        },
        index=[0],
    )

    processed_input = preprocessor.transform(input_array)
    prediction = float(model.predict(processed_input))

    return prediction
