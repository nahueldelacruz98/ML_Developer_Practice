import sqlite3

DATABASE_PATH = "data/predictions.db"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    return conn


def create_table():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            gender VARCHAR(10),
            age INT,
            hypertension INT,
            heart_disease INT,
            smoking_history VARCHAR(20),
            bmi FLOAT,
            HbA1c_level FLOAT,
            blood_glucose_level INT,
            prediction FLOAT
        );
    """
    )
    conn.commit()
    cur.close()
    conn.close()


def insert_prediction(
    gender: str,
    age: int,
    hypertension: int,
    heart_disease: int,
    smoking_history: str,
    bmi: float,
    HbA1c_level: float,
    blood_glucose_level: int,
    prediction: float,
):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        f"""
        INSERT INTO predictions (gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, prediction)
        VALUES ("{gender}", {age}, {hypertension}, {heart_disease}, "{smoking_history}", {bmi}, {HbA1c_level}, {blood_glucose_level}, {prediction});
    """
    )
    conn.commit()
    cur.close()
    conn.close()


def get_predictions():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions")
    predictions = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "gender": p[0],
            "age": p[1],
            "hypertension": p[2],
            "heart_disease": p[3],
            "smoking_history": p[4],
            "bmi": p[5],
            "HbA1c_level": p[6],
            "blood_glucose_level": p[7],
            "prediction": p[8],
        }
        for p in predictions
    ]
