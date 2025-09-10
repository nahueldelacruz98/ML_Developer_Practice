from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np

router = APIRouter()
path_csv_data = "./data/student-mat.csv"
#df_base = pd.read_csv(path_csv_data,delimiter=";")

@router.post("/new_student")
def create_new_student(school:str, sex:str, age:int, g1:str, g2:str, g3:int):
    '''
    It creates a new student and then it adds it to the database. It leaves the other columns empty.
    '''
    df_base = pd.read_csv(path_csv_data,delimiter=";")

    #create new dataframe to add a new row
    df_new_row = pd.DataFrame([{
        "school":school,
        "age":age,
        "sex":sex,
        "G1":g1,
        "G2":g2,
        "G3":g3
    }])

    df_base = pd.concat([df_base,df_new_row],ignore_index=True)

    #return the last student added
    df_row_added = df_base.tail(1)
    
    df_row_added = df_row_added.applymap(
        lambda x: None if pd.isna(x) else (int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x)
    )

    return {
        "new_row" : df_row_added.to_dict(orient='records'),
        "rows_database": len(df_base)
    }
