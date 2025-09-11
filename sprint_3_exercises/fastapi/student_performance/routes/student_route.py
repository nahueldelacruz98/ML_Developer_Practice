from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np

router = APIRouter()
path_csv_data = "./data/student-mat.csv"
#df_base = pd.read_csv(path_csv_data,delimiter=";")

@router.post("/student/new_student")
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

    return {
        "new_row" : df_row_added.to_json(orient='records'),
        "rows_database": len(df_base)
    }


@router.get("/student/students_by_parent_job")
def get_students_by_parent_job(parent_job:str):
    df = pd.read_csv(path_csv_data, delimiter=";")
    #filter by "parent_job" (Fjob ; Mjob)
    df_filtered = df[(df["Fjob"] == parent_job) | (df["Mjob"] == parent_job)]

    if len(df_filtered) == 0:
        raise HTTPException(status_code = 404, detail=f"Cannot find any student where its parent (mom or dad) has the following job: {parent_job}")
    
    #Get top 10
    df_filtered = df_filtered.head(10)

    return {
        "data": df_filtered.to_json(orient='records'),
        "count": len(df_filtered)
    }

@router.get("/student/top_students_less_absences")
def get_top_students_less_absences(n_top:int):
    df = pd.read_csv(path_csv_data, delimiter=';')
    df_top_less_absences = df.sort_values(by="absences",ascending=True).head(n_top)

    list_columns = ['school','sex','age','address','absences','G1','G2','G3']
    df_top_less_absences = df_top_less_absences[list_columns]

    return {
        "data": df_top_less_absences.to_json(orient='records'),
        "count": len(df_top_less_absences)
    }