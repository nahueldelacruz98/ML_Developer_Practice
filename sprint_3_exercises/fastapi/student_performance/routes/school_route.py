from fastapi import APIRouter,HTTPException
import pandas as pd

router = APIRouter()
path_csv_data = "./data/student-mat.csv"

def get_database():
    df_base = pd.read_csv(path_csv_data,delimiter=";")
    return df_base

#GET schools
@router.get("/school/get_schools")
def get_schools():
    df = get_database()

    #list all schools
    list_schools = df["school"].unique().tolist()
    return {
        "schools":list_schools,
        "count":len(list_schools)
    }

@router.get("/school/students_per_school")
def get_students_per_school(school_name:str):
    df = get_database()

    #check if school belongs to the list
    if(school_name not in get_schools()["schools"]):
        raise HTTPException(status_code=400, detail=f'The school {school_name} does not exist in our database.')
    
    #filter by school and return all students. Filter required columns
    list_columns = ['school','sex','age','address','famsize','Pstatus','Medu','Fedu']

    df_school_students = df[df["school"] == school_name]
    df_school_students = df_school_students[list_columns]

    return {
        "students": df_school_students.to_json(orient='records'),
        "count": len(df_school_students)
    }

@router.get("/school/students_best_performance")
def get_students_best_performance(top:int, min_score:int, school_name:str):
    df = get_database()
    
    #check if school belongs to the list
    if(school_name not in get_schools()["schools"]):
        raise HTTPException(status_code='400', detail=f'The school {school_name} does not exist in our database.')
    
    #sort students by final grade score
    df_students_best_score = df[(df['school'] == school_name) & (df['G3'] >= min_score)].sort_values(by="G3",axis=0,ascending=False)
    # get the top N students
    df_students_best_score = df_students_best_score.head(top)

    return {
        "students": df_students_best_score.to_dict(orient='records'),
        "count":len(df_students_best_score),
        "max_score":int(df_students_best_score.iloc[0]['G3'])   #numpy values are not allowed. Remember to parse data.
    }

