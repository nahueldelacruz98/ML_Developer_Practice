from fastapi import FastAPI
from routes.school_route import router as school_route
from routes.student_route import router as student_route

api = FastAPI()

@api.get("/")
def main():
    return {"message":"Empty page. Please follow up the API guide here --> http://127.0.0.1:8000/docs"}

api.include_router(school_route)
api.include_router(student_route)