from fastapi import FastAPI
from routes.api import router


app = FastAPI()
app.include_router(router)


@app.get("/")
def root():
    return {"message": "Hello, World!"}


# test it by passing parameters through url: http://127.0.0.1:8000/add/1/2
@app.get("/add/{num1}/{num2}")
def add_url(num1: int, num2: int):
    """This function takes two url parameters, num1 and num2, and returns a dictionary with the key "sum" and the value of the sum of both."""
    return {"sum": num1 + num2}


# test it by passing query parameters: http://127.0.0.1:8000/add?num1=1&num2=2
@app.get("/add")
def add_get(num1: int, num2: int):
    """This function takes two query parameters, num1 and num2, and returns a dictionary with the key "sum" and the value of the sum of both."""
    return {"sum": num1 + num2}
