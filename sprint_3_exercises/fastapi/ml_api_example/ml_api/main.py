from fastapi import FastAPI
from routes.model_routes import router as model_router
from helpers.database import create_table

app = FastAPI()

create_table()


@app.get("/")
def root():
    return {"message": "Hello ML API!"}


app.include_router(model_router)
