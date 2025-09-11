from fastapi import FastAPI
from routers.skillshots_router import router as router_skillshots

app = FastAPI()

@app.get("/")
def main():
    return {"message":"homepage"}

app.include_router(router_skillshots)
