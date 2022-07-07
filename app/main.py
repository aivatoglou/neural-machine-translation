from fastapi import FastAPI
from routes.rest import router as ServicesRouter

app = FastAPI()
app.include_router(ServicesRouter)


@app.get("/", tags=["root"])
async def root():
    return {"message": "Welcome!"}
