from fastapi import FastAPI

from app.api.routes import router as v1_router

app = FastAPI(title="Medical Robustness Agent (Plan A)", version="2.1.0")
app.include_router(v1_router, prefix="/v1")
