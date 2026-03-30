from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.model import router as model_router

app = FastAPI(
    title="Marketing Communications Backend",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_router)


@app.get("/")
def root():
    return {"message": "Marketing Communications Backend API is running"}