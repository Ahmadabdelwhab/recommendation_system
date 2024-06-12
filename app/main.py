from fastapi import FastAPI
from app.api.endpoints import items


app = FastAPI()
app.include_router(items.router, prefix="/items")
