from fastapi import FastAPI
from app.api.endpoints import items


app = FastAPI()
app.include_router(items.router, prefix="/api/v1")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)