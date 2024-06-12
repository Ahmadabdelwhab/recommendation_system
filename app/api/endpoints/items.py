from fastapi import APIRouter, HTTPException
from app.models.item import Item
from app.services.embeddings import generate_embedding
from app.services.recommendation import get_recommendations
from app.db.recommendationDB import collection
from typing import List

router = APIRouter()

items_db={}

@router.post("/add_item/")
async def add_item(item: Item):
        embedding = generate_embedding(item.description)
        item.embedding = embedding
        items_db[item.id] = item
        collection.insert(item.id, embedding)
        return {"message": "Item added successfully", "item_id": item.id}

@router.get("/recommend/{item_id}")
async def recommend(item_id: str, top_k: int = 5):
        if item_id not in items_db:
                raise HTTPException(status_code=404, detail="Item not found")
        embedding = items_db[item_id].embedding
        recommendations = get_recommendations(embedding, top_k, items_db, item_id)
        return {"recommendations": recommendations}
