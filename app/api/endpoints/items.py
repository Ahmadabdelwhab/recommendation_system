import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..' , "..")))
from fastapi import APIRouter, HTTPException
from app.models.item import Item
from app.models.item import UpdatedItem
from app.models.recommendation import Recommendations
from app.db.recommendationDB import RecommendationDB
from typing import List

router = APIRouter()
COLLECTION_NAME = "items"
db = RecommendationDB()
######post######
@router.post("/item/")
async def add_item(item: Item):
        ret = db.add_item( COLLECTION_NAME, item.__dict__)
        if not ret:
                raise HTTPException(status_code=409, detail="Item already exists")
        return item


#######get######
@router.get("/item/id/{item_id}")
async def get_recommendation_by_id(item_id: str , limit:int =10):
        limit = max(1,limit)
        ret = db.get_recommendations_by_id( COLLECTION_NAME, item_id , k_recommendations=limit)
        if not ret:
                raise HTTPException(status_code=404, detail=f"no item with {item_id} is found")
        return Recommendations(**ret)


@router.get("/item/text/{text}")
async def get_recommendation_by_text(text: str , limit:int =10):
        limit = max(1,limit)
        ret = db.get_recommendations_by_text( COLLECTION_NAME, text , k_recommendations=limit)
        return Recommendations(**ret)

@router.patch("/item/{item_id}")
async def update_item(item_id:str , updated_item: UpdatedItem):
        print(updated_item.__dict__)
        print(item_id)
        ret = db.update_embeddings_by_id( COLLECTION_NAME, item_id, updated_item.__dict__)
        if not ret:
                raise HTTPException(status_code=404, detail=f"no item with {item_id} is found")
        return ret
@router.delete("/item/{item_id}")
async def delete_item(item_id:str):
        ret = db.delete_embeddings_by_id( COLLECTION_NAME, item_id)
        if not ret:
                raise HTTPException(status_code=404, detail=f"no item with {item_id} is found")
        return ret