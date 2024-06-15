from fastapi import APIRouter, HTTPException
from app.models.item import Item
from app.models.item import UpdatedItem
from app.models.recommendation import Recommendations
from app.db.recommendationDB import RecommendationDB
from typing import List

router = APIRouter()

######post######
@router.post("/item/")
async def add_item(item: Item):
        db = RecommendationDB()
        ret = db.add_item( "my_items", item.__dict__)
        if not ret:
                raise HTTPException(status_code=409, detail="Item already exists")
        return item


#######get######
@router.get("/item/id/{item_id}")
async def get_recommendation_by_id(item_id: str , limit:int =10):
        limit = max(1,limit)
        db = RecommendationDB()
        ret = db.get_recommendations_by_id( "my_items", item_id , k_recommendations=limit)
        if not ret:
                raise HTTPException(status_code=404, detail=f"no item with {item_id} is found")
        return Recommendations(**ret)

@router.get("/item/text/{text}")
async def get_recommendation_by_id(text: str , limit:int =10):
        limit = max(1,limit)
        db = RecommendationDB()
        ret = db.get_recommendations_by_text( "my_items", text , k_recommendations=limit)
        return Recommendations(**ret)

@router.patch("/item/{item_id}")
async def update_item(item_id:str , updated_item: UpdatedItem):
        db = RecommendationDB()
        ret = db.update_embeddings_by_id( "my_items", item_id, update_item.__dict__)
        if not ret:
                raise HTTPException(status_code=404, detail=f"no item with {item_id} is found")
        return ret
@router.delete("/item/{item_id}")
async def update_item(item_id:str , updated_item: UpdatedItem):
        db = RecommendationDB()
        ret = db.delete_embeddings_by_id( "my_items", item_id)
        if not ret:
                raise HTTPException(status_code=404, detail=f"no item with {item_id} is found")
        return ret