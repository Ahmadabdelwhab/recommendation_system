import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..' , "..")))
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.models.item import Item , UpdatedItem , Items ,ImageItem , Image_base64 
from app.models.recommendation import Recommendations , ImageRecommendations
from app.db.recommendationDB import RecommendationDB
from typing import List

router = APIRouter()
COLLECTION_NAME = "items"
IMAGE_COLLECTION_NAME = "images"
db = RecommendationDB()
######post######
@router.post("/item/")
async def add_item(item: Item):
        ret = db.add_item( COLLECTION_NAME, item.__dict__)
        if not ret:
                raise HTTPException(status_code=409, detail="Item already exists")
        return item
@router.post("/items/ids")
async def get_recommendations_by_ids(items :Items , limit:int = 50):
        limit = min(50,limit)
        ret = db.get_recommendations_by_ids(COLLECTION_NAME , items.model_dump() , k_recommendations=limit)
        print(ret)
        if not ret:
                raise HTTPException(status_code=404 , detail="No recommendations available")
        else:
                return Items(**ret) 

@router.post("/item/image")
async def add_image(image_item: ImageItem):
        ret = db.add_image(IMAGE_COLLECTION_NAME, image_item.__dict__)
        if not ret:
                raise HTTPException(status_code=404, detail="Item already exists")
        return ImageItem(**ret)
@router.post("/item/image_base64")
async def get_image_recommendations(image_base64: Image_base64 , limit:int = 10):
        limit = max(1,limit)
        ret = db.get_image_recommendations(IMAGE_COLLECTION_NAME , image_base64.__dict__ , k_recommendations=limit)
        if not ret:
                raise HTTPException(status_code=404 , detail="No recommendations available")
        else:
                return ImageRecommendations(**ret)
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
@router.get("/dbfile")
async def download_chroma():
    file_path = "app/db/chroma.sqlite3"  # Specify the path to your file
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename="chroma.sqlite3", media_type="application/octet-stream")
    else:
        return {"error": "File not found"}

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
@router.delete("/item/image/{image_id}")
async def delete_image(image_id:str):
        ret = db.delete_embeddings_by_id( IMAGE_COLLECTION_NAME, image_id)
        if not ret:
                raise HTTPException(status_code=404, detail=f"no image with {image_id} is found")
        return ret

