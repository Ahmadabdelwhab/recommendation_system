from pydantic import BaseModel
from typing import  Optional

class Item(BaseModel):
        id: str
        description: str
        name:str
class UpdatedItem(BaseModel):
        description: Optional[str] = None
        name:Optional[str] = None
class Items(BaseModel):
        ids: list[str]
class ImageItem(BaseModel):
        item_id: str
        image_base64: str
        image_id:str
class Image_base64(BaseModel):
        image_base64: str

