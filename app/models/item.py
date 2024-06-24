from pydantic import BaseModel
from typing import  Optional

class Item(BaseModel):
        id: str
        description: str
        name:str
class UpdatedItem(BaseModel):
        description: Optional[str] = None
        name:Optional[str] = None

