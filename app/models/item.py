from pydantic import BaseModel
from typing import List

class Item(BaseModel):
        id: str
        description: str
        name:str
class UpdatedItem(BaseModel):
        description: str
        name:str

