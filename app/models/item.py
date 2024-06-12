from pydantic import BaseModel
from typing import List

class Item(BaseModel):
        id: str
        description: str
        embedding: List[float] = []
