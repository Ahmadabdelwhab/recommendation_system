from pydantic import BaseModel
from typing import List

class Recommendations(BaseModel):
    ids: List[str]
    distances: List[float]
class ImageRecommendations(BaseModel):
    ids: List[str]
