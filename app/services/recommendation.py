from app.db.recommendationDB import collection
from typing import List, Dict
from app.models.item import Item

def get_recommendations(embedding: List[float], top_k: int, items_db: Dict[str, Item], exclude_id: str) -> List[Item]:
        results = collection.query(embedding, k=top_k + 1)
        recommended_ids = [result[0] for result in results if result[0] != exclude_id][:top_k]
        return [items_db[rec_id] for rec_id in recommended_ids]
        