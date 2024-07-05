import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..' , "..")))
import chromadb 
import base64
import io
from PIL import Image
from chromadb.utils import embedding_functions
from chromadb.utils.data_loaders import ImageLoader
from typing import Dict , List
from dotenv import load_dotenv
from itertools import chain
import numpy as np
# load_dotenv("..env")
ROBOFLOW_API_KEY=os.getenv("ROBOFLOW_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
SERVER_URL = os.getenv("SERVER_URL")
META_DATA = {"hnsw:space":"cosine"}
ITEM_COLLECTION_NAME="items"
IMAGE_COLLECTION_NAME="images"
PATH = "app/db"

class RecommendationDB():
    def __init__(self):
        """
        Initializes a RecommendationDB object.

        Parameters:
        - None

        Returns:
        - None
        """
        self.client = chromadb.PersistentClient(path="app/db")
        self.sentence_embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY)  
        self.image_embedding_function = embedding_functions.RoboflowEmbeddingFunction(api_key=ROBOFLOW_API_KEY ,api_url=SERVER_URL)
        self.client.get_or_create_collection(ITEM_COLLECTION_NAME , embedding_function=self.sentence_embedding_function ,metadata=META_DATA)
        self.client.get_or_create_collection(IMAGE_COLLECTION_NAME ,embedding_function= self.image_embedding_function ,data_loader=ImageLoader(), metadata=META_DATA)
        

    def compine(self, name:str , description:str) -> str:
        """
        Combines the name and description of an item.

        Args:
            name (str): The name of the item.
            description (str): The description of the item.

        Returns:
            str: The combined name and description.
        """
        compined_data = f"name: {name}, description:{description}"
        return compined_data
    def check_recommendations_exist(self, collection_name:str, item_ids:List[str]) -> bool:
        """
        Checks if a list of item IDs exists in the collection.

        Args:
            collection_name (str): The name of the collection to check.
            item_ids (List[str]): The IDs of the items to check.

        Returns:
            bool: True if all item IDs exist, False otherwise.
        """
        try:
            if not item_ids:
                raise ValueError("No item IDs provided.")
            collection = self.client.get_collection(collection_name , embedding_function=self.sentence_embedding_function)
            items = collection.get(ids=item_ids)
            return bool(items["ids"])
        except Exception as e:
            print(e)
    def convert_base64_to_numpy(self, base64_str:str) -> np.ndarray:
        """
        Converts a base64 string to a NumPy array.

        Args:
            base64_str (str): The base64 string to convert.

        Returns:
            np.ndarray: The NumPy array.
        """
        try:
            base64_bytes = base64_str.encode("utf-8")
            image_bytes = base64.b64decode(base64_bytes)
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            return image_np
        except Exception as e:
            print(e)    
    def check_id_exists(self, collection_name:str, item_id:str) -> bool:
        """
        Checks if an item ID exists in the collection.

        Args:
            collection_name (str): The name of the collection to check.
            item_id (str): The ID of the item to check.

        Returns:
            bool: True if the item ID exists, False otherwise.
        """
        try:
            print(item_id)
            collection = self.client.get_collection(collection_name , embedding_function=self.sentence_embedding_function)
            item = collection.get(ids=[item_id])
            print(item)
            return bool(len(item["ids"]))
        except Exception as e:
            print("error , " ,e)

    def get_recommendations_by_id(self, collection_name:str , item_id:str, k_recommendations:int = 10) -> List[str] | None :
        """
        Retrieves recommendations based on the given item ID.

        Args:
            collection_name (str): The name of the collection to query.
            k_recommendations (int): The number of recommendations to retrieve.
            item_id (str): The ID of the item to get recommendations for.

        Returns:
            List[str] | None: A list of recommended item IDs, or None if no recommendations are found.
        """
        try:
            
            if not self.check_id_exists(collection_name, item_id):
                return None
            collection = self.client.get_collection(collection_name , embedding_function=self.sentence_embedding_function)
            item = collection.get(ids=[item_id],
                                include=["embeddings"])
            embedding = item["embeddings"]
            results = collection.query(
                    query_embeddings=embedding,  
                    n_results=k_recommendations + 1)
            
            items_ids = results["ids"][0][1:]
            distances = results["distances"][0][1:]
            recommendations = {
                "ids": items_ids,
                "distances": distances
            }
            print("success")
            return recommendations
        except Exception as e:
            print(e)
    def get_recommendations_by_ids(self , collection_name:str , item_ids:Dict[str,List[str]] , k_recommendations:int = 50) -> Dict[str,List[str]] | None:
        """
        Retrieves recommendations based on the given item IDs.

        Args:
            collection_name (str): The name of the collection to query.
            k_recommendations (int): The number of recommendations to retrieve.
            item_ids (List[str]): The IDs of the items to get recommendations for.

        Returns:
            List[str] | None: A list of recommended item IDs, or None if no recommendations are found.
        """
        try:
            if not self.check_recommendations_exist(collection_name, item_ids["ids"]):
                return None
            collection = self.client.get_collection(collection_name , embedding_function=self.sentence_embedding_function)
            items = collection.get(ids=item_ids["ids"],
                                include=["embeddings"])
            recommendations_per_item = 10
            embeddings = items["embeddings"]
            results = collection.query(
                    query_embeddings=embeddings,  
                    n_results=recommendations_per_item)
            all_ids =results["ids"]
            unique_ids = list(set(chain.from_iterable(all_ids)) - set(item_ids["ids"]))  
            recommendations = {
                "ids": unique_ids[:min(600, k_recommendations)],
            }
            return recommendations
        except Exception as e:
            print(e)
    def get_recommendations_by_text(self, collection_name:str, text:str, k_recommendations:int) -> List[str] | None:
        """
        Retrieves recommendations based on the given text.

        Args:
            collection_name (str): The name of the collection to query.
            k_recommendations (int): The number of recommendations to retrieve.
            text (str): The text to get recommendations for.
 
        Returns:
            List[str] | None: A list of recommended item IDs, or None if no recommendations are found.
        """
        print("text : " , text)
        try:
            collection = self.client.get_collection(collection_name , embedding_function=self.sentence_embedding_function)
            results = collection.query(
                    query_texts=[text],  
                    n_results=k_recommendations)
            print(results)
            items_ids = results["ids"][0]
            distances = results["distances"][0]
            recommendations = {
                "ids": items_ids,
                "distances": distances
            }
            return recommendations
        except Exception as e:
            print(e)
    def add_item(self , collection_name:str , item:Dict[str,str]) ->Dict[str,str] | None:
            """
            Adds a new item to the collection.

            Args:
                collection_name (str): The name of the collection to add the item to.
                item (Dict): The item to add to the collection.

            Returns:
                None
            """
            try:
                if self.check_id_exists(collection_name, item["id"]):
                    return None
                collection = self.client.get_collection(collection_name , embedding_function=self.sentence_embedding_function)
                item_description = item["description"]
                item_name = item["name"]
                item_id = item["id"]
                if self.check_id_exists(collection_name, item_id):
                    print(f"Item with ID '{item_id}' already exists in collection '{collection_name}'. Skipping...")
                    return None
                text = self.compine(item_name , item_description)
                meta_data = {
                    "name": item_name,
                    "description": item_description
                }
                collection.add(
                    documents=[text],
                    ids=[item_id],
                    metadatas=meta_data
                )
                print(f"Item added to collection '{collection_name}' successfully!")
                return item
            except Exception as e:
                print(e)
    def update_embeddings_by_id(self, collection_name:str, item_id:str ,updated_item:Dict[str,str]) -> None:
        """
        Updates the embeddings for an item in the collection.

        Args:
            collection_name (str): The name of the collection to update.
            item_id (str): The ID of the item to update.

        Returns:
            None
        """
        try:
            if not self.check_id_exists(collection_name, item_id):
                print(f"Item with ID '{item_id}' does not exist in collection '{collection_name}'. Skipping...")
                return None
            collection = self.client.get_collection(collection_name , embedding_function=self.sentence_embedding_function)
            old_item = collection.get(ids=[item_id])
            print("old_item : " , old_item)
            item_description = updated_item.get("description" , old_item["metadatas"][0]["description"])
            
            item_name = updated_item.get("name" , old_item["metadatas"][0]["name"])
            text = self.compine(item_name , item_description)
            collection.update(
                ids=[item_id],
                documents=[text]
            )
            print(f"Embeddings updated for item '{item_id}' in collection '{collection_name}' successfully!")
            new_item = collection.get(ids=[item_id])
            return {
                "id": new_item["ids"][0],
                "name": new_item["metadatas"][0]["name"],
                "description": new_item["metadatas"][0]["description"]
            }
        except Exception as e:
            print(e)
    def delete_embeddings_by_id(self, collection_name:str, item_id:str) -> None:
        """
        Deletes the embeddings for an item in the collection.

        Args:
            collection_name (str): The name of the collection to update.
            item_id (str): The ID of the item to update.

        Returns:
            None
        """
        try:
            if not self.check_id_exists(collection_name, item_id):
                print(f"Item with ID '{item_id}' does not exist in collection '{collection_name}'. delete failed...")
                return None
            collection = self.client.get_collection(collection_name , embedding_function=self.sentence_embedding_function)
            collection.delete(ids=[item_id])
            print(f"Embeddings deleted for item '{item_id}' in collection '{collection_name}' successfully!")
            return {
                "status" : "success"
            }
        except Exception as e:
            print(e)
    def add_image(self , collection_name:str , image_item:Dict[str , str]) -> Dict[str , str]| None:
        try:
            if self.check_id_exists(collection_name, image_item["image_id"]):
                print(f"Image with ID '{image_item['image_id']}' already exists in collection '{collection_name}'. Skipping...")
                return None
            
            collection = self.client.get_collection(collection_name, embedding_function=self.image_embedding_function)
            image_id = image_item["image_id"]
            item_id = image_item["item_id"]
            item_base64 = image_item["image_base64"]
            converted_image = self.convert_base64_to_numpy(item_base64)
            item_metadata = {
                "item_id": item_id,
                "image_id": image_id,
                
            }
            collection.add(
            images=[converted_image],
            ids=[image_id],
            metadatas=item_metadata
            )
            print(f"Image added to collection '{collection_name}' successfully!")
            return image_item
        except Exception as e:
            print(e)
    from typing import List, Optional
import numpy as np

def get_image_recommendations(self, collection_name: str, image_base64: str, k_recommendations: int = 10) -> Dict[str, List[str]] | None:
    """
    Retrieves similar images based on the given image.

    Args:
        collection_name (str): The name of the image collection.
        image_base64 (str): The base64 encoded image to get recommendations for.
        k_recommendations (int): The number of recommendations to retrieve.

    Returns:
        Optional[List[str]]: A list of recommended item IDs, or None if no recommendations are found.
    """
    try:
        np_image = self.convert_base64_to_numpy(image_base64)
        collection = self.client.get_collection(collection_name, embedding_function=self.image_embedding_function, data_loader=ImageLoader())
        
        results = collection.query(
            query_images=[np_image],  
            n_results=k_recommendations
        )
        print(results)
        
        metadata = results["metadatas"][0]
        ids_list = [item["item_id"] for item in metadata]
        distances = results["distances"][0]
        
        # Sorting the ids by distance and removing duplicates
        combined_list = list(zip(ids_list, distances))
        min_distance_dict = {}
        for item_id, distance in combined_list:
            if item_id not in min_distance_dict or distance < min_distance_dict[item_id]:
                min_distance_dict[item_id] = distance
        
        sorted_items = sorted(min_distance_dict.items(), key=lambda x: x[1])
        sorted_ids_list = [item[0] for item in sorted_items]
        limit = min(len(sorted_ids_list), k_recommendations)
        recommendations = {"ids":sorted_ids_list[:limit]}
        print({"ids": recommendations})
        
        return recommendations
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
