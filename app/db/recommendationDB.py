__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb 
from chromadb.utils import embedding_functions 
from typing import Dict , List

PATH = "app/db"
# todo
# --- 1. Create data base initilization
# --- 2. create database collections
# --- 3. get recommendations by itemID
# --- 4. get reccomendations by text 
# --- 5. add item to collection
# --- 5. update empeddings by id
# --- 6. Delete embeddings by id

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
        self.sentence_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction("all-mpnet-base-v2")     
        self.client.get_or_create_collection("items" , embedding_function=self.sentence_embedding_function , )
        

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
            collection = self.client.get_collection(collection_name , embedding_function=self.sentence_embedding_function)
            item = collection.get(ids=[item_id])
            print(item)
            print(len(item["ids"]))
            return bool(len(item["ids"]))
        except Exception as e:
            print("error , " ,e)
    
        except chromadb.exceptions.CollectionAlreadyExistsError:
            print(f"Collection '{collection_name}' already exists.")

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
            collection = self.client.get_collection(collection_name , embedding_function=self.sentence_embedding_function)
            item = collection.get(ids=[item_id],
                                include=["embeddings"])
            
            if not self.check_id_exists(collection_name, item_id):
                return None
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
            item_description = updated_item["description"]
            item_name = updated_item["name"]
            text = self.compine(item_name , item_description)
            collection.update(
                ids=[item_id],
                documents=[text]
            )
            self.client.get_collection(collection_name , embedding_function=self.sentence_embedding_function).get()
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
            if self.check_id_exists(collection_name, item_id):
                print(f"Item with ID '{item_id}' does not exist in collection '{collection_name}'. delete failed...")
                return None
            collection = self.client.get_collection(collection_name , embedding_function=self.sentence_embedding_function)
            collection.delete(ids=[item_id])
            print(f"Embeddings deleted for item '{item_id}' in collection '{collection_name}' successfully!")
        except Exception as e:
            print(e)
    def delete_all_embeddings(self, collection_name:str) -> None:
        """
        Deletes all embeddings in the collection.

        Args:
            collection_name (str): The name of the collection to update.

        Returns:
            None
        """
        try:
            collection = self.client.get_collection(collection_name , embedding_function=self.sentence_embedding_function)
            collection.delete_all()
            print(f"All embeddings deleted in collection '{collection_name}' successfully!")
        except Exception as e:
            print(e)

