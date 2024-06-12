import chromadb
from chromadb.utils import embedding_functions 
from typing import Dict , List

PATH = "app/db"
# todo
# -- 1. Create data base initilization # done
# -- 2. create database collections # done
# -- 3. get recommendations by itemID # done
# -- 4. get reccomendations by text 
# -- 5. update empeddings by id
# -- 6. Delete embeddings by id
# -- 7. Delete all embeddings
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
        self.sentence_embedding_function = embedding_functions.all_MiniLM_L6_v2

    def create_collection(self, collection_name:str , meta:Dict = None) ->None:
        """
        Creates a new collection in the database.

        Args:
            collection_name (str): The name of the collection to be created.
            meta (Dict, optional): Additional metadata for the collection. Defaults to None.

        Returns:
            None

        Raises:
            chromadb.exceptions.CollectionAlreadyExistsError: If a collection with the same name already exists.
        """
        try:
            self.client.create_collection(collection_name , metadata=meta , embedding_function=self.sentence_embedding_function)
            print(f"Collection '{collection_name}' created successfully!")
        except chromadb.exceptions.CollectionAlreadyExistsError:
            print(f"Collection '{collection_name}' already exists.")

    def get_recommendations_by_id(self, collection_name:str , k_recommendations:int, item_id:str) -> List[str] | None :
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
            collection = self.client.get(collection_name)
            item = collection.get(ids=[item_id])
            if not item:
                return None
            embedding = item["embeddings"][0]
            results = collection.query(
                    query_embeddings=[embedding],  
                    k=k_recommendations, 
                    distance="cosine")
            items_ids = results["ids"]
            return items_ids
        except chromadb.exceptions.CollectionNotFoundError:
            print(f"Collection '{collection_name}' not found.")
        except Exception as e:
            print(e)
