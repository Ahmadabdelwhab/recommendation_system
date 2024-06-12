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
        self.sentence_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction("all-mpnet-base-v2")
    def truncate_string(self, sentence:str , length:int) -> str:
        """
        Validates the given sentence.

        Args:
            sentence (str): The sentence to validate.

        Returns:
            bool: True if the sentence is valid, False otherwise.
        """
    
        if len(sentence.split()) > length:
            return sentence[:length]        
        
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
            
            if not item["ids"]:
                print("no item found with the given id")
                return None
            embedding = item["embeddings"]
            results = collection.query(
                    query_embeddings=embedding,  
                    n_results=k_recommendations)
            items_ids = results["ids"][0][1:]
            return items_ids
        except Exception as e:
            print(e)
    def get_recommednations_by_text(self, collection_name:str, text:str, k_recommendations:int) -> List[str] | None:
        """
        Retrieves recommendations based on the given text.

        Args:
            collection_name (str): The name of the collection to query.
            k_recommendations (int): The number of recommendations to retrieve.
            text (str): The text to get recommendations for.

        Returns:
            List[str] | None: A list of recommended item IDs, or None if no recommendations are found.
        """
        print(text)
        try:
            collection = self.client.get_collection(collection_name , embedding_function=self.sentence_embedding_function)
            results = collection.query(
                    query_texts=[text],  
                    n_results=k_recommendations)
            print(results)
            items_ids = results["ids"][0]
            return items_ids
        except Exception as e:
            print(e)
    def add_item(self , collection_name:str , item:Dict[str,str]) ->None:
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
            except Exception as e:
                print(e)
