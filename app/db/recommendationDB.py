import chromadb
from chromadb.utils import embedding_functions  # Import embedding functions
PATH = "app/db"
# todo
# -- 1. Create data base initilization
# -- 2. create database collections
# -- 3. get recommendations by itemID
# -- 4. get reccomendations by text 
# -- 5. update empeddings by id
# -- 6. Delete embeddings by id
# -- 7. Delete all embeddings
class RecommendationDB():
    def __int__(self):
        self.client = chromadb.PersistentClient(path="app/db")
        self.sentence_embedding_function = embedding_functions.all_MiniLM_L6_v2
    def create_collection(self, collection_name , meta=None):
        try:
            self.client.create_collection(collection_name , metadata=meta , embedding_function=self.sentence_embedding_function)
            print(f"Collection '{collection_name}' created successfully!")
        except chromadb.exceptions.CollectionAlreadyExistsError:
            print(f"Collection '{collection_name}' already exists.")
    def get_recommendations_by_id(self, collection_name , item_id):
        try:
            recommendations = self.client.get_recommendations_by_id(collection_name , item_id)
            return recommendations
        except chromadb.exceptions.CollectionNotFoundError:
            print(f"Collection '{collection_name}' not found.")
