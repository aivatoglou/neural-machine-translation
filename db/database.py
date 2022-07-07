import pymongo


class MongoDatabase(object):
    def __init__(
        self,
        username,
        password,
        cluster_name,
        database_name,
        collection_documents,
        collection_users,
        mongo_url,
    ):

        self.client = pymongo.MongoClient(
            f"mongodb+srv://{username}:{password}@{cluster_name}.{mongo_url}/?retryWrites=true&w=majority"
        )
        self.database = self.client[database_name]
        self.collection_documents = self.database[collection_documents]
        self.collection_users = self.database[collection_users]

    def insert_one(self, data, collection=None):
        return self.collection_documents.insert_one(data)

    def delete_one(self, data, collection=None):
        return self.collection_documents.delete_one(data)

    def delete_all(self, collection=None):
        return self.collection_documents.delete_many({})

    def find_one(self, query=None, collection=None):
        return self.collection_documents.find_one(query)

    def find_all(self, collection=None):
        return self.collection_documents.find()

    def register_user(self, data, collection=None):
        return self.collection_users.insert_one(data)

    def find_user(self, data, collection=None):
        return self.collection_users.find(data)

    def get_collection_names(self):
        return self.database.list_collection_names()
