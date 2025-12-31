import cocoindex

from qdrant_client import QdrantClient

class QdrantProvider:
    def __init__(self, url):
        self.url = url
        self._connection_name = "qdrant_connection"
        self._connection = None

    def get_connection(self):
        """get connection object for use in Qdrant target"""
        if self._connection is None:
            self._connection = cocoindex.add_auth_entry(
                self._connection_name,
                cocoindex.targets.QdrantConnection(grpc_url=self.url)
            )
        return self._connection

    def get_client(self):
        """get database client information"""
        return QdrantClient(url=self.url, prefer_grpc=True)
    
    def get_target(self, collection: str):
        """get database collection information"""
        return cocoindex.targets.Qdrant(
            connection=self.get_connection(),
            collection_name=collection
        )
        
