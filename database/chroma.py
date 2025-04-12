import chromadb
from chromadb.config import Settings

# Configurar o cliente
client = chromadb.PersistentClient(path="./database/chroma_db")

collection_name = "images_embeddings"
collection = client.get_or_create_collection(name=collection_name)

def add_embedding(id: str, embedding: list[float], metadata: dict = None):
    """
    Adiciona um embedding no ChromaDB.
    """
    collection.add(
        embeddings=[embedding], # vetor de caracteristicas da imagens que vamos utilizar para busca exemplo [[0.1, 0.2, 0.3, 0.4]]
        documents=[id],  # guarda texto/string para utilizar deppos para consulta - exemplo nome da imagem.jpg
        metadatas=[metadata or {}], # dict para guardar informações adicionais ou utilizar para filtrar a consulta através destes dados
        ids=[id], # precisa ser único e usado para consulta. ideia é que seja o nome da imagem.jpg
    )

def query_embedding(embedding: list[float], n_results: int = 5):
    """
    Consulta os embeddings mais próximos.
    """
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )
    return results

def persist():
    """
    Persiste as mudanças no disco.
    """
    client.persist()
