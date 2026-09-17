"""向量库模块，chromadb + bge embedding"""
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from typing import List, Dict


class ChromaVectorStore:
    def __init__(self, collection_name: str = "paper_collection"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=OllamaEmbeddingFunction(model_name="bge-m3")
        )

    def add_chunks(self, chunks: List[Dict], source: str):
        """批量存入向量库"""
        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [{"source": source, "chunk_id": c["chunk_id"]} for c in chunks]
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def similarity_search_with_meta(self, query: str, k: int = 5) -> List[Dict]:
        """
        向量检索，补齐混合检索需要的接口！
        返回格式：[{"chunk_id":"xxx","text":"xxx","source":"xxx"},...]
        """
        res = self.collection.query(query_texts=[query], n_results=k)
        out = []
        for idx in range(len(res["ids"][0])):
            meta = res["metadatas"][0][idx]
            out.append({
                "chunk_id": meta["chunk_id"],
                "text": res["documents"][0][idx],
                "source": meta["source"]
            })
        return out