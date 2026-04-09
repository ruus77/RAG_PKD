from langchain_community.vectorstores import FAISS
from sentence_transformers.util import cos_sim
import os


class EmbeddingProcessor:
    def __init__(self, embedding_model, crossencoder_model, top_k, documents, db_path):
        self.embedding_model = embedding_model
        self.crossencoder_model = crossencoder_model
        self.top_k = top_k
        self.documents = documents
        self.db_path = db_path
        self.db = None

    def create_or_load_db(self):
        
        if os.path.exists(self.db_path):
            self.db = FAISS.load_local(
                    self.db_path, 
                    self.embedding_model, 
                    allow_dangerous_deserialization=True)

        else:
            self.db = FAISS.from_documents(documents=self.documents, embedding=self.embedding_model)
            self.db.save_local(self.db_path)
        return self.db
    
    def get_cos_sim(self, query, doc):
        query_embed = self.embedding_model.embed_query(query)
        doc_embed = self.embedding_model.embed_query(doc)
        return cos_sim(query_embed, doc_embed)

    def _get_initial_candidates(self, query):
        
        if not self.db:
            self.create_or_load_db()
        return self.db.similarity_search(query, k=self.top_k)


    def get_reranked_embeds(self, 
                            query:str, 
                            counter:int=5) -> list[tuple]:

        initial_docs = self._get_initial_candidates(query)
        pairs = [[query, doc.page_content] for doc in initial_docs]
        
        scores = self.crossencoder_model.predict(pairs)
        scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
        
        return scored_docs[:counter]