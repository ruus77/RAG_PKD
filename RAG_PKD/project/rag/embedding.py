import pandas as pd
from sentence_transformers import CrossEncoder
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from data_loader import DataLoader
from config import (PDF_FILE,
                    EXCEL_FILE)

EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
CROSSENCOER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


loader = DataLoader(pdf_path=PDF_FILE,
                    excel_path=EXCEL_FILE)
data = loader.load_data()
documents = []

for idx, row in data.iterrows():
    content = f"{row['pkd_code']} | {row['full_text']}"
    metadata = {"pkd_code": row['pkd_code']}
    documents.append(Document(page_content=content, metadata=metadata))

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

db = FAISS.from_documents(documents=documents, embedding=embeddings)
reranker = CrossEncoder(CROSSENCOER_MODEL)

query="Dzień dobry, proszę o wskazanie właściwego, pięcioznakowego kodu PKD dla działalności gospodarczej polegającej na świadczeniu usług specjalistycznych w sektorze handlu ryżem. Zależy mi na kodzie, który najlepiej oddaje charakter przeważającej działalności zgodnie z aktualną klasyfikacją."
docs_result = db.similarity_search(query, k=10)

pairs = [[query, doc.page_content] for doc in docs_result]
scores = reranker.predict(pairs)

reranked_results = [doc for _, doc in sorted(zip(scores, docs_result), key=lambda x: x[0], reverse=True)]

for i in range(3):
    print(reranked_results[i].metadata)