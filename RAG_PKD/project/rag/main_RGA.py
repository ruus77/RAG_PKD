from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from data_loader import PKDDataImporter
from config import (OLLAMA_URL, MODEL_NAME, EMBEDDING_MODEL, CROSSENCOER_MODEL, 
                    TOP_K, DB_FAISS_PATH, PDK_TEST, TEMPERATURE)
from embedding import EmbeddingProcessor
import torch
from mlflow_tracker import MLflowTracker, RAGWrapper
import time
import warnings
import logging

warnings.filterwarnings(action="ignore")
logging.getLogger("langchain").setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Current device: {device}")

tracker = MLflowTracker()

cross_encoder = CrossEncoder(
    CROSSENCOER_MODEL,
    activation_fn=torch.nn.Identity(),
    device=device,
    max_length=1024,
    model_kwargs={"dtype": torch.bfloat16})

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs = {"device" : device},
    encode_kwargs = {'normalize_embeddings': False})

llm = OllamaLLM(base_url=OLLAMA_URL, 
                model=MODEL_NAME, 
                temperature=TEMPERATURE,
                raw=False,
                seed=77)



importer = PKDDataImporter()
loaders = importer.get_loaders()


processor = EmbeddingProcessor(embedding_model=embeddings,
                                crossencoder_model=cross_encoder,
                                top_k=TOP_K,
                                documents=loaders,
                                db_path=DB_FAISS_PATH)

processor.create_or_load_db()
COUNTER = 10

PDK_TEST = {k: PDK_TEST[k] for k in list(PDK_TEST.keys())[:25]}

results = []
faith_scores = []
start_time = time.time()

total_queries = len(PDK_TEST)

# JEDEN BLOK start_run
with tracker.start_run() as run:
    run_id = run.info.run_id
    
    tracker.log_params({
        "model" : MODEL_NAME,
        "embedding_model" : EMBEDDING_MODEL,
        "cross_encoder" : CROSSENCOER_MODEL,
        "temperature" : TEMPERATURE,
        "top_k" : TOP_K,
        "dataset_size" : total_queries
    })

    correct_preds = 0
    for query, expected_code in PDK_TEST.items():
        print(f"\nZapytanie: {expected_code} | {query}")
        start_query = time.time()

        prompt = f"""Jesteś ekspertem ds. Polskiej Klasyfikacji Działalności (PKD). 
            Twoim zadaniem jest przekształcenie potocznego opisu działalności użytkownika na sformalizowane zapytanie, które językowo i strukturalnie odpowiada oficjalnym opisom oraz wyjaśnieniom kodów PKD oraz wzbocaga je synonimammi i szerszym objaśnieniem danej działaności.
            
            Wytyczne dla Eksperta:
            
            1. Język urzędowy: Zastępuj potoczne sformułowania terminologią stosowaną w rozporządzeniu PKD (np. zamiast "sprzedaż przez internet" użyj "sprzedaż detaliczna prowadzona przez domy sprzedaży wysyłkowej lub Internet").
            2. Ekspansja semantyczna: Wzbogać zapytanie o kluczowe synonimy, terminy bliskoznaczne oraz czynności powiązane, które naturalnie występują w strukturze klasyfikacji dla danej branży.
            2. Kontekst operacyjny: Określ istotę czynności – co jest przedmiotem działalności, dla kogo jest przeznaczona (klient indywidualny vs biznesowy) oraz w jakiej formie jest realizowana (produkcja, handel, usługi).
            3. Czystość formatowania: Odpowiedź musi być jednolitym blokiem tekstu. Nie używaj tabulacji, znaków ucieczki (\n, \t), wypunktowań ani dodatkowych komentarzy. Tekst ma być bezpośrednim, opisowym ciągiem znaków gotowym do porównania przez Cross-Encoder.

            Przykład:
            Użytkownik: "Robię strony internetowe i aplikacje na telefony dla firm."
            Ekspert: Działalność związana z projektowaniem, programowaniem i rozwojem oprogramowania oraz systemów informatycznych, włączając w to tworzenie aplikacji mobilnych, projektowanie witryn internetowych oraz doradztwo w zakresie informatyki i technologii komputerowych.

            Zapytanie użytkownika do przetworzenia: {query}
            Przeformułowane zapytanie wraz z zestawem synonimów i opisem działalności:"""
            
        response = llm.invoke(prompt)
        print(response)

        scored_docs = processor.get_reranked_embeds(query=response, counter=COUNTER)
        for score, doc in scored_docs:
            cos_score = processor.get_cos_sim(query=response, doc=doc.page_content)
            print(f"Reranker Score: {score:.4f} | Cos score: {round(cos_score.item(), 3)} | Code: {doc.metadata}")
            if expected_code == str(doc.metadata.get("code")):        
                correct_preds +=1
                print(f"Poprawność: {correct_preds}")
                #print("Brak idealnego dopasowania")
            
        query_duration = time.time() - start_query
        
    total_duration = time.time() - start_time
    avg_response_time = total_duration / total_queries if total_queries > 0 else 0
    accuracy = correct_preds / total_queries if total_queries > 0 else 0
    
    tracker.log_metrics({
        "accuracy": accuracy,
        "avg_response_time": avg_response_time,
        "total_execution_time": total_duration,
        "correct_predictions": correct_preds,
        "total_queries": total_queries
    })
    
    model_info = tracker.register_model(
                RAGWrapper(OLLAMA_URL, MODEL_NAME, TEMPERATURE), 
                model_name=MODEL_NAME
        )
    
    version = model_info.registered_model_version

    tracker.promoter.auto_promote_model(
            version=version,
            accuracy=accuracy,
            avg_time=avg_response_time,
            total_queries=total_queries)