from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from data_loader import PKDDataImporter
from config import OLLAMA_URL, MODEL_NAME, EMBEDDING_MODEL, CROSSENCOER_MODEL, TOP_K, DB_FAISS_PATH, PDK_TEST, TEMPERATURE
from embedding import EmbeddingProcessor
import torch

device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
print(device)


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
                temperature=TEMPERATURE)

importer = PKDDataImporter()

loaders = importer.get_loaders()
#: "43.34.Z",
    
#query = "Jaki będzie kod PKD dla działalności stewardesy"
processor = EmbeddingProcessor(embedding_model=embeddings,
                                crossencoder_model=cross_encoder,
                                top_k=TOP_K,
                                documents=loaders,
                                db_path=DB_FAISS_PATH)

processor.create_or_load_db()

COUNTER = 5
for query in list(PDK_TEST.keys())[22:]:
    print(f"Zapytanie: {PDK_TEST.get(query)}")
    scored_docs = processor.get_reranked_embeds(query=query, counter=COUNTER)
    
    for score, doc in scored_docs:
        cos_score = processor.get_cos_sim(query=query, doc=doc.page_content)
        print(f"Reranker Score: {score:.4f} | Cos score: {round(cos_score.item(), 3)} | Code: {doc.metadata}")
        
    context = "\n".join([f"Kod: {d[1].metadata.get('code')} - Opis: {d[1].page_content}" for d in scored_docs[:COUNTER]])

    prompt = f"""
    Jesteś ekspertem ds. polskiej klasyfikacji statystycznej, specjalizującym się w systemie PKD 2025. Twoim celem jest precyzyjne przypisanie działalności do kodów na podstawie dostarczonego kontekstu.

    Kontekst: {context}

    Zidentyfikuj 3 najbardziej prawdopodobne kody PKD 2025 dla zapytania: "{query}"

    1. Analizuj kontekst pod kątem przeważającego charakteru działalności oraz niszowych powiązań.
    2. Wybierz trzy kody, które najlepiej oddają zakres zapytania, szeregując je od najbardziej trafnego.
    3. Jeśli kontekst zawiera bezpośrednie dopasowanie, musi ono znaleźć się na pierwszym miejscu.
    4. Dla każdego kodu przygotuj krótkie uzasadnienie merytoryczne na podstawie dostarczonego kontekstu.
    5. W przypadku całkowitego braku informacji w kontekście dla wszystkich opcji, odpowiedz: "Nie znaleziono w bazie PKD 2025".

    Zwróć odpowiedź w poniższym formacie dla każdego z 3 kodów:

    ### 1. [Kod PKD (zastąp twoim kodem) - Najbardziej prawdopodobny]
    - **Uzasadnienie**: [Wyjaśnienie dopasowania do "{query}"]
    - **Pewność**: [Wysoka / Średnia / Niska]

    ### 2. [Kod PKD (zastąp twoim kodem) - Alternatywa 1]
    - **Uzasadnienie**: [Wyjaśnienie, dlaczego ten kod jest istotny jako uzupełnienie lub alternatywa]
    - **Pewność**: [Wysoka / Średnia / Niska]
    """
    response = llm.invoke(prompt)
    print(f"ODPOWIEDZ MODELU: \n{response}")
