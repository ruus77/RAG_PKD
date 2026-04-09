# config.py
import os
from pathlib import Path
# -------------------------
# KONFIGURACJA MLFLOW
# -------------------------
# Wybierz: True = Localhost | False = Onyxia Remote
USE_LOCAL_MLFLOW = False

if USE_LOCAL_MLFLOW:
    # KONFIGURACJA LOKALNA - uruchomić: mlflow ui --host localhost --port 5000
    MLFLOW_TRACKING_URI = "http://localhost:5000"
    MLFLOW_EXPERIMENT = "RAG-PKD-PDF-Local"
else:
    # KONFIGURACJA ZDALNA (Onyxia)
    os.environ["MLFLOW_TRACKING_TOKEN"] = "u9n88zc4qpr9k4yegcyn"
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "user-russ"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "u9n88zc4qpr9k4yegcyn"
    MLFLOW_TRACKING_URI = "https://user-russ-mlflow.user.lab.sspcloud.fr/"
    MLFLOW_EXPERIMENT = "RAG-PKD-PDF-Belik-Onyxia"

# -------------------------
# KONFIGURACJA OLLAMA
# -------------------------

BASE_DIR = Path("/home/onyxia/work/RAG_PKD/RAG_PKD/")
RESOURCE_DIR = BASE_DIR / "resources"


EXCEL_FILE = RESOURCE_DIR / "StrukturaPKD2025.xls"
PDF_FILE = RESOURCE_DIR / "KlasyfikacjaPKD2025.pdf"
DB_FAISS_PATH = RESOURCE_DIR / "db_faiss"




EMBEDDING_MODEL = 'sdadas/stella-pl-retrieval-8k'
CROSSENCOER_MODEL = 'sdadas/polish-reranker-roberta-v3'

OLLAMA_URL = "http://localhost:11434"

MODEL_NAME = "SpeakLeash/bielik-11b-v2.3-instruct:Q8_0"

#
# ollama run SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0
# "SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M"

# -------------------------
# ŚCIEŻKI PLIKÓW - POPRAWIONE
# -------------------------

TOP_K = 50
RERANK_TOP_N = 2000
TEMPERATURE = 0.1

# -------------------------
# TEST QUERIES
# -------------------------
TEST_QUERIES = [
    "Jaki będzie kod PKD dla montażu balii kąpielowych?",
    "Jaki będzie kod PKD dla montażu żaluzji?",
    "Jaki będzie kod PKD dla montażu folii zacieniających i antywłamaniowych na szybę?",
    "Jaki będzie kod dla montażu nagrobków z gotowych elementów?",
    "Do jakiego kodu można przypisać sprzedaż suplementów diety?",
    "Który kod PKD opisuje sprzedaż waty cukrowej i popcornu na stoisku podczas imprezy masowej?",
    "Jaki będzie kod PKD dla windykacji ?",
    "Jaki będzie kod PKD dla szkoleń BHP?",
    "Jaki będzie kod PKD dla sprzedaży proszku do prania w hurcie",
    "Jaki będzie kod PKD dla zabiegów dla koni po zawodach np. masaże",
    "Jaki będzie kod PKD dla działalności stewardesy",
]

CORRECT_PKD = {
    "Jaki będzie kod PKD dla montażu balii kąpielowych?": "43.22.Z",
    "Jaki będzie kod PKD dla montażu żaluzji?": "43.32.Z",
    "Jaki będzie kod PKD dla montażu folii zacieniających i antywłamaniowych na szybę?": "43.34.Z",
    "Jaki będzie kod dla montażu nagrobków z gotowych elementów?": "43.99.Z",
    "Do jakiego kodu można przypisać sprzedaż suplementów diety?": "46.39.Z",
    "Który kod PKD opisuje sprzedaż waty cukrowej i popcornu na stoisku podczas imprezy masowej?": "56.12.Z", 
    "Jaki będzie kod PKD dla windykacji ?": "82.91.Z",
    "Jaki będzie kod PKD dla szkoleń BHP?": "85.59.D",
    "Jaki będzie kod PKD dla sprzedaży proszku do prania w hurcie": "46.44.Z",
    "Jaki będzie kod PKD dla zabiegów dla koni po zawodach np. masaże": "93.19.Z",
    "Jaki będzie kod PKD dla działalności stewardesy": "52.23.Z"
}

# -------------------------
# DANE DO TRENOWANIA
# -------------------------
Trening_QUERIES_aug = [
    "Jaki kod PKD odpowiada instalacji balii ogrodowych do kąpieli?",
    "Który kod PKD obejmuje montowanie balii kąpielowych?",
    "Pod jakim PKD można prowadzić montaż balii kąpielowych?",
    "Jaki kod PKD dotyczy instalowania żaluzji okiennych?",
    "Pod jaki kod PKD podlega montaż żaluzji?",
    "Które PKD opisuje usługę zakładania żaluzji?",
    "Jaki kod PKD obejmuje montowanie folii przeciwsłonecznych na szybach?",
    "Pod jaki kod PKD podlega instalacja folii antywłamaniowej na oknach?",
    "Który kod PKD dotyczy zakładania folii zaciemniających na szyby?",
    "Jaki kod PKD należy wybrać dla montażu nagrobków z prefabrykatów?",
    "Pod jakim PKD można wykonywać instalację nagrobków z gotowych elementów?",
    "Który kod PKD obejmuje składanie nagrobków z przygotowanych części?",
    "Jaki kod PKD odpowiada sprzedaży suplementów diety?",
    "Pod jaki kod PKD podlega handel suplementami diety?",
    "Które PKD obejmuje dystrybucję suplementów diety?",
    "Jaki kod PKD dotyczy sprzedaży waty cukrowej i popcornu na festynach?",
    "Pod jaki kod PKD podlega sprzedaż popcornu i waty cukrowej na stoisku?",
    "Który kod PKD obejmuje sprzedaż przekąsek na imprezach plenerowych?",
    "Jaki kod PKD obejmuje działalność windykacyjną?",
    "Pod jaki kod PKD podlega odzyskiwanie należności?",
    "Który kod PKD opisuje usługi windykacji długów?",
    "Jaki kod PKD dotyczy prowadzenia szkoleń z zakresu BHP?",
    "Pod jakim PKD można organizować kursy BHP?",
    "Który kod PKD obejmuje szkolenia z bezpieczeństwa i higieny pracy?",
    "Jaki kod PKD odpowiada hurtowej sprzedaży proszków do prania?",
    "Pod jaki kod PKD podlega handel hurtowy detergentami do prania?",
    "Który kod PKD obejmuje sprzedaż hurtową środków do prania?",
    "Jaki kod PKD dotyczy masaży i zabiegów regeneracyjnych dla koni sportowych?",
    "Pod jaki kod PKD podlega opieka regeneracyjna dla koni po zawodach?",
    "Który kod PKD obejmuje usługi masażu dla koni?",
    "Jaki kod PKD obejmuje pracę stewardesy?",
    "Pod jaki kod PKD podlega działalność stewardesy lotniczej?",
    "Który kod PKD opisuje zawód stewardesy?"
]

PDK_TEST = {
    "Jaki kod PKD odpowiada instalacji balii ogrodowych do kąpieli?": "43.22.Z",
    "Który kod PKD obejmuje montowanie balii kąpielowych?": "43.22.Z",
    "Pod jakim PKD można prowadzić montaż balii kąpielowych?": "43.22.Z",
    "Jaki kod PKD dotyczy instalowania żaluzji okiennych?": "43.32.Z",
    "Pod jaki kod PKD podlega montaż żaluzji?": "43.32.Z",
    "Które PKD opisuje usługę zakładania żaluzji?": "43.32.Z",
    "Jaki kod PKD obejmuje montowanie folii przeciwsłonecznych na szybach?": "43.34.Z",
    "Pod jaki kod PKD podlega instalacja folii antywłamaniowej na oknach?": "43.34.Z",
    "Który kod PKD dotyczy zakładania folii zaciemniających na szyby?": "43.34.Z",
    "Jaki kod PKD należy wybrać dla montażu nagrobków z prefabrykatów?": "43.99.Z",
    "Pod jakim PKD można wykonywać instalację nagrobków z gotowych elementów?": "43.99.Z",
    "Który kod PKD obejmuje składanie nagrobków z przygotowanych części?": "43.99.Z",
    "Jaki kod PKD odpowiada sprzedaży suplementów diety?": "46.39.Z",
    "Pod jaki kod PKD podlega handel suplementami diety?": "46.39.Z",
    "Które PKD obejmuje dystrybucję suplementów diety?": "46.39.Z",
    "Jaki kod PKD dotyczy sprzedaży waty cukrowej i popcornu na festynach?": "56.12.Z",
    "Pod jaki kod PKD podlega sprzedaż popcornu i waty cukrowej na stoisku?": "56.12.Z",
    "Który kod PKD obejmuje sprzedaż przekąsek na imprezach plenerowych?": "56.12.Z",
    "Jaki kod PKD obejmuje działalność windykacyjną?": "82.91.Z",
    "Pod jaki kod PKD podlega odzyskiwanie należności?": "82.91.Z",
    "Który kod PKD opisuje usługi windykacji długów?": "82.91.Z",
    "Jaki kod PKD dotyczy prowadzenia szkoleń z zakresu BHP?": "85.59.D",
    "Pod jakim PKD można organizować kursy BHP?": "85.59.D",
    "Który kod PKD obejmuje szkolenia z bezpieczeństwa i higieny pracy?": "85.59.D",
    "Jaki kod PKD odpowiada hurtowej sprzedaży proszków do prania?": "46.44.Z",
    "Pod jaki kod PKD podlega handel hurtowy detergentami do prania?": "46.44.Z",
    "Który kod PKD obejmuje sprzedaż hurtową środków do prania?": "46.44.Z",
    "Jaki kod PKD dotyczy masaży i zabiegów regeneracyjnych dla koni sportowych?": "93.19.Z",
    "Pod jaki kod PKD podlega opieka regeneracyjna dla koni po zawodach?": "93.19.Z",
    "Który kod PKD obejmuje usługi masażu dla koni?": "93.19.Z",
    "Jaki kod PKD obejmuje pracę stewardesy?": "52.23.Z",
    "Pod jaki kod PKD podlega działalność stewardesy lotniczej?": "52.23.Z",
    "Który kod PKD opisuje zawód stewardesy?": "52.23.Z"
}