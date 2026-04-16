# mlflow_tracker.py
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import mlflow.langchain
import mlflow.pyfunc
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
import pandas as pd

class RAGModelPromoter:
    def __init__(self, model_name: str = "rag-pkd-pdf-model"):
        self.client = MlflowClient()
        self.model_name = model_name
        self.promotion_thresholds = {
            'accuracy': 0.7,
            'avg_response_time': 30.0,
            'min_queries': 3
        }
   
    def evaluate_model_performance(self, accuracy: float, avg_time: float, total_queries: int):
        meets_accuracy = accuracy >= self.promotion_thresholds['accuracy']
        meets_response_time = avg_time <= self.promotion_thresholds['avg_response_time']
        meets_volume = total_queries >= self.promotion_thresholds['min_queries']
       
        promotion_ready = meets_accuracy and meets_response_time and meets_volume
       
        return {
            'promotion_ready': promotion_ready,
            'criteria_met': {
                'accuracy': meets_accuracy,
                'response_time': meets_response_time,
                'volume': meets_volume
            },
            'score': accuracy * 0.6 + (1 - min(avg_time/60, 1)) * 0.4
        }
   
    def auto_promote_model(self, version: str, accuracy: float, avg_time: float, total_queries: int):
        evaluation = self.evaluate_model_performance(accuracy, avg_time, total_queries)
       
        mlflow.log_params({
            'promotion_threshold_accuracy': self.promotion_thresholds['accuracy'],
            'promotion_threshold_response_time': self.promotion_thresholds['avg_response_time'],
            'promotion_threshold_min_queries': self.promotion_thresholds['min_queries']
        })
       
        mlflow.log_metrics({
            'promotion_score': evaluation['score'],
            'meets_accuracy_criteria': int(evaluation['criteria_met']['accuracy']),
            'meets_response_time_criteria': int(evaluation['criteria_met']['response_time']),
            'meets_volume_criteria': int(evaluation['criteria_met']['volume'])
        })
       
        if evaluation['promotion_ready']:
            try:
                current_prod = self.get_current_production_version()
                if current_prod:
                    self.client.transition_model_version_stage(
                        name=self.model_name,
                        version=current_prod.version,
                        stage="Archived"
                    )
                    print(f"📦 Zarchiwizowano poprzednią wersję production: v{current_prod.version}")
               
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=version,
                    stage="Production"
                )
               
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=version,
                    key="promotion_type",
                    value="auto_promoted"
                )
               
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=version,
                    key="promotion_date",
                    value=datetime.now().isoformat()
                )
               
                print(f"🚀 MODEL PROMOWANY DO PRODUCTION: v{version}")
                mlflow.log_metric("promotion_status", 1)
               
            except Exception as e:
                print(f"❌ Błąd podczas promocji: {e}")
                mlflow.log_metric("promotion_status", -1)
        else:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage="Staging"
            )
            print(f"⚠️ MODEL W STAGING: v{version}")
            mlflow.log_metric("promotion_status", 0)
   
    def get_current_production_version(self):
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            for v in versions:
                if v.current_stage == "Production":
                    return v
            return None
        except Exception as e:
            print(f"Błąd przy pobieraniu production version: {e}")
            return None


class MLflowTracker:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        self.promoter = RAGModelPromoter()
   
    def start_run(self, run_name=None):
        if not run_name:
            run_name = f"RAG-PKD-PDF-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return mlflow.start_run(run_name=run_name)
   
    def log_params(self, params):
        """Loguj parametry do aktywnego runa"""
        mlflow.log_params(params)
   
    def log_training_params(self, model_params):
        """Alias dla log_params - dla kompatybilności"""
        self.log_params(model_params)
   
    def log_metric(self, key, value):
        """Loguj pojedynczą metrykę do aktywnego runa"""
        mlflow.log_metric(key, value)
   
    def log_metrics(self, metrics):
        """Loguj wiele metryk do aktywnego runa"""
        mlflow.log_metrics(metrics)
   
    def log_text(self, text, artifact_file):
        """Loguj tekst jako artefakt do aktywnego runa"""
        mlflow.log_text(text, artifact_file)
   
    def register_model(self, model, model_name="rag-pkd-pdf-model"):
        model_info = mlflow.pyfunc.log_model(
            python_model=model,
            name="rag_pkd_model",
            registered_model_name=model_name
        )
        return model_info

class RAGWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, ollama_url, model_name, temperature):
        # Zapisujemy tylko parametry potrzebne do odtworzenia modelu
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.temperature = temperature

    def load_context(self, context):
        # Inicjalizujemy model LLM dopiero w momencie ładowania modelu przez MLflow
        from langchain_ollama import OllamaLLM
        self.llm_model = OllamaLLM(
            base_url=self.ollama_url, 
            model=self.model_name, 
            temperature=self.temperature
        )

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            prompt = model_input.iloc[0, 0]
        else:
            prompt = model_input
           
        return self.llm_model.invoke(prompt)
