"""
Metryki z Ragas:
- `Faithfulness` - Sprawdza, czy odpowiedź modelu opiera się tylko na dostarczonym kontekście. Wykrywa haucynacje
"""

from abc import ABC, abstractmethod
from ragas.metrics import Faithfulness as RagasFaithfulness, AnswerCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample

class RagasEvaluator(ABC):
    def __init__(self, llm, embeddings=None):
        self.wrapped_llm = LangchainLLMWrapper(llm)
        self.wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings) if embeddings else None
        self.score: Optional[float] = None

    @abstractmethod
    def evaluate(self, **kwargs) -> float:
        pass

class FaithfulnessEvaluator(RagasEvaluator):
    def __init__(self, llm):
        super().__init__(llm)
        self.metric = RagasFaithfulness(llm=self.wrapped_llm)

    def evaluate(self, query: str, response: str, contexts: list[str]) -> float:
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=contexts 
        )
        
        self.score = self.metric.single_turn_score(sample)
        return float(self.score)

class CorrectnessEvaluator(RagasEvaluator):
    def __init__(self, llm, embeddings):
        super().__init__(llm, embeddings)
        self.metric = AnswerCorrectness(
            llm=self.wrapped_llm,
            embeddings=self.wrapped_embeddings
        )

    def evaluate(self, query: str, response: str, reference: str) -> float:
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            reference=reference
        )
        self.score = self.metric.single_turn_score(sample)
        return float(self.score)

class RagasMetricsReport:
    def __init__(self, faithfulness: FaithfulnessEvaluator):
        self.faithfulness = faithfulness
    
    def metrics_report(self, query: str, response: str, contexts: list[str]) -> dict[str, float]:
        faith = self.faithfulness.evaluate(query, response, contexts)

        return {
            "faithfulness": round(float(faith), 3)}