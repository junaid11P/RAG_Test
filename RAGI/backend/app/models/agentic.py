from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum


class QueryType(str, Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"


class QueryAspect(str, Enum):
    DEFINITION = "definition"
    USE_CASES = "use_cases"
    RISKS = "risks"
    BENEFITS = "benefits"
    BEST_PRACTICES = "best_practices"
    LIMITATIONS = "limitations"


class PlanStep(BaseModel):
    step_id: int
    query: str
    rationale: str
    aspect: Optional[QueryAspect] = None


class AgenticPlan(BaseModel):
    query_type: QueryType
    steps: List[PlanStep]
    required_aspects: List[QueryAspect] = []


class ConfidenceDetails(BaseModel):
    context_similarity: float = 0.0
    evidence_coverage: float = 0.0
    answer_alignment: float = 0.0
    completeness_score: float = 0.0
    agreement_score: float = 0.0


class ContextChunk(BaseModel):
    text: str
    semantic_score: float
    temporal_weight: float
    confidence_score: float
    timestamp: Optional[str] = None