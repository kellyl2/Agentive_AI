# Pydantic-style schema (conceptual; adapt to your LangGraph version)
### code source: https://chatgpt.com/share/6971bd32-f310-800c-bda5-b1ff1baaed8d

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class Scope(BaseModel):
    time_window: Dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    region_code: str             # e.g., "ALL", "US-NE"
    entity_types: List[Literal["class", "equipment", "accessory"]]
    channels: List[str] = []     # e.g., ["app", "web", "email"]
    min_volume_thresholds: Dict[str, float] = {}  # e.g., {"sales_units": 50, "class_starts": 500}
    include_watchlist: bool = True

class MetricDef(BaseModel):
    metric_name: str
    metric_id: Optional[str] = None
    definition: Optional[str] = None
    grain: Optional[str] = None
    allowed_dimensions: List[str] = []
    source_ref: Optional[str] = None  # link/id for citations

class FreshnessItem(BaseModel):
    dataset_or_metric: str
    last_updated_ts: Optional[str] = None
    freshness_status: Literal["OK", "WARN", "BLOCK"]
    anomaly_flag: bool = False
    notes: Optional[str] = None

class FreshnessReport(BaseModel):
    overall_status: Literal["OK", "WARN", "BLOCK"]
    items: List[FreshnessItem] = []
    required_caveats: List[str] = []

class SignalTableRef(BaseModel):
    name: str
    rows: List[Dict[str, Any]] = []   # normalized rows for the signal
    schema: Dict[str, str] = {}       # column -> type
    source: Optional[str] = None      # tool/dataset identifier
    as_of_ts: Optional[str] = None

class CandidateFeature(BaseModel):
    entity_type: Literal["class", "equipment", "accessory"]
    entity_id: str
    entity_name: Optional[str] = None

    # core trend features (examples; add/remove as needed)
    sales_velocity_delta: Optional[float] = None
    completion_rate_delta: Optional[float] = None
    search_momentum_delta: Optional[float] = None
    inventory_turnover: Optional[float] = None
    atp_qty: Optional[int] = None
    seasonality_index: Optional[float] = None

    # scoring outputs
    score: Optional[float] = None
    confidence: Optional[float] = None
    drivers: List[str] = []  # human-readable driver strings for explainability

    # policy flags
    eligible: bool = True
    filtered_reason: Optional[str] = None

class RankedTrends(BaseModel):
    classes: List[CandidateFeature] = []
    equipment: List[CandidateFeature] = []
    accessories: List[CandidateFeature] = []
    watchlist: List[CandidateFeature] = []

class Recommendation(BaseModel):
    entity_type: Literal["class", "equipment", "accessory"]
    entity_id: str
    recommendation_type: str  # e.g., "homepage_feature", "email_spotlight", "bundle", "push_notification"
    rationale: str
    constraints: List[str] = []      # inventory/policy constraints applied
    expected_impact: Optional[str] = None

class OutputPackage(BaseModel):
    executive_summary: List[str] = []
    narrative: str = ""
    recommendations: List[Recommendation] = []
    trend_tables: Dict[str, List[Dict[str, Any]]] = {}
    caveats: List[str] = []
    citations: List[str] = []        # metric/source refs included

class AgentState(BaseModel):
    # Inputs
    user_request: str
    scope: Optional[Scope] = None

    # Governance & controls
    approved_metrics: List[MetricDef] = []
    freshness_report: Optional[FreshnessReport] = None

    # Data payloads
    signals: List[SignalTableRef] = []
    candidates: List[CandidateFeature] = []
    ranked_trends: Optional[RankedTrends] = None

    # Outputs
    output: Optional[OutputPackage] = None

    # Operational telemetry
    audit_log: List[Dict[str, Any]] = []
    errors: List[str] = []
