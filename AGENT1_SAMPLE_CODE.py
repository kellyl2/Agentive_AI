"""
Trend & Merchandising Insight Agent (Peloton) — LangGraph Implementation Skeleton
-------------------------------------------------------------------------------
Reference Source: https://chatgpt.com/share/6971bd32-f310-800c-bda5-b1ff1baaed8d

What you get:
- Implementation-ready LangGraph graph wiring (nodes + conditional edges)
- Node function stubs (intake, metric validation, freshness gate, retrieval, scoring, filtering, narrative, validation)
- Tool interfaces (BI semantic layer, Inventory, Search, Seasonality, Freshness/DQ)
- Structured output contract (JSON payload + human narrative)

Notes:
- This is intentionally “production-shaped”: deterministic scoring + policy filtering + LLM narrative.
- You must wire your real tool backends (SQL/semantic layer, inventory API, search analytics, etc.).
- Adjust imports if your LangGraph version differs. Tested structure matches LangGraph 0.2+ patterns.

TODO:
Replace the Dummy* tool classes with real implementations:
- BI semantic layer: Looker/Cube/dbt Semantic Layer API, or a governed SQL service
- Inventory: OMS/WMS API or a curated “ATP” table
- Search analytics: event warehouse aggregates (query clusters, CTR, conversion)
- Seasonality: historical baseline service or precomputed indices
- Freshness: data observability (e.g., Great Expectations results, Monte Carlo, custom checks)
- Provide a real llm in make_default_tools(llm=...) to enable narrative generation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal, Protocol, Tuple
import json
import math
import statistics
from datetime import datetime, timezone, date

# --- LangGraph imports (adjust to your installed version if needed) ---
try:
    from langgraph.graph import StateGraph, START, END
except Exception as e:
    raise ImportError(
        "LangGraph import failed. Ensure `langgraph` is installed and version supports `langgraph.graph`."
    ) from e

# Optional: if you use LangChain chat models for narrative generation
try:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:
    BaseChatModel = Any  # type: ignore


# =============================================================================
# Output Contracts
# =============================================================================

@dataclass
class Recommendation:
    entity_type: Literal["class", "equipment", "accessory"]
    entity_id: str
    entity_name: Optional[str]
    recommendation_type: str   # e.g., "homepage_feature", "email_spotlight", "push_notification", "bundle"
    rationale: str
    constraints: List[str]
    expected_impact: Optional[str] = None


@dataclass
class OutputContract:
    """Structured output to store + return to downstream systems."""
    run_id: str
    generated_at: str
    scope: Dict[str, Any]
    trends: Dict[str, List[Dict[str, Any]]]          # ranked trends tables
    recommendations: List[Recommendation]
    caveats: List[str]
    citations: List[str]
    narrative: str

    def to_json(self) -> str:
        payload = asdict(self)
        # Convert Recommendation dataclasses cleanly
        payload["recommendations"] = [asdict(r) for r in self.recommendations]
        return json.dumps(payload, indent=2, ensure_ascii=False)


# =============================================================================
# Tool Interfaces (Protocols) + Example Stubs
# =============================================================================

class BISemanticLayerTool(Protocol):
    def list_metrics(self) -> List[Dict[str, Any]]:
        """Return governed metrics list, including allowed dims, grain, owner, and a reference id for citations."""
        ...

    def query_metric(
        self,
        metric_name: str,
        start_date: str,
        end_date: str,
        dimensions: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return aggregated metric rows (already governed)."""
        ...


class InventoryTool(Protocol):
    def get_availability(
        self, sku_ids: List[str], region_code: str
    ) -> Dict[str, Dict[str, Any]]:
        """Return ATP/on-hand/turnover/lead-time for each SKU."""
        ...


class SearchAnalyticsTool(Protocol):
    def query_search_momentum(
        self, start_date: str, end_date: str, region_code: str, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """Return search query momentum rows (e.g., query cluster -> delta, CTR)."""
        ...


class SeasonalityTool(Protocol):
    def get_seasonality_index(
        self, entity_type: str, entity_id: str, metric_name: str, start_date: str, end_date: str
    ) -> Dict[str, Any]:
        """Return seasonal baseline / index and expected band."""
        ...


class FreshnessTool(Protocol):
    def check_freshness(
        self, dependencies: List[str]
    ) -> Dict[str, Any]:
        """
        Return freshness status for dependency identifiers.
        Expected:
          {
            "overall_status": "OK|WARN|BLOCK",
            "items": [{"dependency": "...", "status": "...", "last_updated_ts": "...", "notes": "..."}],
            "required_caveats": ["..."]
          }
        """
        ...


# ------------------------------
# Example concrete stub tools
# Replace with real implementations.
# ------------------------------

class DummyBISemanticLayer:
    def __init__(self):
        self._metrics = [
            {
                "metric_name": "sales_velocity_units",
                "grain": "daily",
                "allowed_dimensions": ["region_code", "sku_id", "category"],
                "owner": "Commerce Analytics",
                "source_ref": "metric://sales_velocity_units/v1",
            },
            {
                "metric_name": "class_completion_rate",
                "grain": "daily",
                "allowed_dimensions": ["region_code", "content_id", "modality"],
                "owner": "Content Analytics",
                "source_ref": "metric://class_completion_rate/v1",
            },
        ]

    def list_metrics(self) -> List[Dict[str, Any]]:
        return self._metrics

    def query_metric(
        self,
        metric_name: str,
        start_date: str,
        end_date: str,
        dimensions: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        # Return synthetic aggregated data as placeholder.
        # In production: call semantic layer (dbt semantic, Looker, Cube, etc.)
        if metric_name == "sales_velocity_units":
            return [
                {"sku_id": "SKU-ABC12345", "region_code": filters.get("region_code", "ALL"), "value": 1200, "value_prev": 800},
                {"sku_id": "SKU-XYZ99999", "region_code": filters.get("region_code", "ALL"), "value": 900, "value_prev": 850},
            ]
        if metric_name == "class_completion_rate":
            return [
                {"content_id": "C-101", "region_code": filters.get("region_code", "ALL"), "value": 0.74, "value_prev": 0.62},
                {"content_id": "C-202", "region_code": filters.get("region_code", "ALL"), "value": 0.69, "value_prev": 0.68},
            ]
        return []


class DummyInventoryTool:
    def get_availability(self, sku_ids: List[str], region_code: str) -> Dict[str, Dict[str, Any]]:
        return {
            sku: {
                "atp_qty": 25 if "ABC" in sku else 3,
                "on_hand_qty": 40 if "ABC" in sku else 5,
                "inventory_turnover": 3.2 if "ABC" in sku else 1.1,
                "vendor_lead_time_days_p50": 10 if "ABC" in sku else 35,
            }
            for sku in sku_ids
        }


class DummySearchTool:
    def query_search_momentum(self, start_date: str, end_date: str, region_code: str, limit: int = 200) -> List[Dict[str, Any]]:
        return [
            {"query_cluster": "bike shoes", "delta": 0.42, "ctr_delta": 0.10},
            {"query_cluster": "yoga beginner", "delta": 0.28, "ctr_delta": 0.03},
        ]


class DummySeasonalityTool:
    def get_seasonality_index(self, entity_type: str, entity_id: str, metric_name: str, start_date: str, end_date: str) -> Dict[str, Any]:
        # Simple stub: index > 1 means seasonally expected high
        return {"seasonality_index": 1.05, "expected_band": {"low": 0.9, "high": 1.1}, "source_ref": "seasonality://baseline/v1"}


class DummyFreshnessTool:
    def check_freshness(self, dependencies: List[str]) -> Dict[str, Any]:
        return {
            "overall_status": "OK",
            "items": [{"dependency": d, "status": "OK", "last_updated_ts": datetime.now(timezone.utc).isoformat(), "notes": ""} for d in dependencies],
            "required_caveats": [],
        }


# =============================================================================
# State Shape
# =============================================================================

# We keep state as a plain dict for LangGraph compatibility across versions.
State = Dict[str, Any]


# =============================================================================
# Helpers (deterministic scoring, validation)
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_default_scope(user_request: str) -> Dict[str, Any]:
    # Production: robust parsing; here: defaults
    today = date.today()
    start = (today.replace(day=max(1, today.day - 28))).isoformat()
    end = today.isoformat()
    return {
        "time_window": {"start": start, "end": end},
        "region_code": "ALL",
        "entity_types": ["class", "equipment", "accessory"],
        "channels": ["app", "web", "email"],
        "min_volume_thresholds": {"sales_units": 50, "class_starts": 500},
        "include_watchlist": True,
    }


def zscore(values: List[float]) -> List[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    mu = statistics.mean(values)
    sd = statistics.pstdev(values) or 1e-9
    return [(v - mu) / sd for v in values]


def safe_pct_delta(curr: float, prev: float) -> float:
    if prev == 0:
        return 0.0 if curr == 0 else 1.0
    return (curr - prev) / abs(prev)


# =============================================================================
# Node Implementations (stubs with production-ready structure)
# =============================================================================

def n0_intake_scope(state: State) -> State:
    req = state["user_request"]
    scope = parse_default_scope(req)
    state["scope"] = scope
    state.setdefault("audit_log", []).append({"node": "intake_scope", "ts": utc_now_iso(), "scope": scope})
    return state


def n1_validate_metrics_semantic_layer(state: State) -> State:
    scope = state["scope"]
    bi: BISemanticLayerTool = state["tools"]["bi"]

    governed = bi.list_metrics()
    # For this agent, we need at least these KPIs
    required = ["sales_velocity_units", "class_completion_rate"]  # can expand
    approved = []
    missing = []
    for m in required:
        match = next((x for x in governed if x["metric_name"] == m), None)
        if not match:
            missing.append(m)
        else:
            approved.append(match)

    if missing:
        state.setdefault("errors", []).append(f"Missing governed metrics: {missing}")
        # In production: either raise or route to a clarification node
    state["approved_metrics"] = approved
    state.setdefault("audit_log", []).append({"node": "validate_metrics", "ts": utc_now_iso(), "approved": [m["metric_name"] for m in approved]})
    return state


def n2_check_data_freshness(state: State) -> State:
    freshness: FreshnessTool = state["tools"]["freshness"]
    approved = state.get("approved_metrics", [])

    deps = [m.get("source_ref", m["metric_name"]) for m in approved]
    deps += ["search://analytics", "inventory://availability", "seasonality://baseline"]
    report = freshness.check_freshness(deps)

    state["freshness_report"] = report
    state.setdefault("audit_log", []).append({"node": "check_freshness", "ts": utc_now_iso(), "report": report})
    return state


def freshness_router(state: State) -> str:
    status = (state.get("freshness_report") or {}).get("overall_status", "OK")
    if status == "BLOCK":
        return "n2b_handle_blocked_freshness"
    return "n3_retrieve_analytics_signals"


def n2b_handle_blocked_freshness(state: State) -> State:
    report = state.get("freshness_report", {})
    caveats = report.get("required_caveats", [])
    narrative = (
        "Unable to produce trend insights because one or more critical data dependencies are stale or unhealthy.\n"
        f"Details: {json.dumps(report, indent=2)}"
    )
    out = OutputContract(
        run_id=state["run_id"],
        generated_at=utc_now_iso(),
        scope=state.get("scope", {}),
        trends={"classes": [], "equipment": [], "accessories": [], "watchlist": []},
        recommendations=[],
        caveats=caveats or ["Data freshness gate blocked execution."],
        citations=[],
        narrative=narrative,
    )
    state["output_contract"] = out
    return state


def n3_retrieve_analytics_signals(state: State) -> State:
    scope = state["scope"]
    bi: BISemanticLayerTool = state["tools"]["bi"]
    inv: InventoryTool = state["tools"]["inventory"]
    search: SearchAnalyticsTool = state["tools"]["search"]
    seas: SeasonalityTool = state["tools"]["seasonality"]

    start = scope["time_window"]["start"]
    end = scope["time_window"]["end"]
    region = scope["region_code"]

    # BI metrics
    sales_rows = bi.query_metric(
        metric_name="sales_velocity_units",
        start_date=start,
        end_date=end,
        dimensions={"sku_id": True},
        filters={"region_code": region},
    )
    completion_rows = bi.query_metric(
        metric_name="class_completion_rate",
        start_date=start,
        end_date=end,
        dimensions={"content_id": True},
        filters={"region_code": region},
    )

    # Search momentum (query clusters)
    search_rows = search.query_search_momentum(start, end, region_code=region)

    # Inventory availability for sales SKUs
    sku_ids = [r["sku_id"] for r in sales_rows if "sku_id" in r]
    inventory = inv.get_availability(sku_ids, region_code=region) if sku_ids else {}

    # Seasonality indices for each entity candidate (minimal example)
    seasonality = {}
    for r in sales_rows:
        sku = r.get("sku_id")
        if sku:
            seasonality[( "equipment", sku )] = seas.get_seasonality_index("equipment", sku, "sales_velocity_units", start, end)
    for r in completion_rows:
        cid = r.get("content_id")
        if cid:
            seasonality[( "class", cid )] = seas.get_seasonality_index("class", cid, "class_completion_rate", start, end)

    state["signals"] = {
        "sales_velocity_units": sales_rows,
        "class_completion_rate": completion_rows,
        "search_momentum": search_rows,
        "inventory": inventory,
        "seasonality": seasonality,
    }
    state.setdefault("audit_log", []).append({"node": "retrieve_signals", "ts": utc_now_iso(), "counts": {k: len(v) if isinstance(v, list) else len(v) for k, v in state["signals"].items() if k != "seasonality"}})
    return state


def n4_normalize_and_join_signals(state: State) -> State:
    sig = state["signals"]
    inv = sig["inventory"]
    seas = sig["seasonality"]

    candidates: List[Dict[str, Any]] = []

    # Equipment trends from sales velocity
    for r in sig["sales_velocity_units"]:
        sku = r["sku_id"]
        curr = float(r.get("value", 0))
        prev = float(r.get("value_prev", 0))
        delta = safe_pct_delta(curr, prev)
        inv_row = inv.get(sku, {})
        sea = seas.get(("equipment", sku), {})
        candidates.append({
            "entity_type": "equipment",
            "entity_id": sku,
            "entity_name": sku,
            "sales_velocity_delta": delta,
            "completion_rate_delta": None,
            "search_momentum_delta": None,
            "inventory_turnover": inv_row.get("inventory_turnover"),
            "atp_qty": inv_row.get("atp_qty"),
            "seasonality_index": sea.get("seasonality_index"),
            "drivers": [],
        })

    # Class trends from completion deltas
    for r in sig["class_completion_rate"]:
        cid = r["content_id"]
        curr = float(r.get("value", 0))
        prev = float(r.get("value_prev", 0))
        delta = safe_pct_delta(curr, prev)
        sea = seas.get(("class", cid), {})
        candidates.append({
            "entity_type": "class",
            "entity_id": cid,
            "entity_name": cid,
            "sales_velocity_delta": None,
            "completion_rate_delta": delta,
            "search_momentum_delta": None,
            "inventory_turnover": None,
            "atp_qty": None,
            "seasonality_index": sea.get("seasonality_index"),
            "drivers": [],
        })

    # Accessory trends could come from sales velocity too (if you tag SKUs by category).
    # For now, we leave accessories empty; in production add category dimension and split.

    state["candidates"] = candidates
    state.setdefault("audit_log", []).append({"node": "normalize_join", "ts": utc_now_iso(), "candidate_count": len(candidates)})
    return state


def n5_score_and_rank_trends(state: State) -> State:
    cands = state["candidates"]

    # Build feature arrays for z-scoring
    sales_deltas = [c["sales_velocity_delta"] for c in cands if c["sales_velocity_delta"] is not None]
    comp_deltas = [c["completion_rate_delta"] for c in cands if c["completion_rate_delta"] is not None]

    sales_z = zscore([float(x) for x in sales_deltas]) if sales_deltas else []
    comp_z = zscore([float(x) for x in comp_deltas]) if comp_deltas else []

    # Assign scores
    si = 0
    ci = 0
    for c in cands:
        score = 0.0
        drivers: List[str] = []
        if c["sales_velocity_delta"] is not None:
            z = sales_z[si] if si < len(sales_z) else 0.0
            score += 1.2 * z
            drivers.append(f"Sales velocity delta: {c['sales_velocity_delta']:.2%}")
            si += 1
        if c["completion_rate_delta"] is not None:
            z = comp_z[ci] if ci < len(comp_z) else 0.0
            score += 1.0 * z
            drivers.append(f"Completion rate delta: {c['completion_rate_delta']:.2%}")
            ci += 1

        # Seasonality adjustment (down-weight trends that are fully seasonality-explained)
        sidx = c.get("seasonality_index")
        if sidx is not None:
            # if seasonality is already high, reduce "novelty"
            score -= 0.3 * max(0.0, float(sidx) - 1.0)
            drivers.append(f"Seasonality index: {float(sidx):.2f}")

        # Confidence heuristic: bounded sigmoid of abs(score)
        confidence = 1 / (1 + math.exp(-abs(score)))  # 0.5..1.0
        c["score"] = score
        c["confidence"] = float(confidence)
        c["drivers"] = drivers

    # Rank per entity type
    def top_for(t: str) -> List[Dict[str, Any]]:
        items = [c for c in cands if c["entity_type"] == t]
        return sorted(items, key=lambda x: x.get("score", -9999), reverse=True)

    ranked = {
        "classes": top_for("class"),
        "equipment": top_for("equipment"),
        "accessories": top_for("accessory"),
        "watchlist": [],
    }
    state["ranked_trends"] = ranked
    state.setdefault("audit_log", []).append({"node": "score_rank", "ts": utc_now_iso()})
    return state


def n6_apply_inventory_and_policy_filters(state: State) -> State:
    scope = state["scope"]
    ranked = state["ranked_trends"]
    min_atp = 5  # example threshold; make config-driven

    recs: List[Recommendation] = []
    watchlist: List[Dict[str, Any]] = []

    # Equipment recommendations must respect ATP
    filtered_equipment = []
    for c in ranked["equipment"]:
        atp = c.get("atp_qty")
        if atp is not None and atp < min_atp:
            c["eligible"] = False
            c["filtered_reason"] = f"Low ATP ({atp} < {min_atp})"
            watchlist.append(c)
            continue
        c["eligible"] = True
        filtered_equipment.append(c)

        recs.append(Recommendation(
            entity_type="equipment",
            entity_id=c["entity_id"],
            entity_name=c.get("entity_name"),
            recommendation_type="homepage_feature",
            rationale="; ".join(c.get("drivers", [])),
            constraints=[f"ATP >= {min_atp}"],
            expected_impact="Improve conversion and reduce stockout-driven CX issues."
        ))

    # Class recommendations (inventory not applicable)
    filtered_classes = []
    for c in ranked["classes"]:
        c["eligible"] = True
        filtered_classes.append(c)
        recs.append(Recommendation(
            entity_type="class",
            entity_id=c["entity_id"],
            entity_name=c.get("entity_name"),
            recommendation_type="email_spotlight",
            rationale="; ".join(c.get("drivers", [])),
            constraints=[],
            expected_impact="Increase engagement and retention by highlighting trending classes."
        ))

    ranked["equipment"] = filtered_equipment
    ranked["classes"] = filtered_classes
    if scope.get("include_watchlist", True):
        ranked["watchlist"] = watchlist

    state["ranked_trends"] = ranked
    state["recommendations"] = recs
    state.setdefault("audit_log", []).append({"node": "policy_filter", "ts": utc_now_iso(), "recommendation_count": len(recs)})
    return state


def no_recs_router(state: State) -> str:
    recs = state.get("recommendations", [])
    if not recs:
        return "n6b_generate_no_recs_response"
    return "n7_generate_narrative_and_actions"


def n6b_generate_no_recs_response(state: State) -> State:
    report = state.get("freshness_report", {})
    caveats = report.get("required_caveats", [])
    narrative = (
        "No eligible recommendations could be produced for the requested scope. "
        "This is typically due to inventory constraints, insufficient volume, or data quality gating.\n"
    )
    state["output_contract"] = OutputContract(
        run_id=state["run_id"],
        generated_at=utc_now_iso(),
        scope=state.get("scope", {}),
        trends=state.get("ranked_trends", {"classes": [], "equipment": [], "accessories": [], "watchlist": []}),
        recommendations=[],
        caveats=caveats,
        citations=[],
        narrative=narrative,
    )
    return state


def n7_generate_narrative_and_actions(state: State) -> State:
    """
    LLM narrative generator. You can:
    - Use a real ChatModel (recommended)
    - Or keep deterministic narrative until you wire the model.
    """
    scope = state["scope"]
    ranked = state["ranked_trends"]
    recs: List[Recommendation] = state.get("recommendations", [])
    freshness = state.get("freshness_report", {})
    approved = state.get("approved_metrics", [])

    citations = [m.get("source_ref") for m in approved if m.get("source_ref")]
    if isinstance(ranked.get("watchlist"), list) and ranked["watchlist"]:
        citations.append("inventory://availability")

    caveats = freshness.get("required_caveats", [])
    if freshness.get("overall_status") == "WARN":
        caveats = caveats + ["Some dependencies were in WARN state; interpret trends with caution."]

    # If a chat model is provided, use it; otherwise generate a structured deterministic narrative.
    model: Optional[BaseChatModel] = state["tools"].get("llm")  # may be None

    # Build compact trend summary for prompting
    def summarize(items: List[Dict[str, Any]], limit: int = 5) -> str:
        out = []
        for c in items[:limit]:
            out.append(f"- {c['entity_type']} {c['entity_id']} score={c.get('score'):.2f} conf={c.get('confidence'):.2f} drivers={'; '.join(c.get('drivers', []))}")
        return "\n".join(out) if out else "(none)"

    trend_summary = (
        f"Scope: {json.dumps(scope)}\n\n"
        f"Top Equipment:\n{summarize(ranked.get('equipment', []))}\n\n"
        f"Top Classes:\n{summarize(ranked.get('classes', []))}\n\n"
        f"Watchlist:\n{summarize(ranked.get('watchlist', []))}\n"
    )

    if model and hasattr(model, "invoke"):
        sys = SystemMessage(content=(
            "You are a Peloton merchandising analytics assistant. "
            "Use only the provided trend summaries. Do not invent numbers. "
            "Explain why items are trending, consider seasonality and inventory, and propose actions by channel."
        ))
        user = HumanMessage(content=(
            "Generate an executive narrative with:\n"
            "1) Executive summary bullets\n"
            "2) What is trending (equipment/classes/accessories)\n"
            "3) Why (drivers)\n"
            "4) Recommended actions (channel-specific)\n"
            "5) Caveats\n\n"
            "Trend data:\n"
            f"{trend_summary}\n\n"
            f"Caveats: {caveats}\n"
        ))
        llm_resp = model.invoke([sys, user])
        narrative = getattr(llm_resp, "content", str(llm_resp))
    else:
        narrative = (
            "Trend & Merchandising Insights\n"
            "------------------------------\n"
            f"Time window: {scope['time_window']['start']} to {scope['time_window']['end']}\n"
            f"Region: {scope['region_code']}\n\n"
            "Top trends (evidence-based):\n"
            f"{trend_summary}\n\n"
            "Recommended actions:\n"
            + "\n".join([f"- {r.recommendation_type}: {r.entity_type} {r.entity_id} — {r.rationale}" for r in recs[:10]])
            + ("\n\nCaveats:\n" + "\n".join([f"- {c}" for c in caveats]) if caveats else "")
        )

    out = OutputContract(
        run_id=state["run_id"],
        generated_at=utc_now_iso(),
        scope=scope,
        trends=ranked,
        recommendations=recs,
        caveats=caveats,
        citations=[c for c in citations if c],
        narrative=narrative,
    )
    state["output_contract"] = out
    state.setdefault("audit_log", []).append({"node": "narrative", "ts": utc_now_iso(), "citations": out.citations})
    return state


def n8_citation_and_output_validation(state: State) -> State:
    """
    Enforce:
    - citations present for governed metrics
    - no empty narrative
    - recommendations comply with filters (eligible)
    """
    out: OutputContract = state["output_contract"]
    errors = state.setdefault("errors", [])

    if not out.narrative or len(out.narrative.strip()) < 20:
        errors.append("Narrative too short or missing.")

    # Ensure governed metric citations exist
    approved = state.get("approved_metrics", [])
    required_refs = [m.get("source_ref") for m in approved if m.get("source_ref")]
    missing_refs = [r for r in required_refs if r and r not in out.citations]
    if missing_refs:
        errors.append(f"Missing metric citations: {missing_refs}")

    # Ensure no recommendations for ineligible items (example check for equipment low ATP filtered)
    ranked = state.get("ranked_trends", {})
    ineligible_equipment = {c["entity_id"] for c in ranked.get("watchlist", []) if c.get("filtered_reason")}
    for r in out.recommendations:
        if r.entity_type == "equipment" and r.entity_id in ineligible_equipment:
            errors.append(f"Recommendation violates inventory filter: {r.entity_id}")

    state.setdefault("audit_log", []).append({"node": "validate_output", "ts": utc_now_iso(), "errors": errors})
    return state


# =============================================================================
# Graph Assembly
# =============================================================================

def build_trend_merchandising_graph() -> Any:
    graph = StateGraph(State)

    graph.add_node("n0_intake_scope", n0_intake_scope)
    graph.add_node("n1_validate_metrics", n1_validate_metrics_semantic_layer)
    graph.add_node("n2_check_freshness", n2_check_data_freshness)
    graph.add_node("n2b_handle_blocked_freshness", n2b_handle_blocked_freshness)
    graph.add_node("n3_retrieve_analytics_signals", n3_retrieve_analytics_signals)
    graph.add_node("n4_normalize_join", n4_normalize_and_join_signals)
    graph.add_node("n5_score_rank", n5_score_and_rank_trends)
    graph.add_node("n6_policy_filter", n6_apply_inventory_and_policy_filters)
    graph.add_node("n6b_generate_no_recs_response", n6b_generate_no_recs_response)
    graph.add_node("n7_generate_narrative_and_actions", n7_generate_narrative_and_actions)
    graph.add_node("n8_validate_output", n8_citation_and_output_validation)

    graph.add_edge(START, "n0_intake_scope")
    graph.add_edge("n0_intake_scope", "n1_validate_metrics")
    graph.add_edge("n1_validate_metrics", "n2_check_freshness")

    graph.add_conditional_edges(
        "n2_check_freshness",
        freshness_router,
        {
            "n2b_handle_blocked_freshness": "n2b_handle_blocked_freshness",
            "n3_retrieve_analytics_signals": "n3_retrieve_analytics_signals",
        },
    )

    graph.add_edge("n2b_handle_blocked_freshness", "n8_validate_output")
    graph.add_edge("n3_retrieve_analytics_signals", "n4_normalize_join")
    graph.add_edge("n4_normalize_join", "n5_score_rank")
    graph.add_edge("n5_score_rank", "n6_policy_filter")

    graph.add_conditional_edges(
        "n6_policy_filter",
        no_recs_router,
        {
            "n6b_generate_no_recs_response": "n6b_generate_no_recs_response",
            "n7_generate_narrative_and_actions": "n7_generate_narrative_and_actions",
        },
    )
    graph.add_edge("n6b_generate_no_recs_response", "n8_validate_output")
    graph.add_edge("n7_generate_narrative_and_actions", "n8_validate_output")
    graph.add_edge("n8_validate_output", END)

    return graph.compile()


# =============================================================================
# Runner / Example Usage
# =============================================================================

def make_default_tools(llm: Optional[BaseChatModel] = None) -> Dict[str, Any]:
    return {
        "bi": DummyBISemanticLayer(),
        "inventory": DummyInventoryTool(),
        "search": DummySearchTool(),
        "seasonality": DummySeasonalityTool(),
        "freshness": DummyFreshnessTool(),
        "llm": llm,  # optional
    }


def run_agent(user_request: str, tools: Optional[Dict[str, Any]] = None) -> Tuple[OutputContract, State]:
    graph = build_trend_merchandising_graph()
    state: State = {
        "run_id": f"run_{int(datetime.now(timezone.utc).timestamp())}",
        "user_request": user_request,
        "tools": tools or make_default_tools(llm=None),
        "audit_log": [],
        "errors": [],
    }
    final_state = graph.invoke(state)
    out: OutputContract = final_state["output_contract"]
    return out, final_state


if __name__ == "__main__":
    request = "Give me trending classes and equipment for the last month; include inventory constraints and seasonality."
    output, final_state = run_agent(request)

    print("=== HUMAN NARRATIVE ===")
    print(output.narrative)
    print("\n=== JSON OUTPUT CONTRACT ===")
    print(output.to_json())
    print("\n=== ERRORS ===")
    print(final_state.get("errors", []))
