"""
Pydantic Schemas — Request / Response Models
=============================================
"""

from typing import Optional, List, Dict
from pydantic import BaseModel


class ContractRequest(BaseModel):
    source_code: str
    contract_name: Optional[str] = None


class VulnerabilityResult(BaseModel):
    swc_id: str
    vulnerability_type: str
    severity: str
    confidence: float
    function_affected: str
    description: str
    remediation: str
    cross_function: bool = False
    line_hint: Optional[int] = None


class GraphStats(BaseModel):
    function_nodes: int
    statement_nodes: int
    variable_nodes: int
    call_edges: int
    control_flow_edges: int
    data_flow_edges: int
    ast_nodes: int
    total_nodes: int
    total_edges: int


class ModelMetrics(BaseModel):
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    auc_roc: float
    risk_score: float
    model_name: str
    inference_time_ms: float


class AnalysisResponse(BaseModel):
    contract_hash: str
    contract_name: str
    vulnerabilities: List[VulnerabilityResult]
    metrics: ModelMetrics
    graph_stats: GraphStats
    analysis_time: float
    safe: bool
    summary: str
    call_graph: Dict
    cfg_graph: Dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    model_name: str
