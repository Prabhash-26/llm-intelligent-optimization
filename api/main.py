"""
Angel One Agentic AI Enterprise API Server (FastAPI)
======================================================
Production-ready REST and Agent API endpoints supporting:
  - Multi-Agent Orchestration & Chat
  - Talk-to-Data Analytics
  - Model Context Protocol (MCP) Tool Execution
  - Combinatorial Optimization Engine
  - Role & Task Autonomy Matrix

Run locally:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations
import os
import sys
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Ensure workspace root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.orchestrator import MultiAgentOrchestrator
from agent.mcp_server import MCPServer
from agent.talk_to_data import TalkToDataAgent
from agent.role_task_analyzer import RoleTaskAnalyzer

app = FastAPI(
    title="Angel One Agentic AI Platform",
    description="Multi-Agent systems, Talk-to-Data analytics, MCP tool orchestration, and intelligent optimization.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service singletons
orchestrator = MultiAgentOrchestrator()
mcp_server = MCPServer()
data_agent = TalkToDataAgent()
role_analyzer = RoleTaskAnalyzer()


# Pydantic Schemas
class ChatRequest(BaseModel):
    query: str = Field(..., example="Investigate latency spike on Gateway-4 and suggest reallocation")
    session_id: Optional[str] = Field(None, example="sess-102")


class MCPToolCallRequest(BaseModel):
    tool_name: str = Field(..., example="angel_query_telemetry")
    arguments: Dict[str, Any] = Field(default_factory=dict)
    caller_id: Optional[str] = "api_client"


class OptimizationRequest(BaseModel):
    problem_type: str = Field("traffic_dispatch", example="traffic_dispatch")
    constraints: Dict[str, Any] = Field(default_factory=lambda: {"max_latency_ms": 25, "sla_p99": 99.9})
    algorithm_mode: str = Field("chain_of_thought", example="chain_of_thought")


class HumanApprovalRequest(BaseModel):
    session_id: str
    action_id: str
    approved: bool
    reviewer_notes: Optional[str] = None


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Angel One Agentic AI Platform",
        "mcp_server": "ONLINE",
        "protocol_version": "2024-11-05",
    }


@app.post("/api/agent/chat")
def run_agent_chat(req: ChatRequest):
    """Executes the multi-agent orchestration loop with ReAct traces and tool calls."""
    try:
        execution_state = orchestrator.execute_workflow(query=req.query)
        return {
            "session_id": execution_state.session_id,
            "user_query": execution_state.user_query,
            "active_agent": execution_state.active_agent.value,
            "confidence_score": execution_state.confidence_score,
            "risk_level": execution_state.risk_level.value,
            "requires_human_approval": execution_state.requires_human_approval,
            "human_approved": execution_state.human_approved,
            "reasoning_traces": execution_state.reasoning_traces,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "parameters": tc.parameters,
                    "output": tc.output,
                    "latency_ms": tc.latency_ms,
                }
                for tc in execution_state.tool_calls
            ],
            "final_output": execution_state.final_output,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agent/talk-to-data")
def talk_to_data(query: str = Query("Why did latency spike this morning?")):
    """Converts natural language questions into statistical charts, anomalies, and decisions."""
    res = data_agent.query(query)
    return {
        "query": query,
        "summary": res.natural_language_summary,
        "key_metrics": res.key_metrics,
        "chart_type": res.chart_type,
        "chart_data": res.chart_data,
        "detected_anomalies_count": res.detected_anomalies_count,
        "recommended_decision": res.recommended_decision,
        "sql_equivalent": res.sql_equivalent,
    }


@app.get("/api/mcp/tools")
def list_mcp_tools():
    """Lists all tools hosted on the Model Context Protocol server."""
    return {"tools": mcp_server.list_tools(), "server": mcp_server.server_name}


@app.post("/api/mcp/call")
def call_mcp_tool(req: MCPToolCallRequest):
    """Executes a specific tool via the Model Context Protocol interface."""
    try:
        return mcp_server.call_tool(name=req.tool_name, arguments=req.arguments, caller_id=req.caller_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/optimizer/solve")
def solve_combinatorial(req: OptimizationRequest):
    """Runs combinatorial optimization using LLM reasoning and heuristics."""
    result = mcp_server.call_tool(
        name="angel_combinatorial_optimizer",
        arguments={
            "problem_type": req.problem_type,
            "constraints": req.constraints,
            "algorithm_mode": req.algorithm_mode,
        },
    )
    return result["raw_result"]


@app.get("/api/roles/analyze")
def analyze_role(role_title: str = Query("FinTech SRE / DevOps Engineer")):
    """Evaluates task autonomy and human-AI labor division for a given enterprise role."""
    report = role_analyzer.analyze_role(role_title)
    return {
        "role_title": report.role_title,
        "department": report.department,
        "autonomous_ai_pct": report.autonomous_ai_pct,
        "collaborative_pct": report.collaborative_pct,
        "human_judgment_pct": report.human_judgment_pct,
        "task_breakdown": [
            {
                "task_name": t.task_name,
                "category": t.autonomy_category,
                "automation_potential_pct": t.automation_potential_pct,
                "reasoning": t.explainability_reasoning,
                "risk_if_unattended": t.risk_if_unattended,
            }
            for t in report.task_breakdown
        ],
        "career_advice": report.career_growth_recommendations,
        "executive_takeaway": report.executive_takeaway,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
