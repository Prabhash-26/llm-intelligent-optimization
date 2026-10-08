"""
Model Context Protocol (MCP) Enterprise Tool Connector
========================================================
Implements MCP-compliant tool protocols for enterprise tool execution at Angel One.
Enables LLM agents to securely interact with production databases, monitoring systems,
Jira/Slack/PagerDuty dispatch, and compute optimization clusters.
"""

from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class MCPToolParameterProperty:
    type: str
    description: str
    enum: Optional[List[str]] = None
    default: Optional[Any] = None


@dataclass
class MCPToolSchema:
    name: str
    description: str
    inputSchema: Dict[str, Any]
    category: str = "enterprise_operations"
    requires_approval: bool = False


class MCPServer:
    """
    Simulated MCP Tool Server hosting enterprise tools for Angel One.
    Conforms to Model Context Protocol standard schema specifications.
    """

    def __init__(self, server_name: str = "angelone-enterprise-mcp"):
        self.server_name = server_name
        self.protocol_version = "2024-11-05"
        self._tools: Dict[str, MCPToolSchema] = {}
        self._execution_log: List[Dict[str, Any]] = []
        self._initialize_tools()

    def _initialize_tools(self):
        """Initializes enterprise tool definitions."""
        self.register_tool(
            MCPToolSchema(
                name="angel_query_telemetry",
                description="Queries real-time IoT and FinTech trading gateway telemetry, latency percentiles, and anomaly markers.",
                category="observability",
                requires_approval=False,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "stream_id": {"type": "string", "description": "Identifier for the sensor or trading gateway (e.g., 'GATEWAY-04')"},
                        "metric": {"type": "string", "enum": ["latency_ms", "packet_loss", "order_throughput", "cpu_load"], "description": "Metric to analyze"},
                        "lookback_minutes": {"type": "integer", "description": "Minutes of history to query", "default": 60},
                    },
                    "required": ["stream_id", "metric"],
                },
            )
        )

        self.register_tool(
            MCPToolSchema(
                name="angel_combinatorial_optimizer",
                description="Solves NP-hard combinatorial resource allocation, workforce shift assignment, and packet routing using LLM + OR solvers.",
                category="optimization",
                requires_approval=False,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "problem_type": {"type": "string", "enum": ["traffic_dispatch", "workforce_scheduling", "bin_packing_compute", "knapsack_portfolio"], "description": "Optimization problem class"},
                        "constraints": {"type": "object", "description": "Dictionary of budget, SLA, and capacity bounds"},
                        "algorithm_mode": {"type": "string", "enum": ["chain_of_thought", "self_consistency", "rag_augmented"], "default": "chain_of_thought"},
                    },
                    "required": ["problem_type", "constraints"],
                },
            )
        )

        self.register_tool(
            MCPToolSchema(
                name="angel_dispatch_jira_incident",
                description="Creates and assigns an enterprise incident ticket in Angel One Jira/ServiceNow queue.",
                category="collaboration",
                requires_approval=False,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "Concise headline of the incident"},
                        "severity": {"type": "string", "enum": ["P1-Critical", "P2-High", "P3-Medium", "P4-Low"]},
                        "affected_service": {"type": "string", "description": "Name of the service (e.g., 'OrderExecutionEngine')"},
                        "root_cause_summary": {"type": "string", "description": "Agent-synthesized root cause analysis"},
                    },
                    "required": ["summary", "severity", "affected_service"],
                },
            )
        )

        self.register_tool(
            MCPToolSchema(
                name="angel_rebalance_trading_cluster",
                description="Rebalances trading traffic nodes or re-routes order queues during high-volatility anomalies (High Risk).",
                category="infrastructure",
                requires_approval=True,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_node": {"type": "string", "description": "Degraded node ID"},
                        "target_nodes": {"type": "array", "items": {"type": "string"}, "description": "Healthy target nodes"},
                        "traffic_percentage": {"type": "number", "description": "Percentage of traffic to reroute (1-100)"},
                        "drain_timeout_seconds": {"type": "integer", "description": "Graceful connection drain timeout", "default": 30},
                    },
                    "required": ["source_node", "target_nodes", "traffic_percentage"],
                },
            )
        )

        self.register_tool(
            MCPToolSchema(
                name="angel_hr_role_task_decomposer",
                description="Decomposes a specific employee role into distinct tasks and categorizes autonomy levels (Autonomous vs Human-Judgment).",
                category="hr_analytics",
                requires_approval=False,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "role_title": {"type": "string", "description": "Title of the job/role to evaluate"},
                        "department": {"type": "string", "description": "Department name (e.g., 'Operations', 'Engineering', 'HR')"},
                    },
                    "required": ["role_title"],
                },
            )
        )

    def register_tool(self, tool: MCPToolSchema):
        self._tools[tool.name] = tool

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns the list of available tools in standard MCP format."""
        return [asdict(t) for t in self._tools.values()]

    def call_tool(self, name: str, arguments: Dict[str, Any], caller_id: str = "agent_orchestrator") -> Dict[str, Any]:
        """
        Executes a registered MCP tool and logs the interaction.
        """
        start_time = time.time()
        if name not in self._tools:
            raise ValueError(f"MCP Tool '{name}' not found on server '{self.server_name}'.")

        tool_meta = self._tools[name]

        # Dispatch simulated execution logic
        if name == "angel_query_telemetry":
            result = {
                "stream_id": arguments.get("stream_id", "GATEWAY-04"),
                "metric": arguments.get("metric", "latency_ms"),
                "p50_ms": 2.1,
                "p95_ms": 14.8,
                "p99_ms": 118.4,
                "anomaly_spikes_detected": 3,
                "health_status": "DEGRADED",
            }
        elif name == "angel_combinatorial_optimizer":
            result = {
                "problem": arguments.get("problem_type"),
                "algorithm": arguments.get("algorithm_mode", "chain_of_thought"),
                "optimal_schedule": [
                    {"step": 1, "action": "Reroute 40% traffic to Gateway-East", "latency_delta": "-32ms"},
                    {"step": 2, "action": "Scale worker replicas +2", "cost_delta": "+$4.20/hr"},
                    {"step": 3, "action": "Throttle non-critical analytics sync", "headroom_gained": "25%"},
                ],
                "objective_value": 94.6,
                "convergence_time_ms": 320,
            }
        elif name == "angel_dispatch_jira_incident":
            ticket_num = int(time.time()) % 10000
            result = {
                "ticket_key": f"ANGEL-OPS-{ticket_num}",
                "status": "CREATED",
                "severity": arguments.get("severity", "P2-High"),
                "affected_service": arguments.get("affected_service"),
                "slack_channel_notified": "#fintech-infra-alerts",
            }
        elif name == "angel_rebalance_trading_cluster":
            result = {
                "operation": "CLUSTER_TRAFFIC_REROUTE",
                "source": arguments.get("source_node"),
                "target": arguments.get("target_nodes"),
                "rerouted_pct": arguments.get("traffic_percentage"),
                "status": "APPROVED_AND_EXECUTED",
                "safe_drain_completed": True,
            }
        elif name == "angel_hr_role_task_decomposer":
            result = {
                "role": arguments.get("role_title"),
                "department": arguments.get("department", "Engineering"),
                "autonomy_breakdown": {
                    "autonomous_ai_pct": 45,
                    "human_ai_collaborative_pct": 35,
                    "strict_human_judgment_pct": 20,
                },
                "high_leverage_upskilling": [
                    "Prompt evaluation & agentic flow engineering",
                    "Exception auditing and edge-case governance",
                    "Domain architectural reviews",
                ],
            }
        else:
            result = {"status": "ok", "echo": arguments}

        latency_ms = round((time.time() - start_time) * 1000, 2)
        log_entry = {
            "timestamp": time.time(),
            "caller_id": caller_id,
            "tool_name": name,
            "arguments": arguments,
            "latency_ms": latency_ms,
            "requires_approval": tool_meta.requires_approval,
            "success": True,
        }
        self._execution_log.append(log_entry)

        return {
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
            "raw_result": result,
            "meta": {
                "server": self.server_name,
                "latency_ms": latency_ms,
                "mcp_version": self.protocol_version,
            },
        }

    def get_audit_log(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self._execution_log[-limit:]


if __name__ == "__main__":
    server = MCPServer()
    print("Registered MCP Tools:")
    for t in server.list_tools():
        print(f" - [{t['category'].upper()}] {t['name']} (Approval: {t['requires_approval']})")

    print("\nSimulating MCP tool call: angel_query_telemetry...")
    res = server.call_tool("angel_query_telemetry", {"stream_id": "GATEWAY-04", "metric": "latency_ms"})
    print(res["content"][0]["text"])
