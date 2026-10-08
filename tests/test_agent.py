"""
Unit Tests for Angel One Agentic AI Platform
==============================================
Tests multi-agent triage, MCP tool calling, risk gating, and Talk-to-Data engines.
Runs with either pytest or python -m unittest.
"""

import unittest
from agent.orchestrator import MultiAgentOrchestrator, AgentRole, ActionRiskLevel
from agent.mcp_server import MCPServer
from agent.talk_to_data import TalkToDataAgent
from agent.role_task_analyzer import RoleTaskAnalyzer


class TestAngelOneAgenticPlatform(unittest.TestCase):

    def setUp(self):
        self.orchestrator = MultiAgentOrchestrator()
        self.mcp_server = MCPServer()
        self.data_agent = TalkToDataAgent()
        self.role_analyzer = RoleTaskAnalyzer()

    def test_triage_routing_to_sre_for_high_risk(self):
        role, risk, tools = self.orchestrator.plan_and_route("Rebalance cluster nodes and restart trading gateway")
        self.assertEqual(role, AgentRole.OPS_SRE)
        self.assertEqual(risk, ActionRiskLevel.HIGH)
        self.assertIn("simulate_workload_rebalance", tools)

    def test_triage_routing_to_optimizer(self):
        role, risk, tools = self.orchestrator.plan_and_route("Optimize order dispatch across 5 gateways under latency constraints")
        self.assertEqual(role, AgentRole.OPTIMIZER)
        self.assertEqual(risk, ActionRiskLevel.MEDIUM)
        self.assertIn("run_combinatorial_optimizer", tools)

    def test_mcp_server_listing_and_execution(self):
        tools = self.mcp_server.list_tools()
        tool_names = [t["name"] for t in tools]
        self.assertIn("angel_query_telemetry", tool_names)
        self.assertIn("angel_combinatorial_optimizer", tool_names)

        res = self.mcp_server.call_tool("angel_query_telemetry", {"stream_id": "GATEWAY-04", "metric": "latency_ms"})
        self.assertEqual(res["raw_result"]["stream_id"], "GATEWAY-04")
        self.assertIn("p99_ms", res["raw_result"])

    def test_talk_to_data_anomaly_query(self):
        res = self.data_agent.query("Why were there spikes in telemetry?")
        self.assertGreater(res.detected_anomalies_count, 0)
        self.assertGreater(len(res.chart_data), 0)
        self.assertIn("SELECT", res.sql_equivalent)

    def test_role_task_analyzer(self):
        report = self.role_analyzer.analyze_role("FinTech SRE / DevOps Engineer")
        self.assertGreater(report.autonomous_ai_pct, 0)
        self.assertGreaterEqual(len(report.task_breakdown), 3)
        self.assertGreaterEqual(len(report.career_growth_recommendations), 2)

    def test_human_in_the_loop_gate(self):
        state = self.orchestrator.execute_workflow("Emergency failover of gateway nodes")
        self.assertTrue(state.requires_human_approval)


if __name__ == "__main__":
    unittest.main()
