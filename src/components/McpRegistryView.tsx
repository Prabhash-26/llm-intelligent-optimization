import React, {useState, useEffect} from 'react';
import {
  Plug,
  Play,
  CheckCircle,
  Clock,
  ShieldCheck,
  ShieldAlert,
  Code2,
  Terminal,
  Server,
  Layers,
  Sparkles,
  ExternalLink,
} from 'lucide-react';
import type {MCPToolDef} from '../types/agent';

export const McpRegistryView: React.FC = () => {
  const [tools, setTools] = useState<MCPToolDef[]>([]);
  const [selectedTool, setSelectedTool] = useState<MCPToolDef | null>(null);
  const [customArgs, setCustomArgs] = useState<string>('{}');
  const [executing, setExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState<any>(null);

  useEffect(() => {
    fetch('/api/mcp/tools')
      .then(res => res.json())
      .then(data => {
        if (data.tools) {
          setTools(data.tools);
          if (data.tools.length > 0) {
            handleSelectTool(data.tools[0]);
          }
        }
      })
      .catch(err => console.error('Error fetching MCP tools:', err));
  }, []);

  const handleSelectTool = (tool: MCPToolDef) => {
    setSelectedTool(tool);
    setExecutionResult(null);

    // Populate realistic default arguments based on tool name
    let sampleArgs: any = {};
    if (tool.name === 'angel_query_telemetry') {
      sampleArgs = {stream_id: 'GW-MUMBAI-01', metric: 'latency_ms', lookback_minutes: 60};
    } else if (tool.name === 'angel_combinatorial_optimizer') {
      sampleArgs = {
        problem_type: 'traffic_dispatch',
        constraints: {max_latency_ms: 20, min_throughput_ops: 4000},
        algorithm_mode: 'chain_of_thought',
      };
    } else if (tool.name === 'angel_dispatch_jira_incident') {
      sampleArgs = {
        summary: 'P99 Latency spike breach on Order Routing Engine',
        severity: 'P2-High',
        affected_service: 'OrderExecutionGateway',
      };
    } else if (tool.name === 'angel_rebalance_trading_cluster') {
      sampleArgs = {
        source_node: 'GW-MUMBAI-01',
        target_nodes: ['GW-BLR-02', 'GW-HYD-03'],
        traffic_percentage: 40,
      };
    } else if (tool.name === 'angel_hr_role_task_decomposer') {
      sampleArgs = {
        role_title: 'FinTech SRE / DevOps Engineer',
        department: 'Infrastructure',
      };
    }

    setCustomArgs(JSON.stringify(sampleArgs, null, 2));
  };

  const executeMcpTool = async () => {
    if (!selectedTool) return;
    setExecuting(true);
    setExecutionResult(null);

    try {
      let parsed = {};
      try {
        parsed = JSON.parse(customArgs);
      } catch (e) {
        alert('Invalid JSON in arguments');
        setExecuting(false);
        return;
      }

      const res = await fetch('/api/mcp/call', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          tool_name: selectedTool.name,
          arguments: parsed,
        }),
      });

      const data = await res.json();
      setExecutionResult(data);
    } catch (err: any) {
      console.error('MCP execution error:', err);
      setExecutionResult({error: err.message});
    } finally {
      setExecuting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="p-5 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900 to-slate-950 border border-slate-800 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="px-2 py-0.5 text-[11px] font-bold uppercase rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
                Model Context Protocol (MCP) Standard
              </span>
              <span className="text-xs text-slate-400">Spec: v2024-11-05</span>
            </div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              Enterprise Tool Registry & Execution Server
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Connects AI agents to Angel One's live telemetry, Jira queues, trading cluster rebalancers, and HR systems with strict schema contracts.
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <span className="px-3 py-1 rounded-lg bg-slate-950 border border-slate-800 text-slate-300 font-mono">
              Server: <strong className="text-purple-400">angelone-enterprise-mcp</strong>
            </span>
          </div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Tool List (4 cols) */}
        <div className="lg:col-span-4 space-y-3">
          <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider px-1">
            Registered Enterprise Tools ({tools.length})
          </h3>

          <div className="space-y-2">
            {tools.map(tool => {
              const isSelected = selectedTool?.name === tool.name;
              return (
                <div
                  key={tool.name}
                  onClick={() => handleSelectTool(tool)}
                  className={`p-3.5 rounded-xl border transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-purple-950/20 border-purple-500/60 shadow-lg shadow-purple-950/30 ring-1 ring-purple-500/30'
                      : 'bg-slate-900/60 hover:bg-slate-900 border-slate-800'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-mono text-xs font-bold text-slate-200">
                      {tool.name}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded uppercase font-semibold font-mono bg-slate-800 text-slate-400">
                      {tool.category}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                    {tool.description}
                  </p>

                  <div className="mt-2.5 flex items-center justify-between text-[11px]">
                    {tool.requires_approval ? (
                      <span className="flex items-center gap-1 text-red-400 font-medium">
                        <ShieldAlert className="w-3 h-3" />
                        Requires HITL Approval
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-emerald-400 font-medium">
                        <ShieldCheck className="w-3 h-3" />
                        Autonomous Ready
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Tool Playground & Schema Inspector (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          {selectedTool && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-xl space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div>
                  <div className="flex items-center gap-2">
                    <Plug className="w-4 h-4 text-purple-400" />
                    <h3 className="text-sm font-bold text-white font-mono">
                      {selectedTool.name}
                    </h3>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    {selectedTool.description}
                  </p>
                </div>

                <button
                  onClick={executeMcpTool}
                  disabled={executing}
                  className="px-4 py-2 rounded-lg bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:opacity-50 text-white font-semibold text-xs shadow-md transition-all cursor-pointer flex items-center gap-2"
                >
                  {executing ? (
                    <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Play className="w-3.5 h-3.5 fill-white" />
                  )}
                  <span>Invoke MCP Tool</span>
                </button>
              </div>

              {/* JSON Input Parameters */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold text-slate-300 flex items-center gap-1">
                    <Code2 className="w-3.5 h-3.5 text-orange-400" />
                    <span>Tool Arguments (JSON Payload conforming to MCP Schema)</span>
                  </label>
                  <span className="text-[10px] text-slate-400 font-mono">Editable</span>
                </div>
                <textarea
                  value={customArgs}
                  onChange={e => setCustomArgs(e.target.value)}
                  rows={6}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs font-mono text-purple-200 focus:outline-none focus:ring-1 focus:ring-purple-500"
                />
              </div>

              {/* Execution Result Box */}
              {executionResult && (
                <div className="space-y-2 pt-2 border-t border-slate-800">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Terminal className="w-4 h-4 text-emerald-400" />
                      <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                        MCP Execution Result
                      </span>
                    </div>
                    {executionResult.latency_ms && (
                      <span className="text-[11px] font-mono text-emerald-400">
                        ⚡ Latency: {executionResult.latency_ms} ms
                      </span>
                    )}
                  </div>
                  <pre className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-emerald-300 overflow-x-auto max-h-64">
                    {JSON.stringify(executionResult.result || executionResult, null, 2)}
                  </pre>
                </div>
              )}

              {/* Input Schema Specification Details */}
              <div className="space-y-1 pt-2 border-t border-slate-800">
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                  JSON Schema Specification
                </span>
                <pre className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 text-[11px] font-mono text-slate-400 overflow-x-auto max-h-40">
                  {JSON.stringify(selectedTool.inputSchema, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
