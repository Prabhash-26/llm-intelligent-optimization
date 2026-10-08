import React, {useState, useEffect} from 'react';
import {
  BarChart3,
  Search,
  Sparkles,
  AlertTriangle,
  Database,
  ArrowRight,
  TrendingUp,
  Activity,
  CheckCircle,
  HelpCircle,
} from 'lucide-react';
import type {TalkToDataResult, TelemetryPoint} from '../types/agent';

const SAMPLE_QUESTIONS = [
  'Why did latency spike this morning?',
  'What is the peak order throughput and capacity headroom across gateways?',
  'Show me overall gateway health overview and p95 SLAs',
];

export const TalkToDataView: React.FC = () => {
  const [question, setQuestion] = useState('Why did latency spike this morning?');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TalkToDataResult | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<TelemetryPoint | null>(null);

  const executeQuery = async (queryText: string) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/agent/talk-to-data?query=${encodeURIComponent(queryText)}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error('Talk-to-data error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    executeQuery('Why did latency spike this morning?');
  }, []);

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="p-5 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900 to-slate-950 border border-slate-800 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="px-2 py-0.5 text-[11px] font-bold uppercase rounded bg-orange-500/10 text-orange-400 border border-orange-500/20">
                OptiLLM No-SQL Analytics
              </span>
              <span className="text-xs text-slate-400">JD Problem: "Talk to Data"</span>
            </div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              Plain-English Telemetry & Business Insights
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Ask natural language questions about trading engine telemetry, order queues, and anomaly spikes. No SQL or dashboards required.
            </p>
          </div>
        </div>

        {/* Search Input Bar */}
        <div className="mt-4 flex flex-col sm:flex-row gap-2">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3.5" />
            <input
              type="text"
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && executeQuery(question)}
              placeholder="Ask anything (e.g. 'Why did latency spike between 09:15 and 09:45 AM?')..."
              className="w-full bg-slate-950 border border-slate-700 rounded-xl pl-10 pr-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-orange-500"
            />
          </div>
          <button
            onClick={() => executeQuery(question)}
            disabled={loading}
            className="px-5 py-2.5 bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 disabled:opacity-50 text-white font-semibold text-xs rounded-xl shadow-md transition-all cursor-pointer flex items-center justify-center gap-2"
          >
            {loading ? (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <Sparkles className="w-4 h-4" />
            )}
            <span>Ask Data</span>
          </button>
        </div>

        {/* Suggestion Chips */}
        <div className="mt-3 flex items-center flex-wrap gap-2">
          <span className="text-[11px] text-slate-400 font-medium">Try asking:</span>
          {SAMPLE_QUESTIONS.map((q, idx) => (
            <button
              key={idx}
              onClick={() => {
                setQuestion(q);
                executeQuery(q);
              }}
              className="text-[11px] px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-700/60 transition-all cursor-pointer"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Results View */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Main Visual & Insights (8 cols) */}
          <div className="lg:col-span-8 space-y-6">
            {/* Dynamic Telemetry Graph */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-xl">
              <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-orange-400" />
                  <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                    Dynamic Telemetry Curve (Real-Time Sensor Feed)
                  </h3>
                </div>
                <div className="flex items-center gap-3 text-xs">
                  <span className="flex items-center gap-1.5 text-blue-400">
                    <span className="w-3 h-0.5 bg-blue-500 inline-block" /> Latency (ms)
                  </span>
                  <span className="flex items-center gap-1.5 text-red-400">
                    <span className="w-2 h-2 rounded-full bg-red-500 inline-block animate-ping" /> Anomaly Spike
                  </span>
                </div>
              </div>

              {/* Responsive SVG Chart */}
              <div className="relative h-64 w-full bg-slate-950/70 rounded-lg p-3 border border-slate-800/80 flex flex-col justify-end">
                {result.records && result.records.length > 0 ? (
                  <svg className="w-full h-full overflow-visible" viewBox="0 0 800 200">
                    {/* Grid lines */}
                    <line x1="0" y1="50" x2="800" y2="50" stroke="#1e293b" strokeDasharray="4 4" />
                    <line x1="0" y1="100" x2="800" y2="100" stroke="#1e293b" strokeDasharray="4 4" />
                    <line x1="0" y1="150" x2="800" y2="150" stroke="#1e293b" strokeDasharray="4 4" />

                    {/* Polyline Path */}
                    {(() => {
                      const maxVal = Math.max(...result.records.map(r => r.latency_ms), 60);
                      const points = result.records
                        .map((r, i) => {
                          const x = (i / (result.records.length - 1)) * 780 + 10;
                          const y = 190 - (r.latency_ms / maxVal) * 170;
                          return `${x},${y}`;
                        })
                        .join(' ');

                      return (
                        <>
                          <polyline
                            fill="none"
                            stroke="#3b82f6"
                            strokeWidth="2.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            points={points}
                          />

                          {/* Data dots and anomaly markers */}
                          {result.records.map((r, i) => {
                            const x = (i / (result.records.length - 1)) * 780 + 10;
                            const y = 190 - (r.latency_ms / maxVal) * 170;
                            const isAnomaly = r.anomaly === 1;

                            return (
                              <g
                                key={i}
                                onMouseEnter={() => setHoveredPoint(r)}
                                onMouseLeave={() => setHoveredPoint(null)}
                                className="cursor-pointer"
                              >
                                {isAnomaly ? (
                                  <circle
                                    cx={x}
                                    cy={y}
                                    r="6"
                                    className="fill-red-500 stroke-red-200 stroke-2 animate-pulse"
                                  />
                                ) : (
                                  <circle
                                    cx={x}
                                    cy={y}
                                    r="2.5"
                                    className="fill-blue-400 hover:fill-white transition-colors"
                                  />
                                )}
                              </g>
                            );
                          })}
                        </>
                      );
                    })()}
                  </svg>
                ) : (
                  <div className="flex items-center justify-center h-full text-slate-500 text-xs">
                    No chart points
                  </div>
                )}

                {/* Hover Tooltip Overlay */}
                {hoveredPoint && (
                  <div className="absolute top-3 right-3 bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-xs shadow-xl pointer-events-none">
                    <p className="font-semibold text-slate-200">{hoveredPoint.timestamp} ({hoveredPoint.gateway_id})</p>
                    <p className="text-blue-400">Latency: {hoveredPoint.latency_ms} ms</p>
                    <p className="text-slate-400">Throughput: {hoveredPoint.throughput_ops} ops/s</p>
                    {hoveredPoint.anomaly === 1 && (
                      <p className="text-red-400 font-bold mt-0.5">⚠️ Anomaly Spike Detected</p>
                    )}
                  </div>
                )}
              </div>

              {/* Chart Caption */}
              <div className="flex items-center justify-between text-[11px] text-slate-400 mt-2 px-1">
                <span>00:00 (Midnight)</span>
                <span>09:15 (Market Open Spikes)</span>
                <span>Current Time</span>
              </div>
            </div>

            {/* Natural Language Synthesis Box */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-xl space-y-3">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-amber-400" />
                <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                  Agentic Synthesis & Executive Summary
                </h3>
              </div>
              <p className="text-sm text-slate-200 leading-relaxed font-sans">
                {result.summary}
              </p>

              {/* Actionable Decision Callout */}
              <div className="p-3.5 rounded-lg bg-orange-500/10 border border-orange-500/30 flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-orange-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-bold text-orange-300 uppercase tracking-wider">
                    Recommended Autonomous Action
                  </h4>
                  <p className="text-xs text-slate-200 mt-0.5">
                    {result.decision}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Metrics & SQL Translation (4 cols) */}
          <div className="lg:col-span-4 space-y-6">
            {/* KPI Cards */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-xl space-y-3">
              <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                Statistical Aggregates
              </h3>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800">
                  <span className="text-slate-400 text-[11px] block">Peak Latency</span>
                  <span className="text-red-400 font-bold text-lg font-mono">
                    {result.key_metrics.peak_latency_ms} ms
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800">
                  <span className="text-slate-400 text-[11px] block">Baseline Latency</span>
                  <span className="text-emerald-400 font-bold text-lg font-mono">
                    {result.key_metrics.baseline_latency_ms} ms
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800">
                  <span className="text-slate-400 text-[11px] block">Anomalies Detected</span>
                  <span className="text-amber-400 font-bold text-lg font-mono">
                    {result.key_metrics.anomalies_detected} Spikes
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800">
                  <span className="text-slate-400 text-[11px] block">Active Gateways</span>
                  <span className="text-blue-400 font-bold text-lg font-mono">
                    {result.key_metrics.active_gateways} Nodes
                  </span>
                </div>
              </div>
            </div>

            {/* Generated SQL Translation Box */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-xl space-y-2">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Database className="w-4 h-4 text-purple-400" />
                  <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                    Generated SQL Equivalent
                  </h4>
                </div>
                <span className="text-[10px] text-purple-300 bg-purple-500/10 px-1.5 py-0.5 rounded font-mono">
                  Text-to-SQL
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                How the agent translated plain English into enterprise warehouse logic:
              </p>
              <pre className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-[11px] font-mono text-purple-300 overflow-x-auto whitespace-pre-wrap">
                {result.sql_equivalent}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
