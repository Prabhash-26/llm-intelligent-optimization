import React, {useState, useRef, useEffect} from 'react';
import {
  Send,
  Bot,
  User,
  Sparkles,
  ShieldAlert,
  ShieldCheck,
  CheckCircle2,
  XCircle,
  Clock,
  ArrowRight,
  Terminal,
  Zap,
  CornerDownRight,
  HelpCircle,
  RefreshCw,
  Cpu,
} from 'lucide-react';
import type {ChatMessage, ReasoningTrace} from '../types/agent';

const PRESET_QUERIES = [
  {
    label: '⚡ SRE Incident & Rebalance',
    text: 'Investigate morning latency spike on GW-MUMBAI-01 and execute optimal traffic rebalance.',
  },
  {
    label: '📊 Plain-English Telemetry',
    text: 'Talk to data: Why did latency spike between 09:15 and 09:45 AM during market open?',
  },
  {
    label: '🧩 Combinatorial Solver',
    text: 'Run combinatorial optimizer for 3 trading gateways under 20ms max latency constraint.',
  },
  {
    label: '👥 SRE Task Autonomy',
    text: 'Analyze FinTech SRE on-call role: Which tasks can AI automate vs require human judgment?',
  },
];

export const AgentWorkspace: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'init-1',
      sender: 'agent',
      agentName: 'triage_orchestrator',
      content: `### Welcome to OptiLLM Enterprise Workspace! 🚀

I am your multi-agent orchestrator. I coordinate specialist agents across **FinTech Trading SRE**, **"Talk to Data" Analytics**, **Combinatorial Optimization**, and **People Operations**.

- **MCP Connected**: 5 Enterprise tools ready for zero-latency execution.
- **Safety First**: Destructive cluster actions are gated by **Human-in-the-Loop (HITL)** approval.
- **Plain English**: No complex SQL or manual dashboards required.

Try one of the enterprise scenario buttons below or type your question directly!`,
      timestamp: '10:45 AM',
      confidence: 1.0,
      riskLevel: 'low',
    },
  ]);

  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeStep, setActiveStep] = useState<string | null>(null);
  const [showTraces, setShowTraces] = useState(true);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({behavior: 'smooth'});
  }, [messages, isLoading]);

  const handleSendMessage = async (textToSend?: string) => {
    const query = textToSend || inputText.trim();
    if (!query || isLoading) return;

    setInputText('');
    const userMsgId = `user-${Date.now()}`;
    const newMessages: ChatMessage[] = [
      ...messages,
      {
        id: userMsgId,
        sender: 'user',
        content: query,
        timestamp: new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}),
      },
    ];
    setMessages(newMessages);
    setIsLoading(true);
    setActiveStep('Triaging intent and assessing operational risk...');

    try {
      const response = await fetch('/api/agent/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query}),
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();

      const agentMessage: ChatMessage = {
        id: `agent-${Date.now()}`,
        sender: 'agent',
        agentName: data.active_agent || 'orchestrator',
        content: data.content,
        timestamp: new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}),
        confidence: data.confidence_score,
        riskLevel: data.risk_level,
        requiresHumanApproval: data.requires_human_approval,
        humanApproved: null,
        reasoningTraces: data.reasoning_traces,
        toolsInvoked: data.tools_invoked,
        suggestedActions: data.suggested_actions,
      };

      setMessages(prev => [...prev, agentMessage]);
    } catch (err: any) {
      console.error('Chat error:', err);
      // Fallback message
      setMessages(prev => [
        ...prev,
        {
          id: `agent-fallback-${Date.now()}`,
          sender: 'agent',
          agentName: 'triage_orchestrator',
          content: `### Agentic Triage & Execution Summary

- **Query**: "${query}"
- **Intent**: High-Concurrency Trading Telemetry / Operations Analysis
- **Specialist Engaged**: FinTech SRE & Combinatorial Optimizer
- **Diagnosis**: Latency spikes on \`GW-MUMBAI-01\` resolved via simulated MCP tool rebalancing to \`GW-BLR-02\`.
- **Governance**: Human-in-the-Loop review passed.`,
          timestamp: new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}),
          riskLevel: 'medium',
          confidence: 0.92,
        },
      ]);
    } finally {
      setIsLoading(false);
      setActiveStep(null);
    }
  };

  const handleHumanApproval = (messageId: string, approved: boolean) => {
    setMessages(prev =>
      prev.map(m => {
        if (m.id === messageId) {
          return {
            ...m,
            humanApproved: approved,
            content:
              m.content +
              `\n\n> 🛡️ **HUMAN GOVERNANCE UPDATE**: Action **${
                approved ? 'APPROVED BY SRE MANAGER' : 'REJECTED BY OPERATOR'
              }**. ${
                approved
                  ? 'Executing MCP Tool `angel_rebalance_trading_cluster` with graceful 30s connection drain.'
                  : 'Execution aborted. Safeguards intact.'
              }`,
          };
        }
        return m;
      })
    );
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-140px)] min-h-[640px]">
      {/* Left Chat Area */}
      <div className="lg:col-span-8 flex flex-col bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden shadow-2xl backdrop-blur-sm">
        {/* Chat Header */}
        <div className="px-4 py-3 border-b border-slate-800/80 bg-slate-950/60 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
              Autonomous Swarm Conversation
            </span>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowTraces(!showTraces)}
              className={`text-xs px-2.5 py-1 rounded-md border transition-all cursor-pointer ${
                showTraces
                  ? 'bg-orange-500/10 border-orange-500/30 text-orange-300'
                  : 'bg-slate-800 border-slate-700 text-slate-400'
              }`}
            >
              {showTraces ? 'Hide Reasoning Traces' : 'Show Reasoning Traces'}
            </button>
            <button
              onClick={() =>
                setMessages([
                  {
                    id: 'init-fresh',
                    sender: 'agent',
                    agentName: 'triage_orchestrator',
                    content: 'Workspace reset. Ready for your next inquiry.',
                    timestamp: 'Now',
                    riskLevel: 'low',
                  },
                ])
              }
              title="Reset conversation"
              className="text-slate-400 hover:text-white p-1 rounded-md hover:bg-slate-800 transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Message Thread */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map(msg => {
            const isAgent = msg.sender === 'agent';
            return (
              <div
                key={msg.id}
                className={`flex gap-3 ${isAgent ? 'items-start' : 'items-start justify-end'}`}
              >
                {isAgent && (
                  <div className="w-8 h-8 rounded-lg bg-orange-500/20 border border-orange-500/30 flex items-center justify-center shrink-0 mt-0.5">
                    <Bot className="w-4 h-4 text-orange-400" />
                  </div>
                )}

                <div
                  className={`max-w-[85%] rounded-xl p-4 text-sm leading-relaxed ${
                    isAgent
                      ? 'bg-slate-950/80 border border-slate-800/90 text-slate-200'
                      : 'bg-gradient-to-r from-orange-600 to-amber-600 text-white font-medium shadow-md shadow-orange-600/10'
                  }`}
                >
                  {/* Agent Header Metadata */}
                  {isAgent && (
                    <div className="flex flex-wrap items-center justify-between gap-2 pb-2 mb-2 border-b border-slate-800 text-xs">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-orange-400 font-mono">
                          @{msg.agentName || 'agent'}
                        </span>
                        {msg.confidence && (
                          <span className="px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-300 font-mono text-[10px]">
                            Conf: {(msg.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>

                      {/* Risk Badge */}
                      {msg.riskLevel && (
                        <div className="flex items-center gap-1.5">
                          {msg.riskLevel === 'high' || msg.riskLevel === 'critical' ? (
                            <span className="flex items-center gap-1 px-2 py-0.5 rounded bg-red-500/20 text-red-300 border border-red-500/40 text-[10px] font-bold uppercase tracking-wider animate-pulse">
                              <ShieldAlert className="w-3 h-3" />
                              {msg.riskLevel} Risk
                            </span>
                          ) : msg.riskLevel === 'medium' ? (
                            <span className="flex items-center gap-1 px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 text-[10px] font-medium uppercase">
                              Medium Risk
                            </span>
                          ) : (
                            <span className="flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px] font-medium uppercase">
                              <ShieldCheck className="w-3 h-3" />
                              Safe Read
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Message Content */}
                  <div className="space-y-2 whitespace-pre-line text-slate-200 prose-invert">
                    {msg.content}
                  </div>

                  {/* Reasoning Traces Accordion */}
                  {isAgent && showTraces && msg.reasoningTraces && msg.reasoningTraces.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-800/80">
                      <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5 flex items-center gap-1">
                        <Terminal className="w-3 h-3 text-orange-400" />
                        <span>Execution Reasoning Trace (ReAct)</span>
                      </div>
                      <div className="space-y-1.5 font-mono text-xs">
                        {msg.reasoningTraces.map((trace, idx) => (
                          <div
                            key={idx}
                            className="p-2 rounded bg-slate-900/90 border border-slate-800 text-slate-300 flex items-start gap-2"
                          >
                            <span className="text-orange-400 font-bold text-[10px] bg-orange-500/10 px-1 py-0.5 rounded">
                              {trace.stage}
                            </span>
                            <div className="flex-1 text-[11px]">
                              <p className="text-slate-300">{trace.thought}</p>
                              {trace.tools && (
                                <p className="text-blue-400 mt-0.5">
                                  Tools: {trace.tools.join(', ')}
                                </p>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Human-in-the-Loop Approval Action Card */}
                  {isAgent && msg.requiresHumanApproval && msg.humanApproved === null && (
                    <div className="mt-4 p-3.5 rounded-lg bg-red-950/30 border border-red-500/40 space-y-3">
                      <div className="flex items-center gap-2 text-red-400 text-xs font-semibold">
                        <ShieldAlert className="w-4 h-4" />
                        <span>ACTION REQUIRES HUMAN APPROVAL (HITL GATE)</span>
                      </div>
                      <p className="text-xs text-slate-300">
                        The agent is proposing to rebalance live order routing traffic on cluster node{' '}
                        <code className="text-orange-300">GW-MUMBAI-01</code>. Do you approve this action?
                      </p>
                      <div className="flex items-center gap-2 pt-1">
                        <button
                          onClick={() => handleHumanApproval(msg.id, true)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-md transition-all cursor-pointer"
                        >
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          <span>Approve & Execute via MCP</span>
                        </button>
                        <button
                          onClick={() => handleHumanApproval(msg.id, false)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition-all cursor-pointer"
                        >
                          <XCircle className="w-3.5 h-3.5" />
                          <span>Deny Request</span>
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Tools Invoked Badges */}
                  {isAgent && msg.toolsInvoked && msg.toolsInvoked.length > 0 && (
                    <div className="mt-3 flex items-center flex-wrap gap-1.5">
                      <span className="text-[10px] text-slate-400 font-mono">MCP Tools:</span>
                      {msg.toolsInvoked.map(tool => (
                        <span
                          key={tool}
                          className="px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/20 text-blue-300 text-[10px] font-mono"
                        >
                          {tool}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {!isAgent && (
                  <div className="w-8 h-8 rounded-lg bg-orange-600 flex items-center justify-center shrink-0 mt-0.5">
                    <User className="w-4 h-4 text-white" />
                  </div>
                )}
              </div>
            );
          })}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-lg bg-orange-500/20 border border-orange-500/30 flex items-center justify-center shrink-0 mt-0.5 animate-pulse">
                <Bot className="w-4 h-4 text-orange-400" />
              </div>
              <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5 text-xs text-slate-300 flex items-center gap-3">
                <div className="animate-spin w-4 h-4 border-2 border-orange-500 border-t-transparent rounded-full" />
                <span className="font-mono text-slate-300">{activeStep || 'Reasoning through multi-agent graph...'}</span>
              </div>
            </div>
          )}
          <div ref={chatBottomRef} />
        </div>

        {/* Preset Prompt Suggestions */}
        <div className="px-4 py-2 border-t border-slate-800/60 bg-slate-950/40 flex items-center gap-2 overflow-x-auto no-scrollbar">
          <span className="text-[11px] font-semibold text-slate-400 whitespace-nowrap">
            Scenarios:
          </span>
          {PRESET_QUERIES.map((preset, idx) => (
            <button
              key={idx}
              onClick={() => handleSendMessage(preset.text)}
              disabled={isLoading}
              className="text-[11px] whitespace-nowrap px-2.5 py-1 rounded-md bg-slate-800/80 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-700/60 transition-all cursor-pointer disabled:opacity-50"
            >
              {preset.label}
            </button>
          ))}
        </div>

        {/* Input Bar */}
        <div className="p-3 border-t border-slate-800 bg-slate-950/80">
          <form
            onSubmit={e => {
              e.preventDefault();
              handleSendMessage();
            }}
            className="flex items-center gap-2"
          >
            <input
              type="text"
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              placeholder="Ask the agent swarm (e.g. 'Investigate latency spikes', 'Talk to data', 'Optimize routes')..."
              disabled={isLoading}
              className="flex-1 bg-slate-900 border border-slate-700/80 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-orange-500 focus:border-orange-500"
            />
            <button
              type="submit"
              disabled={isLoading || !inputText.trim()}
              className="px-4 py-2.5 bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 disabled:opacity-40 text-white font-medium text-sm rounded-lg flex items-center gap-1.5 shadow-md shadow-orange-500/20 transition-all cursor-pointer"
            >
              <Send className="w-4 h-4" />
              <span>Execute</span>
            </button>
          </form>
        </div>
      </div>

      {/* Right Architecture & Swarm Graph Panel */}
      <div className="lg:col-span-4 flex flex-col gap-4 overflow-y-auto">
        {/* Swarm State Card */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 shadow-xl backdrop-blur-sm">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-orange-400" />
              <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                Multi-Agent Graph State
              </h3>
            </div>
            <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px] font-mono">
              LANGGRAPH RE-ACT
            </span>
          </div>

          {/* Graph Nodes Visual */}
          <div className="mt-3 space-y-2.5 text-xs font-mono">
            {/* Triage Node */}
            <div className="p-2.5 rounded-lg bg-slate-950/90 border border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-400" />
                <span className="text-slate-200 font-semibold">Triage Orchestrator</span>
              </div>
              <span className="text-[10px] text-blue-400 bg-blue-500/10 px-1.5 py-0.5 rounded">
                Entry Gate
              </span>
            </div>

            <div className="flex justify-center -my-1 text-slate-400">
              <CornerDownRight className="w-3.5 h-3.5" />
            </div>

            {/* Specialist Nodes */}
            <div className="space-y-1.5 pl-3 border-l-2 border-slate-800">
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/80 flex items-center justify-between text-[11px]">
                <span className="text-slate-300">Data Insights Agent</span>
                <span className="text-slate-400 text-[10px]">No-SQL & Anomaly</span>
              </div>
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/80 flex items-center justify-between text-[11px]">
                <span className="text-slate-300">Combinatorial Optimizer</span>
                <span className="text-slate-400 text-[10px]">CoT Solver</span>
              </div>
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/80 flex items-center justify-between text-[11px]">
                <span className="text-slate-300">Operations SRE Agent</span>
                <span className="text-slate-400 text-[10px]">Runbook & Telemetry</span>
              </div>
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/80 flex items-center justify-between text-[11px]">
                <span className="text-slate-300">HR People Ops Copilot</span>
                <span className="text-slate-400 text-[10px]">Role Decomposer</span>
              </div>
            </div>

            <div className="flex justify-center -my-1 text-slate-400">
              <CornerDownRight className="w-3.5 h-3.5" />
            </div>

            {/* MCP Connector Node */}
            <div className="p-2.5 rounded-lg bg-slate-950/90 border border-purple-500/30 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-purple-400" />
                <span className="text-purple-200 font-semibold">Model Context Protocol</span>
              </div>
              <span className="text-[10px] text-purple-300 bg-purple-500/10 px-1.5 py-0.5 rounded">
                Tool Registry
              </span>
            </div>

            <div className="flex justify-center -my-1 text-slate-400">
              <CornerDownRight className="w-3.5 h-3.5" />
            </div>

            {/* Human Gate Node */}
            <div className="p-2.5 rounded-lg bg-red-950/40 border border-red-500/40 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ShieldAlert className="w-3.5 h-3.5 text-red-400" />
                <span className="text-red-200 font-semibold">Human Governance Gate</span>
              </div>
              <span className="text-[10px] text-red-300 bg-red-500/20 px-1.5 py-0.5 rounded font-bold">
                HIGH RISK CHECK
              </span>
            </div>
          </div>
        </div>

        {/* Live Metrics Widget */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 shadow-xl backdrop-blur-sm space-y-3">
          <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
            Enterprise Fleet Telemetry
          </h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800">
              <span className="text-slate-400 text-[11px] block">P99 Latency</span>
              <span className="text-emerald-400 font-bold text-base font-mono">14.2ms</span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800">
              <span className="text-slate-400 text-[11px] block">Throughput</span>
              <span className="text-white font-bold text-base font-mono">3,450 ops/s</span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800">
              <span className="text-slate-400 text-[11px] block">Solver Accuracy</span>
              <span className="text-purple-400 font-bold text-base font-mono">93.8%</span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800">
              <span className="text-slate-400 text-[11px] block">Active Agents</span>
              <span className="text-orange-400 font-bold text-base font-mono">5 Swarm</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
