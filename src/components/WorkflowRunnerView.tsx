import React, {useState} from 'react';
import {
  Workflow,
  Play,
  CheckCircle2,
  AlertOctagon,
  ShieldAlert,
  ArrowRight,
  RefreshCw,
  Cpu,
  Clock,
  Sparkles,
  Zap,
} from 'lucide-react';

interface WorkflowStep {
  id: number;
  title: string;
  agent: string;
  status: 'pending' | 'running' | 'completed' | 'waiting_approval';
  details: string;
  output?: any;
}

export const WorkflowRunnerView: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [humanApprovalGiven, setHumanApprovalGiven] = useState<boolean | null>(null);

  const [steps, setSteps] = useState<WorkflowStep[]>([
    {
      id: 1,
      title: 'Real-Time Telemetry Anomaly Detection',
      agent: 'observability_sensor_agent',
      status: 'pending',
      details: 'Continuously monitors order gateway latency. Flags P99 spikes exceeding 25ms threshold.',
    },
    {
      id: 2,
      title: 'Multi-Agent Root Cause Synthesis',
      agent: 'triage_orchestrator',
      status: 'pending',
      details: 'Correlates anomaly with 09:15 AM market open auction burst. Verifies secondary gateway headroom.',
    },
    {
      id: 3,
      title: 'Combinatorial Optimization Solver',
      agent: 'combinatorial_optimizer',
      status: 'pending',
      details: 'Calculates optimal multi-gateway traffic split vector to minimize p99 latency without server overload.',
    },
    {
      id: 4,
      title: 'Human-in-the-Loop (HITL) Governance Gate',
      agent: 'human_escalation_gate',
      status: 'pending',
      details: 'High-risk action safeguard: Requires SRE on-call manager confirmation before altering live production traffic.',
    },
    {
      id: 5,
      title: 'MCP Execution & Health Verification',
      agent: 'operations_sre_agent',
      status: 'pending',
      details: 'Invokes `angel_rebalance_trading_cluster` via Model Context Protocol and verifies SLA recovery.',
    },
  ]);

  const startWorkflow = () => {
    setIsRunning(true);
    setCurrentStepIndex(0);
    setHumanApprovalGiven(null);

    // Reset steps
    setSteps(prev =>
      prev.map((s, i) => ({
        ...s,
        status: i === 0 ? 'running' : 'pending',
      }))
    );

    // Step 1
    setTimeout(() => {
      setSteps(prev =>
        prev.map((s, i) =>
          i === 0
            ? {
                ...s,
                status: 'completed',
                output: 'ALERT: GW-MUMBAI-01 latency spiked to 84.6ms (Threshold: 25ms). 4,200 ops/s queue delay.',
              }
            : i === 1
            ? {...s, status: 'running'}
            : s
        )
      );
      setCurrentStepIndex(1);

      // Step 2
      setTimeout(() => {
        setSteps(prev =>
          prev.map((s, i) =>
            i === 1
              ? {
                  ...s,
                  status: 'completed',
                  output: 'Diagnosis: Morning auction volatility saturation. Secondary node GW-BLR-02 has 62% available headroom.',
                }
              : i === 2
              ? {...s, status: 'running'}
              : s
          )
        );
        setCurrentStepIndex(2);

        // Step 3
        setTimeout(() => {
          setSteps(prev =>
            prev.map((s, i) =>
              i === 2
                ? {
                    ...s,
                    status: 'completed',
                    output: 'Optimal Vector Formulated: Shift 40% volume from GW-MUMBAI-01 to GW-BLR-02. Expected latency drop: -38.2%.',
                  }
                : i === 3
                ? {...s, status: 'waiting_approval'}
                : s
            )
          );
          setCurrentStepIndex(3);
        }, 1200);
      }, 1200);
    }, 1200);
  };

  const handleApprove = (approved: boolean) => {
    setHumanApprovalGiven(approved);
    if (approved) {
      setSteps(prev =>
        prev.map((s, i) =>
          i === 3
            ? {
                ...s,
                status: 'completed',
                output: 'Approved by Angel One SRE Manager. Authorizing MCP traffic reroute.',
              }
            : i === 4
            ? {...s, status: 'running'}
            : s
        )
      );
      setCurrentStepIndex(4);

      // Step 5 execution
      setTimeout(() => {
        setSteps(prev =>
          prev.map((s, i) =>
            i === 4
              ? {
                  ...s,
                  status: 'completed',
                  output: 'MCP Tool executed successfully. P99 latency restored to 8.2ms. SLA: 100% nominal.',
                }
              : s
          )
        );
        setIsRunning(false);
      }, 1500);
    } else {
      setSteps(prev =>
        prev.map((s, i) =>
          i === 3
            ? {
                ...s,
                status: 'completed',
                output: 'Action denied by Human Operator. Reroute aborted. Traffic retained on current gateway.',
              }
            : s
        )
      );
      setIsRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="p-5 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900 to-slate-950 border border-slate-800 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="px-2 py-0.5 text-[11px] font-bold uppercase rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Autonomous Incident Remediation
              </span>
              <span className="text-xs text-slate-400">JD Problem: "Automate Busywork, Keep Human Judgment"</span>
            </div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              End-to-End Agentic Workflow Runner
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Watch an end-to-end multi-agent loop handle an incident in real-time, compute optimal parameters, halt at the Human-in-the-Loop gate, and dispatch via MCP.
            </p>
          </div>

          <button
            onClick={startWorkflow}
            disabled={isRunning && steps[3].status !== 'waiting_approval'}
            className="px-5 py-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-50 text-white font-semibold text-xs rounded-xl shadow-lg transition-all cursor-pointer flex items-center gap-2 self-start md:self-auto"
          >
            {isRunning ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4 fill-white" />
            )}
            <span>Run Incident Simulation</span>
          </button>
        </div>
      </div>

      {/* Stepper Timeline */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 shadow-xl space-y-6">
        <div className="space-y-6">
          {steps.map((step, idx) => {
            const isDone = step.status === 'completed';
            const isCurrent = step.status === 'running';
            const isWaiting = step.status === 'waiting_approval';

            return (
              <div key={step.id} className="relative flex items-start gap-4">
                {/* Connecting Line */}
                {idx < steps.length - 1 && (
                  <div
                    className={`absolute left-5 top-10 w-0.5 h-14 ${
                      isDone ? 'bg-emerald-500' : 'bg-slate-800'
                    }`}
                  />
                )}

                {/* Step Icon Badge */}
                <div
                  className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 z-10 font-bold text-xs transition-all ${
                    isDone
                      ? 'bg-emerald-500 text-slate-950 shadow-lg shadow-emerald-500/20'
                      : isCurrent
                      ? 'bg-orange-500 text-white animate-pulse shadow-lg shadow-orange-500/30'
                      : isWaiting
                      ? 'bg-red-500 text-white animate-bounce shadow-lg shadow-red-500/30'
                      : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {isDone ? (
                    <CheckCircle2 className="w-5 h-5" />
                  ) : isWaiting ? (
                    <ShieldAlert className="w-5 h-5" />
                  ) : (
                    <span>{step.id}</span>
                  )}
                </div>

                {/* Step Body */}
                <div className="flex-1 bg-slate-950/70 border border-slate-800/80 rounded-xl p-4 space-y-2">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
                    <h4 className="text-sm font-bold text-slate-100">{step.title}</h4>
                    <span className="text-[11px] font-mono text-orange-400 bg-orange-500/10 px-2 py-0.5 rounded self-start sm:self-auto">
                      @{step.agent}
                    </span>
                  </div>

                  <p className="text-xs text-slate-400 leading-relaxed">{step.details}</p>

                  {/* Output Log */}
                  {step.output && (
                    <div className="mt-2 p-2.5 rounded bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300">
                      {step.output}
                    </div>
                  )}

                  {/* Interactive Approval Gate Modal */}
                  {isWaiting && (
                    <div className="mt-4 p-4 rounded-xl bg-red-950/40 border border-red-500/40 space-y-3">
                      <div className="flex items-center gap-2 text-red-400 text-xs font-bold">
                        <ShieldAlert className="w-4 h-4" />
                        <span>HUMAN APPROVAL REQUIRED BEFORE MCP REBALANCE DISPATCH</span>
                      </div>
                      <p className="text-xs text-slate-300">
                        The agent has prepared a 40% traffic reroute from <code className="text-orange-300">GW-MUMBAI-01</code> to <code className="text-emerald-300">GW-BLR-02</code>. SRE Manager approval is required to proceed.
                      </p>
                      <div className="flex items-center gap-3 pt-1">
                        <button
                          onClick={() => handleApprove(true)}
                          className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs shadow-md cursor-pointer flex items-center gap-1.5"
                        >
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          <span>Approve Reroute</span>
                        </button>
                        <button
                          onClick={() => handleApprove(false)}
                          className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-xs cursor-pointer"
                        >
                          Deny / Retain Traffic
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
