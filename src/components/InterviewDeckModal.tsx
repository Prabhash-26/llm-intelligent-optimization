import React from 'react';
import {
  X,
  Award,
  CheckCircle2,
  Sparkles,
  Zap,
  ShieldCheck,
  Code2,
  Cpu,
  Layers,
  ExternalLink,
} from 'lucide-react';

interface InterviewDeckModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const InterviewDeckModal: React.FC<InterviewDeckModalProps> = ({isOpen, onClose}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md overflow-y-auto">
      <div className="relative w-full max-w-4xl bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl p-6 sm:p-8 space-y-6 max-h-[90vh] overflow-y-auto text-slate-200">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-orange-500/20 border border-orange-500/30 flex items-center justify-center text-orange-400">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">
                OptiLLM Enterprise — Angel One Interview Dossier & Talking Points
              </h2>
              <p className="text-xs text-slate-400">
                Strategic guide to acing your interview round with this project
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* 1. Elevator Pitch */}
        <div className="p-4 rounded-xl bg-gradient-to-r from-orange-500/10 via-amber-500/10 to-transparent border border-orange-500/30 space-y-2">
          <div className="flex items-center gap-2 text-xs font-bold text-orange-400 uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5" />
            <span>30-Second Elevator Pitch for Your Interview</span>
          </div>
          <p className="text-xs text-slate-200 leading-relaxed font-sans">
            "I built <strong>OptiLLM Enterprise</strong>, an end-to-end Agentic AI system designed specifically for high-concurrency environments like Angel One. It unites a <strong>Multi-Agent Swarm (Triage, SRE, HR, and Data agents)</strong> with the <strong>Model Context Protocol (MCP)</strong> for standard tool invocation, a <strong>'Talk-to-Data' plain-English telemetry engine</strong>, and a <strong>Combinatorial Optimization Solver</strong> backed by Chain-of-Thought heuristics. Crucially, I engineered <strong>Human-in-the-Loop (HITL) risk gates</strong> so the agents automate busywork autonomously while keeping human judgment strictly in control of critical trading infrastructure."
          </p>
        </div>

        {/* 2. Direct JD Problem Matrix */}
        <div className="space-y-3">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
            How This Project Answers Every Angel One JD Problem
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
            <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800/90 space-y-1.5">
              <span className="font-bold text-white flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                "Every employee's go-to colleague"
              </span>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Solved via a multi-agent swarm architecture where queries are triaged dynamically, delegated to specialist agents (SRE, Data, HR), and escalated safely when out-of-domain.
              </p>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800/90 space-y-1.5">
              <span className="font-bold text-white flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                "Talk to data without SQL"
              </span>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Solved via our <code>TalkToDataAgent</code> that translates natural English into semantic queries, runs statistical anomaly detection (Z-scores/rolling metrics), and generates dynamic interactive charts with executive action recommendations.
              </p>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800/90 space-y-1.5">
              <span className="font-bold text-white flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                "Plugging into the real world with MCP"
              </span>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Implemented a compliant <strong>Model Context Protocol (MCP)</strong> server with schema contracts for querying telemetry, running combinatorial optimizations, dispatching Jira incidents, and rebalancing cluster nodes.
              </p>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800/90 space-y-1.5">
              <span className="font-bold text-white flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                "What work humans do vs AI"
              </span>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Solved via our <strong>Role & Task Autonomy Matrix</strong>, providing explainable breakdowns of enterprise jobs (SRE, FinTech Operations, HR) into % Autonomous, % Collaborative, and % Human-Only Judgment.
              </p>
            </div>
          </div>
        </div>

        {/* 3. Deep-Dive Q&A to Wow the Interviewer */}
        <div className="space-y-3">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
            High-Impact Technical Interview Questions & Answers
          </h3>

          <div className="space-y-3 text-xs">
            <div className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 space-y-1">
              <p className="font-semibold text-orange-300">
                Q: Why did you choose Model Context Protocol (MCP) instead of simple REST endpoints?
              </p>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                "MCP decouples the LLM agent client from the underlying enterprise systems. By standardizing tool definitions, capabilities, and execution sandboxes under MCP schema specs, agents can discover tools dynamically, enforce authorization policies, and emit standardized telemetry without tight coupling to proprietary APIs."
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 space-y-1">
              <p className="font-semibold text-orange-300">
                Q: In a financial enterprise like Angel One, how do you prevent agent hallucinations from breaking production?
              </p>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                "I implemented a 3-layer safeguard: (1) Deterministic validation of tool outputs before action dispatch; (2) An automated Risk Classification matrix where read operations are autonomous, but high-impact write operations (like cluster rebalancing) pause at a mandatory Human-in-the-Loop gate; (3) Chain-of-Thought with self-consistency to verify reasoning consistency."
              </p>
            </div>
          </div>
        </div>

        {/* Close Button */}
        <div className="flex justify-end pt-2 border-t border-slate-800">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-orange-500 hover:bg-orange-600 text-white font-semibold text-xs transition-colors cursor-pointer"
          >
            Got it, ready to demo!
          </button>
        </div>
      </div>
    </div>
  );
};
