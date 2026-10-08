import React from 'react';
import {
  Bot,
  Layers,
  Activity,
  Cpu,
  ShieldCheck,
  Sparkles,
  BarChart3,
  Plug,
  Users,
  Workflow,
  HelpCircle,
} from 'lucide-react';
import type {AgentTab} from '../types/agent';

interface HeaderProps {
  currentTab: AgentTab;
  onTabChange: (tab: AgentTab) => void;
  onOpenInterviewDeck: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  onTabChange,
  onOpenInterviewDeck,
}) => {
  const tabs = [
    {id: 'workspace' as AgentTab, label: 'Multi-Agent Workspace', icon: Bot, badge: 'Swarm'},
    {id: 'talk-to-data' as AgentTab, label: 'Talk to Data', icon: BarChart3, badge: 'No-SQL'},
    {id: 'mcp-registry' as AgentTab, label: 'MCP Tools', icon: Plug, badge: 'v2024-11'},
    {id: 'role-matrix' as AgentTab, label: 'Role & Task Matrix', icon: Users, badge: 'HR Intel'},
    {id: 'workflow-runner' as AgentTab, label: 'Autonomous Workflow', icon: Workflow, badge: 'HITL'},
  ];

  return (
    <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-0 z-40">
      {/* Top Banner */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex flex-wrap items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-orange-500 via-amber-500 to-blue-600 shadow-lg shadow-orange-500/20">
            <Cpu className="w-5 h-5 text-white" />
            <span className="absolute -top-1 -right-1 flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold tracking-tight text-white text-lg">
                Opti<span className="text-orange-500">LLM</span>
              </span>
              <span className="px-2 py-0.5 text-[10px] font-bold tracking-wide uppercase bg-orange-500/10 text-orange-400 border border-orange-500/20 rounded-md">
                ENTERPRISE
              </span>
              <span className="hidden sm:inline-block px-2 py-0.5 text-[9px] font-medium tracking-wide uppercase bg-slate-800 text-slate-300 rounded">
                LLM Architecture
              </span>
            </div>
            <p className="text-xs text-slate-400 font-medium">
              Autonomous Multi-Agent Swarm, MCP Tool Orchestration & Combinatorial Solvers
            </p>
          </div>
        </div>

        {/* Live System Indicators */}
        <div className="flex items-center flex-wrap gap-2 sm:gap-3 text-xs">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-300">
            <Sparkles className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-slate-400">LLM:</span>
            <span className="font-semibold text-purple-300">Gemini 3.8 Flash</span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-300">
            <Plug className="w-3.5 h-3.5 text-blue-400" />
            <span className="text-slate-400">MCP:</span>
            <span className="font-semibold text-emerald-400">Connected (5 Tools)</span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-300">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-slate-400">HITL Gate:</span>
            <span className="font-semibold text-emerald-300">Enforced</span>
          </div>

          <button
            onClick={onOpenInterviewDeck}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white font-medium text-xs shadow-md shadow-orange-500/20 transition-all cursor-pointer"
          >
            <HelpCircle className="w-3.5 h-3.5" />
            <span>Interview Dossier</span>
          </button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 border-t border-slate-900/60 flex space-x-1 overflow-x-auto no-scrollbar">
        {tabs.map(tab => {
          const Icon = tab.icon;
          const isActive = currentTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`flex items-center gap-2 px-3 py-2.5 text-xs font-semibold whitespace-nowrap border-b-2 transition-all cursor-pointer ${
                isActive
                  ? 'border-orange-500 text-white bg-slate-900/40'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? 'text-orange-400' : 'text-slate-400'}`} />
              <span>{tab.label}</span>
              <span
                className={`text-[10px] px-1.5 py-0.2 rounded font-mono ${
                  isActive
                    ? 'bg-orange-500/20 text-orange-300'
                    : 'bg-slate-800 text-slate-400'
                }`}
              >
                {tab.badge}
              </span>
            </button>
          );
        })}
      </div>
    </header>
  );
};
