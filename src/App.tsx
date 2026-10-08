import React, {useState} from 'react';
import {Header} from './components/Header';
import {AgentWorkspace} from './components/AgentWorkspace';
import {TalkToDataView} from './components/TalkToDataView';
import {McpRegistryView} from './components/McpRegistryView';
import {RoleMatrixView} from './components/RoleMatrixView';
import {WorkflowRunnerView} from './components/WorkflowRunnerView';
import {InterviewDeckModal} from './components/InterviewDeckModal';
import type {AgentTab} from './types/agent';

export default function App() {
  const [currentTab, setCurrentTab] = useState<AgentTab>('workspace');
  const [isInterviewDeckOpen, setIsInterviewDeckOpen] = useState(false);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-orange-500 selection:text-white">
      {/* App Header */}
      <Header
        currentTab={currentTab}
        onTabChange={setCurrentTab}
        onOpenInterviewDeck={() => setIsInterviewDeckOpen(true)}
      />

      {/* Main Body Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {currentTab === 'workspace' && <AgentWorkspace />}
        {currentTab === 'talk-to-data' && <TalkToDataView />}
        {currentTab === 'mcp-registry' && <McpRegistryView />}
        {currentTab === 'role-matrix' && <RoleMatrixView />}
        {currentTab === 'workflow-runner' && <WorkflowRunnerView />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/90 py-4 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <p>
            OptiLLM Enterprise • Powered by Multi-Agent Swarm, MCP & Gemini 3.8 Flash
          </p>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsInterviewDeckOpen(true)}
              className="text-orange-400 hover:text-orange-300 font-semibold cursor-pointer underline"
            >
              Open Interview Dossier
            </button>
            <span>•</span>
            <span className="font-mono text-slate-400">Spec: v2.0-Production</span>
          </div>
        </div>
      </footer>

      {/* Interview Talking Points Modal */}
      <InterviewDeckModal
        isOpen={isInterviewDeckOpen}
        onClose={() => setIsInterviewDeckOpen(false)}
      />
    </div>
  );
}
