import React, {useState, useEffect} from 'react';
import {
  Users,
  Brain,
  ShieldCheck,
  CheckCircle2,
  TrendingUp,
  Award,
  Sparkles,
  ArrowRight,
  HelpCircle,
} from 'lucide-react';
import type {RoleAnalysis} from '../types/agent';

export const RoleMatrixView: React.FC = () => {
  const [selectedRole, setSelectedRole] = useState('FinTech SRE / DevOps Engineer');
  const [analysis, setAnalysis] = useState<RoleAnalysis | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchRoleAnalysis(selectedRole);
  }, [selectedRole]);

  const fetchRoleAnalysis = async (roleName: string) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/roles/analyze?role=${encodeURIComponent(roleName)}`);
      const data = await res.json();
      setAnalysis(data);
    } catch (err) {
      console.error('Error fetching role analysis:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="p-5 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900 to-slate-950 border border-slate-800 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="px-2 py-0.5 text-[11px] font-bold uppercase rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                Angel One Enterprise Workforce Intelligence
              </span>
              <span className="text-xs text-slate-400">JD Problem: "Human vs. AI Labor Division"</span>
            </div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              Role & Task Autonomy Matrix + Career Insights
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Decomposes enterprise roles into discrete tasks to establish what runs autonomously, what is AI-assisted, and where human judgment is non-negotiable.
            </p>
          </div>

          {/* Role Selector Tabs */}
          <div className="flex items-center gap-2 bg-slate-950 p-1.5 rounded-xl border border-slate-800">
            <button
              onClick={() => setSelectedRole('FinTech SRE / DevOps Engineer')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                selectedRole.includes('SRE')
                  ? 'bg-orange-500 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              FinTech SRE / DevOps
            </button>
            <button
              onClick={() => setSelectedRole('Enterprise People Operations / HR Analyst')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                selectedRole.includes('HR')
                  ? 'bg-orange-500 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              People Operations / HR
            </button>
          </div>
        </div>
      </div>

      {/* Analysis View */}
      {analysis && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Main Task Taxonomy Breakdown (8 cols) */}
          <div className="lg:col-span-8 space-y-6">
            {/* High-level Split Visual Bar */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-xl space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-white">{analysis.role_title}</h3>
                  <p className="text-xs text-slate-400">{analysis.department}</p>
                </div>
                <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">
                  Autonomous Fleet Ready
                </span>
              </div>

              {/* Progress Bar */}
              <div className="space-y-1.5">
                <div className="h-4 w-full rounded-full bg-slate-950 overflow-hidden flex border border-slate-800">
                  <div
                    style={{width: `${analysis.autonomous_ai_pct}%`}}
                    className="bg-emerald-500 h-full transition-all duration-500"
                    title={`Autonomous AI: ${analysis.autonomous_ai_pct}%`}
                  />
                  <div
                    style={{width: `${analysis.collaborative_pct}%`}}
                    className="bg-blue-500 h-full transition-all duration-500"
                    title={`Human-AI Collaborative: ${analysis.collaborative_pct}%`}
                  />
                  <div
                    style={{width: `${analysis.human_judgment_pct}%`}}
                    className="bg-amber-500 h-full transition-all duration-500"
                    title={`Human-Only Judgment: ${analysis.human_judgment_pct}%`}
                  />
                </div>

                <div className="flex items-center justify-between text-xs pt-1">
                  <span className="flex items-center gap-1.5 text-emerald-400 font-semibold">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                    Autonomous AI ({analysis.autonomous_ai_pct}%)
                  </span>
                  <span className="flex items-center gap-1.5 text-blue-400 font-semibold">
                    <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                    Human-AI Collaborative ({analysis.collaborative_pct}%)
                  </span>
                  <span className="flex items-center gap-1.5 text-amber-400 font-semibold">
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
                    Human-Only Judgment ({analysis.human_judgment_pct}%)
                  </span>
                </div>
              </div>
            </div>

            {/* Task Breakdown Cards */}
            <div className="space-y-3">
              <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider px-1">
                Decomposed Task Taxonomy & Explainability Reasonings
              </h4>

              {analysis.tasks.map((task, idx) => (
                <div
                  key={idx}
                  className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 shadow-xl space-y-2 hover:border-slate-700 transition-all"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <h5 className="text-sm font-semibold text-slate-100">{task.task_name}</h5>
                    <span
                      className={`text-xs px-2.5 py-0.5 rounded-md font-semibold self-start sm:self-auto ${
                        task.category.includes('Autonomous')
                          ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/30'
                          : task.category.includes('Collaborative')
                          ? 'bg-blue-500/10 text-blue-300 border border-blue-500/30'
                          : 'bg-amber-500/10 text-amber-300 border border-amber-500/30'
                      }`}
                    >
                      {task.category} ({task.potential_pct}% Automation)
                    </span>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed font-sans">
                    <strong className="text-slate-400">Explainability: </strong>
                    {task.reasoning}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Right Career Insights & Guidance (4 cols) */}
          <div className="lg:col-span-4 space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-xl space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-slate-800">
                <TrendingUp className="w-4 h-4 text-orange-400" />
                <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                  Personal Career Growth Insights
                </h3>
              </div>

              <p className="text-xs text-slate-300 leading-relaxed">
                As autonomous agents absorb routine triage and ticket handling, human professionals ascend into strategic oversight and system orchestration:
              </p>

              <div className="space-y-2.5">
                {analysis.career_advice.map((advice, i) => (
                  <div
                    key={i}
                    className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 flex items-start gap-2.5 text-xs text-slate-200"
                  >
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                    <span className="leading-relaxed">{advice}</span>
                  </div>
                ))}
              </div>

              <div className="p-3 rounded-lg bg-orange-500/10 border border-orange-500/20 text-xs text-orange-200 leading-relaxed">
                💡 <strong>Interview Talking Point</strong>: Demonstrate how this framework protects company compliance while empowering employees with high-leverage creative work.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
