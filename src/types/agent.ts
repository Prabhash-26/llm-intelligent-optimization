export type AgentTab =
  | 'workspace'
  | 'talk-to-data'
  | 'mcp-registry'
  | 'role-matrix'
  | 'workflow-runner'
  | 'architecture-deck';

export interface ReasoningTrace {
  stage: string;
  agent: string;
  thought: string;
  risk?: string;
  tools?: string[];
  approvalRequired?: boolean;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'agent';
  agentName?: string;
  content: string;
  timestamp: string;
  confidence?: number;
  riskLevel?: 'low' | 'medium' | 'high' | 'critical';
  requiresHumanApproval?: boolean;
  humanApproved?: boolean | null;
  reasoningTraces?: ReasoningTrace[];
  toolsInvoked?: string[];
  suggestedActions?: string[];
}

export interface TelemetryPoint {
  timestamp: string;
  latency_ms: number;
  throughput_ops: number;
  gateway_id: string;
  anomaly: number;
  error_rate_pct: number;
}

export interface TalkToDataResult {
  query: string;
  summary: string;
  decision: string;
  sql_equivalent: string;
  chart_type: string;
  records: TelemetryPoint[];
  key_metrics: {
    total_records_analyzed: number;
    anomalies_detected: number;
    peak_latency_ms: number;
    baseline_latency_ms: number;
    active_gateways: number;
  };
}

export interface MCPToolDef {
  name: string;
  description: string;
  category: string;
  requires_approval: boolean;
  inputSchema: {
    type: string;
    properties: Record<string, any>;
    required?: string[];
  };
}

export interface TaskEvaluation {
  task_name: string;
  category: string;
  potential_pct: number;
  reasoning: string;
  risk_if_unattended?: string;
}

export interface RoleAnalysis {
  role_title: string;
  department: string;
  autonomous_ai_pct: number;
  collaborative_pct: number;
  human_judgment_pct: number;
  tasks: TaskEvaluation[];
  career_advice: string[];
}
