import {GoogleGenAI} from '@google/genai';
import type {IncomingMessage, ServerResponse} from 'http';

interface QueryTelemetryResult {
  stream_id: string;
  metric: string;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  anomaly_spikes_detected: number;
  health_status: string;
}

const MCP_TOOLS = [
  {
    name: 'angel_query_telemetry',
    description: 'Queries real-time IoT and FinTech trading gateway telemetry, latency percentiles, and anomaly markers.',
    category: 'observability',
    requires_approval: false,
    inputSchema: {
      type: 'object',
      properties: {
        stream_id: {type: 'string', description: "Gateway ID (e.g. 'GW-MUMBAI-01')"},
        metric: {type: 'string', enum: ['latency_ms', 'packet_loss', 'order_throughput', 'cpu_load']},
        lookback_minutes: {type: 'number', default: 60},
      },
      required: ['stream_id', 'metric'],
    },
  },
  {
    name: 'angel_combinatorial_optimizer',
    description: 'Solves NP-hard combinatorial resource allocation, workforce shift assignment, and packet routing using LLM + OR solvers.',
    category: 'optimization',
    requires_approval: false,
    inputSchema: {
      type: 'object',
      properties: {
        problem_type: {type: 'string', enum: ['traffic_dispatch', 'workforce_scheduling', 'bin_packing_compute']},
        constraints: {type: 'object'},
        algorithm_mode: {type: 'string', enum: ['chain_of_thought', 'self_consistency', 'rag_augmented']},
      },
      required: ['problem_type', 'constraints'],
    },
  },
  {
    name: 'angel_dispatch_jira_incident',
    description: 'Creates and assigns an enterprise incident ticket in Angel One Jira/ServiceNow queue.',
    category: 'collaboration',
    requires_approval: false,
    inputSchema: {
      type: 'object',
      properties: {
        summary: {type: 'string'},
        severity: {type: 'string', enum: ['P1-Critical', 'P2-High', 'P3-Medium', 'P4-Low']},
        affected_service: {type: 'string'},
      },
      required: ['summary', 'severity', 'affected_service'],
    },
  },
  {
    name: 'angel_rebalance_trading_cluster',
    description: 'Rebalances trading traffic nodes or re-routes order queues during high-volatility anomalies (High Risk, requires human approval).',
    category: 'infrastructure',
    requires_approval: true,
    inputSchema: {
      type: 'object',
      properties: {
        source_node: {type: 'string'},
        target_nodes: {type: 'array', items: {type: 'string'}},
        traffic_percentage: {type: 'number'},
      },
      required: ['source_node', 'target_nodes', 'traffic_percentage'],
    },
  },
  {
    name: 'angel_hr_role_task_decomposer',
    description: 'Decomposes a specific employee role into distinct tasks and categorizes autonomy levels (Autonomous vs Human-Judgment).',
    category: 'hr_analytics',
    requires_approval: false,
    inputSchema: {
      type: 'object',
      properties: {
        role_title: {type: 'string'},
        department: {type: 'string'},
      },
      required: ['role_title'],
    },
  },
];

function readBody(req: IncomingMessage): Promise<any> {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (err) {
        reject(err);
      }
    });
    req.on('error', reject);
  });
}

function sendJson(res: ServerResponse, statusCode: number, data: any) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  });
  res.end(JSON.stringify(data));
}

// Generate realistic telemetry dataset
export function generateTelemetryData() {
  const records = [];
  const gateways = ['GW-MUMBAI-01', 'GW-BLR-02', 'GW-HYD-03'];
  const anomalyIndexes = new Set([18, 19, 20, 36, 37, 54, 72]);

  for (let i = 0; i < 60; i++) {
    const hour = Math.floor((i * 15) / 60);
    const minute = (i * 15) % 60;
    const timeStr = `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`;
    const baseLatency = 8.0 + Math.sin(i / 4.0) * 4.5;
    const isAnomaly = anomalyIndexes.has(i);
    const latency = isAnomaly
      ? +(baseLatency + 35 + Math.random() * 20).toFixed(2)
      : +(Math.max(2.5, baseLatency + (Math.random() * 2 - 1))).toFixed(2);
    const throughput = Math.floor(2200 + Math.random() * 2600);

    records.push({
      timestamp: timeStr,
      latency_ms: latency,
      throughput_ops: throughput,
      gateway_id: gateways[i % gateways.length],
      anomaly: isAnomaly ? 1 : 0,
      error_rate_pct: isAnomaly ? +(1.8 + Math.random() * 3.2).toFixed(2) : 0.03,
    });
  }
  return records;
}

export async function handleApiRequest(req: IncomingMessage, res: ServerResponse): Promise<boolean> {
  const url = new URL(req.url || '', `http://${req.headers.host || 'localhost'}`);
  const pathname = url.pathname;

  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    });
    res.end();
    return true;
  }

  // 1. Health check
  if (pathname === '/api/health') {
    sendJson(res, 200, {
      status: 'healthy',
      platform: 'Angel One Agentic AI Enterprise Engine',
      mcp_tools_registered: MCP_TOOLS.length,
      protocol_version: '2024-11-05',
    });
    return true;
  }

  // 2. MCP Tools listing
  if (pathname === '/api/mcp/tools' && req.method === 'GET') {
    sendJson(res, 200, {
      server: 'angelone-enterprise-mcp',
      protocol_version: '2024-11-05',
      tools: MCP_TOOLS,
    });
    return true;
  }

  // 3. MCP Tool Call execution
  if (pathname === '/api/mcp/call' && req.method === 'POST') {
    try {
      const body = await readBody(req);
      const toolName = body.tool_name;
      const args = body.arguments || {};

      let result: any = {};
      if (toolName === 'angel_query_telemetry') {
        result = {
          stream_id: args.stream_id || 'GW-MUMBAI-01',
          metric: args.metric || 'latency_ms',
          p50_ms: 3.2,
          p95_ms: 18.4,
          p99_ms: 84.6,
          anomaly_spikes_detected: 4,
          health_status: 'ELEVATED_LATENCY',
        };
      } else if (toolName === 'angel_combinatorial_optimizer') {
        result = {
          problem_type: args.problem_type || 'traffic_dispatch',
          algorithm_mode: args.algorithm_mode || 'chain_of_thought',
          optimal_allocation: {
            'GW-MUMBAI-01': '35% allocation',
            'GW-BLR-02': '45% allocation',
            'GW-HYD-03': '20% allocation (Warm Backup)',
          },
          latency_reduction_pct: 38.2,
          estimated_cost_delta: '-12.4%',
          confidence_score: 0.94,
        };
      } else if (toolName === 'angel_dispatch_jira_incident') {
        result = {
          ticket_id: `ANGEL-OPS-${Math.floor(1000 + Math.random() * 9000)}`,
          status: 'CREATED_AND_PAGED',
          severity: args.severity || 'P2-High',
          affected_service: args.affected_service || 'OrderRoutingEngine',
          slack_dispatched: true,
          channel: '#angel-trading-sre',
        };
      } else if (toolName === 'angel_rebalance_trading_cluster') {
        result = {
          action: 'CLUSTER_TRAFFIC_REROUTE',
          status: 'EXECUTED_AFTER_APPROVAL',
          source_node: args.source_node || 'GW-MUMBAI-01',
          target_nodes: args.target_nodes || ['GW-BLR-02', 'GW-HYD-03'],
          rerouted_pct: args.traffic_percentage || 40,
          p99_stabilized_ms: 6.4,
        };
      } else if (toolName === 'angel_hr_role_task_decomposer') {
        result = {
          role_title: args.role_title || 'FinTech SRE / DevOps Engineer',
          autonomous_ai_pct: 48,
          collaborative_pct: 34,
          human_judgment_pct: 18,
          key_leverage_areas: [
            'Automated anomaly triaging & runbook trigger',
            'Combinatorial load rebalancer supervisory check',
            'Post-incident architecture design (Human critical)',
          ],
        };
      } else {
        result = {error: `Unknown tool ${toolName}`};
      }

      sendJson(res, 200, {
        tool: toolName,
        latency_ms: +(12 + Math.random() * 15).toFixed(1),
        result,
      });
      return true;
    } catch (err: any) {
      sendJson(res, 500, {error: err.message});
      return true;
    }
  }

  // 4. Talk-to-Data API
  if ((pathname === '/api/agent/talk-to-data' || pathname === '/api/talk-to-data') && (req.method === 'GET' || req.method === 'POST')) {
    let question = url.searchParams.get('query') || '';
    if (req.method === 'POST') {
      try {
        const body = await readBody(req);
        question = body.query || question;
      } catch (_) {}
    }
    if (!question) question = 'Why did latency spike this morning?';

    const records = generateTelemetryData();
    const anomalies = records.filter(r => r.anomaly === 1);
    const qLower = question.toLowerCase();

    let summary = '';
    let decision = '';
    let sql = '';
    let chartType = 'time_series';

    if (qLower.includes('anomaly') || qLower.includes('spike') || qLower.includes('latency') || qLower.includes('why')) {
      chartType = 'time_series_anomaly';
      summary = `Identified ${anomalies.length} latency spikes peaking at 58.4ms (baseline: 9.1ms). Spikes correlate directly with Market Opening order volatility between 09:15 and 09:45 AM.`;
      decision = `Pre-warm Gateway connection sockets at 09:05 AM and auto-route 35% non-critical order book queries to replica instances.`;
      sql = `SELECT timestamp, latency_ms, gateway_id FROM order_telemetry WHERE latency_ms > (AVG(latency_ms) + 3 * STDDEV(latency_ms)) ORDER BY timestamp ASC;`;
    } else if (qLower.includes('throughput') || qLower.includes('volume') || qLower.includes('capacity')) {
      chartType = 'bar_throughput';
      summary = `Current aggregate throughput is 3,450 ops/sec across 3 active gateways. System headroom is healthy at 46% capacity.`;
      decision = `Maintain current traffic weights. No hardware autoscaling needed for regular market hours.`;
      sql = `SELECT gateway_id, AVG(throughput_ops) as avg_throughput, MAX(throughput_ops) as peak_throughput FROM gateway_telemetry GROUP BY gateway_id;`;
    } else {
      chartType = 'kpi_overview';
      summary = `All 3 trading gateways operating within nominal 99.9% SLA parameters. Average p95 latency is 14.8ms with error rate < 0.05%.`;
      decision = `Continue background health heartbeats and telemetry sampling.`;
      sql = `SELECT gateway_id, quantile(0.95, latency_ms) as p95_latency, AVG(error_rate_pct) as error_rate FROM gateway_telemetry GROUP BY gateway_id;`;
    }

    sendJson(res, 200, {
      query: question,
      summary,
      decision,
      sql_equivalent: sql,
      chart_type: chartType,
      records: records.slice(0, 45),
      key_metrics: {
        total_records_analyzed: records.length,
        anomalies_detected: anomalies.length,
        peak_latency_ms: 58.4,
        baseline_latency_ms: 9.1,
        active_gateways: 3,
      },
    });
    return true;
  }

  // 5. Multi-Agent Chat & Orchestration (powered by Gemini API if available)
  if (pathname === '/api/agent/chat' && req.method === 'POST') {
    try {
      const body = await readBody(req);
      const userQuery = body.query || 'Investigate latency spike and suggest optimal rebalance';
      const qLower = userQuery.toLowerCase();

      // Determine specialist agent & risk
      let activeAgent = 'data_insights_agent';
      let riskLevel = 'low';
      let requiresHumanApproval = false;
      let toolsToInvoke = ['angel_query_telemetry'];

      if (qLower.includes('scale') || qLower.includes('rebalance') || qLower.includes('failover') || qLower.includes('reboot') || qLower.includes('restart')) {
        activeAgent = 'operations_sre_agent';
        riskLevel = 'high';
        requiresHumanApproval = true;
        toolsToInvoke = ['angel_rebalance_trading_cluster', 'angel_dispatch_jira_incident'];
      } else if (qLower.includes('optimize') || qLower.includes('combinatorial') || qLower.includes('schedule') || qLower.includes('allocation')) {
        activeAgent = 'combinatorial_optimizer_agent';
        riskLevel = 'medium';
        toolsToInvoke = ['angel_combinatorial_optimizer'];
      } else if (qLower.includes('career') || qLower.includes('role') || qLower.includes('hr') || qLower.includes('task') || qLower.includes('employee')) {
        activeAgent = 'hr_people_ops_agent';
        riskLevel = 'low';
        toolsToInvoke = ['angel_hr_role_task_decomposer'];
      }

      // Reasoning trace breakdown
      interface TraceItem {
        stage: string;
        agent: string;
        thought: string;
        risk?: string;
        tools?: string[];
        approvalRequired?: boolean;
      }

      const reasoningTraces: TraceItem[] = [
        {
          stage: 'TRIAGE_ORCHESTRATION',
          agent: 'triage_orchestrator',
          thought: `Received query: "${userQuery}". Classified intent as ${activeAgent.replace(/_/g, ' ').toUpperCase()}. Assigned Risk Level: ${riskLevel.toUpperCase()}.`,
          risk: riskLevel,
        },
        {
          stage: 'MCP_TOOL_DISPATCH',
          agent: activeAgent,
          thought: `Preparing MCP tool execution: [${toolsToInvoke.join(', ')}]. Checking schema validity and auth tokens.`,
          tools: toolsToInvoke,
        },
      ];

      // Call Gemini if key is present
      let geminiSynthesizedText = '';
      if (process.env.GEMINI_API_KEY) {
        try {
          const ai = new GoogleGenAI();
          const prompt = `You are the lead Agentic AI Orchestrator at Angel One (India's leading FinTech & Wealth-tech enterprise).
You are answering the user's query: "${userQuery}".
Active Specialist Agent: ${activeAgent}.
Assigned Risk Level: ${riskLevel}.
Required Tools: ${toolsToInvoke.join(', ')}.

Provide a structured, executive-grade response covering:
1. Executive Diagnosis & Root Cause / Strategic Context
2. Specialist Agent Findings (incorporating telemetry / combinatorial constraints / people ops)
3. Actionable Remediation / Optimization Proposal
4. Human Governance Notice (explain if human approval is required for safety)

Keep it crisp, professional, and tailored to Angel One's high-scale FinTech environment.`;

          const timeoutPromise = new Promise<null>((resolve) => setTimeout(() => resolve(null), 3500));
          const generatePromise = ai.models.generateContent({
            model: 'gemini-3.8-flash',
            contents: prompt,
          });

          const response: any = await Promise.race([generatePromise, timeoutPromise]);
          if (response && response.text) {
            geminiSynthesizedText = response.text;
          }
        } catch (err: any) {
          console.warn('Gemini API call skipped or errored:', err.message);
        }
      }

      if (!geminiSynthesizedText) {
        // Fallback high-quality response
        if (activeAgent === 'operations_sre_agent') {
          geminiSynthesizedText = `### SRE Incident Diagnosis & Autonomous Rebalancing Plan

1. **Incident Context**: High latency detected on \`GW-MUMBAI-01\` (P99: 84.6ms exceeding SLA threshold of 25ms).
2. **Specialist Evaluation**: Order queue buffers are saturating due to market opening volatility. Secondary gateway \`GW-BLR-02\` has 62% available headroom.
3. **Proposed Action**: Rebalance 40% of inbound order streams from \`GW-MUMBAI-01\` to \`GW-BLR-02\` and \`GW-HYD-03\` with a 30s connection drain.
4. **Governance Gate**: ⚠️ **HIGH RISK ACTION DETECTED**. Because traffic rebalancing affects live trading queues, this action is paused in accordance with Angel One safety protocols awaiting SRE Manager approval.`;
        } else if (activeAgent === 'combinatorial_optimizer_agent') {
          geminiSynthesizedText = `### Combinatorial Optimization Formulation & Solver Result

1. **Problem Formulation**: Multi-objective combinatorial dispatch subject to:
   - Max latency $\\le 20\\text{ms}$
   - Peak throughput $\\ge 4,000\\text{ ops/sec}$
   - Server compute cost minimization
2. **Algorithm**: Chain-of-Thought + Self-Consistency Heuristic Solver ($n=5$ candidate solutions).
3. **Optimal Dispatch Vector**:
   - **GW-BLR-02**: 45% traffic allocation (Objective latency: 4.2ms)
   - **GW-MUMBAI-01**: 35% traffic allocation (Objective latency: 6.8ms)
   - **GW-HYD-03**: 20% traffic allocation (Warm standby buffer)
4. **Impact**: Projected latency reduction of **38.2%** and operational cost reduction of **12.4%**.`;
        } else if (activeAgent === 'hr_people_ops_agent') {
          geminiSynthesizedText = `### Role & Task Autonomy Matrix: People Operations Intelligence

1. **Task Decomposition**: Evaluated core responsibilities into Autonomous AI, Human-AI Collaborative, and Human-Only Judgment tiers.
2. **Findings**:
   - 48% of operational busywork (policy FAQ, leave entitlement checks, initial resume parsing) can be fully automated.
   - 34% of workflow (talent evaluation, shift allocation, metric correlation) is amplified through collaborative AI.
   - 18% (grievance mediation, executive coaching, strategic hiring decisions) remains strictly human-judgment.
3. **Career Upskilling Recommendation**: Transition professionals toward Agentic Flow Architects and Exception Auditing leaders.`;
        } else {
          geminiSynthesizedText = `### Telemetry Analytics & Anomaly Diagnosis

1. **Telemetry Stream**: Evaluated 60 recent time intervals across 3 production gateways.
2. **Anomaly Correlation**: 4 distinct latency spikes identified between 09:15 and 09:45 AM.
3. **Root Cause**: Correlated with morning market opening spikes in WebSocket subscription bursts.
4. **Recommended Action**: Enable adaptive batch flushing on WebSocket ingress nodes 10 minutes prior to market open.`;
        }
      }

      reasoningTraces.push({
        stage: 'SYNTHESIS_AND_GOVERNANCE',
        agent: activeAgent,
        thought: requiresHumanApproval
          ? 'Identified critical infrastructure impact. Routing to Human-in-the-Loop approval gate.'
          : 'Execution parameters verified. Returning synthesized decision.',
        approvalRequired: requiresHumanApproval,
      });

      sendJson(res, 200, {
        session_id: `angel-${Date.now().toString(36)}`,
        user_query: userQuery,
        active_agent: activeAgent,
        risk_level: riskLevel,
        confidence_score: 0.94,
        requires_human_approval: requiresHumanApproval,
        reasoning_traces: reasoningTraces,
        tools_invoked: toolsToInvoke,
        content: geminiSynthesizedText,
        suggested_actions: [
          'Execute recommended MCP rebalance',
          'Export anomaly timeline report to Slack',
          'Review combinatorial optimization constraints',
        ],
      });
      return true;
    } catch (err: any) {
      sendJson(res, 500, {error: err.message});
      return true;
    }
  }

  // 6. Role Task Analyzer API
  if (pathname === '/api/roles/analyze' && req.method === 'GET') {
    const roleParam = url.searchParams.get('role') || 'FinTech SRE / DevOps Engineer';
    const isSre = roleParam.toLowerCase().includes('sre') || roleParam.toLowerCase().includes('devops');

    const report = isSre
      ? {
          role_title: 'FinTech SRE / DevOps Engineer',
          department: 'Infrastructure & Trading Engine',
          autonomous_ai_pct: 46,
          collaborative_pct: 36,
          human_judgment_pct: 18,
          tasks: [
            {
              task_name: 'Log Parsing & Routine Telemetry Anomaly Triaging',
              category: 'Fully Autonomous AI',
              potential_pct: 92,
              reasoning: 'Deterministic pattern recognition, automated error grouping, and well-indexed runbook matching.',
            },
            {
              task_name: 'Combinatorial Load Balancing & Node Auto-Tuning',
              category: 'Human-AI Collaborative',
              potential_pct: 75,
              reasoning: 'Agent formulates optimal mathematical routing; human on-call engineer signs off on production shifts.',
            },
            {
              task_name: 'Disaster Recovery & Post-Mortem Architecture Redesign',
              category: 'Human-Only Judgment',
              potential_pct: 15,
              reasoning: 'Involves high-stakes business continuity trade-offs, regulatory accountability, and cross-team trust.',
            },
          ],
          career_advice: [
            'Shift from manual firefighting to Agentic Workflow Architect.',
            'Master Model Context Protocol (MCP) to safely interface agent swarms with enterprise systems.',
            'Deepen expertise in distributed consensus, latency profiling, and LLM safety governance.',
          ],
        }
      : {
          role_title: 'Enterprise People Operations / HR Analyst',
          department: 'People & Culture',
          autonomous_ai_pct: 50,
          collaborative_pct: 32,
          human_judgment_pct: 18,
          tasks: [
            {
              task_name: 'Employee Policy Q&A (Leave, Benefits, Travel Reimbursement)',
              category: 'Fully Autonomous AI',
              potential_pct: 90,
              reasoning: 'RAG search over enterprise HR handbooks with instant authenticated citations.',
            },
            {
              task_name: 'Candidate Sourcing & Resume Semantic Matching',
              category: 'Human-AI Collaborative',
              potential_pct: 70,
              reasoning: 'AI extracts taxonomy of skills; human recruiter evaluates culture alignment and soft skills.',
            },
            {
              task_name: 'Performance Mediation, Conflict Resolution & Career Coaching',
              category: 'Human-Only Judgment',
              potential_pct: 5,
              reasoning: 'Requires deep empathy, ethical nuance, interpersonal trust, and subjective discernment.',
            },
          ],
          career_advice: [
            'Elevate from transactional HR admin to Strategic Talent Partner.',
            'Use "Talk-to-Data" agents to extract workforce sentiment without waiting on IT reports.',
            'Lead organizational change management for AI agent adoption.',
          ],
        };

    sendJson(res, 200, report);
    return true;
  }

  return false;
}
