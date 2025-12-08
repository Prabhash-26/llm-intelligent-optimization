"""
Interactive Plotly Dash Dashboard
Author: Prabhash S

Real-time KPI tracking for IoT analytics and LLM optimization results.
Run: python dashboard/app.py → open http://localhost:8050
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

np.random.seed(42)
n          = 200
timestamps = pd.date_range("2025-01-01", periods=n, freq="1h")
values     = np.sin(np.linspace(0, 10, n)) * 40 + 100 + np.random.normal(0, 5, n)
anomalies  = np.zeros(n)
anomalies[np.random.choice(n, size=10, replace=False)] = 1

df = pd.DataFrame({"timestamp": timestamps, "value": values, "anomaly": anomalies})

model_df = pd.DataFrame({
    "Method" : ["Zero-Shot", "Few-Shot", "CoT", "RAG"],
    "Score"  : [71.3, 84.7, 91.2, 93.8],
})

app.layout = html.Div([
    html.H1("LLM Optimization & IoT Analytics Dashboard",
            style={"textAlign": "center", "fontFamily": "Arial",
                   "color": "#2c3e50", "padding": "20px"}),
    html.Div([
        html.Div([
            html.H3("📡 IoT Sensor Stream", style={"color": "#2980b9"}),
            dcc.Graph(id="sensor-graph"),
        ], style={"width": "60%", "display": "inline-block", "padding": "10px"}),
        html.Div([
            html.H3("🤖 LLM Method Comparison", style={"color": "#27ae60"}),
            dcc.Graph(id="model-graph"),
        ], style={"width": "38%", "display": "inline-block", "padding": "10px"}),
    ]),
    html.Div([
        html.H3("KPI Summary", style={"color": "#8e44ad", "textAlign": "center"}),
        html.Div(id="kpi-cards",
                 style={"display": "flex", "justifyContent": "center", "gap": "20px"}),
    ]),
    dcc.Interval(id="interval", interval=3000, n_intervals=0),
], style={"backgroundColor": "#f8f9fa", "padding": "20px"})


@app.callback(Output("sensor-graph", "figure"), Input("interval", "n_intervals"))
def update_sensor(_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["value"],
                             mode="lines", name="Sensor",
                             line=dict(color="#3498db", width=1.5)))
    adf = df[df["anomaly"] == 1]
    fig.add_trace(go.Scatter(x=adf["timestamp"], y=adf["value"],
                             mode="markers", name="Anomaly",
                             marker=dict(color="red", size=8, symbol="x")))
    fig.update_layout(plot_bgcolor="#ffffff",
                      margin=dict(l=40, r=20, t=20, b=40))
    return fig


@app.callback(Output("model-graph", "figure"), Input("interval", "n_intervals"))
def update_model(_):
    fig = px.bar(model_df, x="Method", y="Score",
                 color="Score", color_continuous_scale="Greens", text="Score")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis=dict(range=[0, 100]),
                      plot_bgcolor="#ffffff",
                      margin=dict(l=20, r=20, t=20, b=40),
                      coloraxis_showscale=False)
    return fig


@app.callback(Output("kpi-cards", "children"), Input("interval", "n_intervals"))
def update_kpis(_):
    kpis = [("🔴 Anomalies", str(int(anomalies.sum()))),
            ("✅ Best LLM", "93.8%"),
            ("⚡ Avg Latency", "3.6s"),
            ("📈 F1 Score", "0.89")]
    return [html.Div([
        html.H2(v, style={"margin": "0", "color": "#2c3e50"}),
        html.P(k, style={"margin": "5px 0 0", "color": "#7f8c8d", "fontSize": "13px"}),
    ], style={"background": "#fff", "padding": "20px 30px", "borderRadius": "10px",
              "boxShadow": "0 2px 8px rgba(0,0,0,0.1)", "textAlign": "center",
              "minWidth": "150px"}) for k, v in kpis]


if __name__ == "__main__":
    app.run(debug=True)
