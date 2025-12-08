# LLM-Based Intelligent Optimization & Data Analytics System

> GPT-based LLMs for combinatorial optimization + real-time IoT predictive maintenance pipeline with Plotly Dash dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)
![Dash](https://img.shields.io/badge/Dashboard-Plotly%20Dash-purple)

## Quick Start
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here

# Run IoT pipeline
python iot/pipeline.py

# Launch dashboard
python dashboard/app.py
# Open http://localhost:8050
```

## Results

| Method | Optimization Score |
|---|---|
| Zero-Shot GPT-4 | 71.3% |
| Few-Shot GPT-4 | 84.7% |
| CoT + Self-Consistency | 91.2% |
| RAG-augmented | 93.8% |

## Author
Prabhash S — VIT Vellore | IIT (BHU) Varanasi
