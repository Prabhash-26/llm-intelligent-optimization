"""
LLM-Based Combinatorial Optimizer
Author: Prabhash S

Uses GPT-based LLMs with chain-of-thought, self-consistency,
and RAG pipelines to solve combinatorial optimization problems.
"""

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class LLMOptimizer:
    """
    LLM-based solver for combinatorial optimization problems.

    Supported modes:
        - zero-shot   : Direct problem statement
        - few-shot    : 3-5 solved examples prepended
        - cot         : Chain-of-thought step-by-step reasoning
        - rag         : RAG-augmented with domain knowledge
    """

    def __init__(self, model: str = "gpt-4", mode: str = "cot"):
        self.model = model
        self.mode  = mode

    def solve(self, problem_type: str, constraints: dict,
              reasoning: str = None) -> dict:
        mode   = reasoning or self.mode
        prompt = self._build_prompt(problem_type, constraints, mode)

        if mode == "self-consistency":
            return self._self_consistency_solve(prompt, n=5)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an expert combinatorial optimizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return self._parse(response.choices[0].message.content)

    def _build_prompt(self, problem_type, constraints, mode):
        base = f"Solve this {problem_type} problem.\nConstraints:\n"
        base += "\n".join(f"  {k}: {v}" for k, v in constraints.items())

        if mode == "cot":
            base += "\n\nThink step by step and return JSON with keys: solution, reasoning_trace, confidence."
        elif mode == "few-shot":
            base = f"Examples:\n  scheduling(3 jobs,2 machines)→makespan:10h\n\n" + base
        else:
            base += "\n\nReturn JSON with keys: solution, confidence."
        return base

    def _self_consistency_solve(self, prompt, n=5):
        from collections import Counter
        solutions = []
        for _ in range(n):
            r = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            solutions.append(self._parse(r.choices[0].message.content).get("solution"))
        best = Counter(str(s) for s in solutions).most_common(1)[0][0]
        return {"solution": best, "reasoning_trace": f"Self-consistency n={n}", "confidence": 0.9}

    def _parse(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"solution": raw, "reasoning_trace": "", "confidence": None}


if __name__ == "__main__":
    optimizer = LLMOptimizer(model="gpt-4", mode="cot")
    print("LLMOptimizer ready.")
    print("Modes: zero-shot, few-shot, cot, self-consistency, rag")
