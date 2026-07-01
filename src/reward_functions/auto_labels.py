"""
PRM-auto step scoring via GPT-4o mini, replacing the locally-trained
classifier used in PRM-human. One API call judges every step of a
completion at once (not one call per step) — GRPO calls the reward
function thousands of times over a training run, so per-step calls
would mean tens of thousands of API round-trips.
"""

import os
import json
import re
import concurrent.futures
from openai import OpenAI

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


JUDGE_PROMPT = """You are grading a student's step-by-step math solution. For each step, decide if it is mathematically correct given everything before it (correct arithmetic/algebra, valid logical move, consistent with prior steps).

Problem: {problem}

Steps:
{numbered_steps}

Respond with ONLY a JSON array of "correct" or "incorrect", one per step, in order. Example for 3 steps: ["correct", "correct", "incorrect"]"""


def _judge_one_completion(problem, steps, model="gpt-4o-mini"):
    """Returns a list of booleans, one per step. Falls back to all-True
    (a netural-ish default) if the API call or parsing fails, so a
    transient error doesn't crash the whole training step."""
    if not steps:
        return []

    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    prompt = JUDGE_PROMPT.format(problem=problem, numbered_steps=numbered)

    for attempt in range(3):
        try:
            client = get_client()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )
            text = response.choices[0].message.content.strip()
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                labels = json.loads(match.group(0))
                if len(labels) == len(steps):
                    return [str(l).strip().lower() == "correct" for l in labels]
        except Exception as e:
            if attempt == 2:
                print(f"  [auto_labels] API call failed after 3 attempts: {e}")

    # fallback: treat as all-correct rather than crash the training run.
    # this is rare (network/parsing failures only) and dilutes signal for
    # that one completion rather than losing the whole batch
    return [True] * len(steps)


def score_completions_batch(problems_and_steps, model="gpt-4o-mini", max_workers=8):
    """problems_and_steps: list of (problem_text, steps_list) tuples.
    Returns a list of scores in [0, 1], one per input — the fraction of
    steps judged correct. Calls run in parallel since they're independent,
    otherwise sequential network latency would dominate training time."""
    results = [None] * len(problems_and_steps)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_judge_one_completion, problem, steps, model): i
            for i, (problem, steps) in enumerate(problems_and_steps)
        }
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            labels = future.result()
            results[i] = (sum(labels) / len(labels)) if labels else 0.0

    return results
