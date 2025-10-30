# ========================================
# Agentic Web Researcher with Stepwise Streaming Evaluation (GPT-4.1)
# Cleaned and Lint-Compliant for Colab
# ========================================

import os
import json
import re
import textwrap
from typing import List, Dict
from duckduckgo_search import DDGS
from openai import OpenAI

# --- SETUP ---
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL_REASONING = "gpt-4.1"
MODEL_LIGHT = "gpt-4.1-mini"


# --- UTILITIES ---
def wrap(text: str, width: int = 100) -> str:
    """Word-wrap text for console readability."""
    return textwrap.fill(text, width=width)


def web_search(query: str, max_results: int = 6, site_filters: List[str] = None) -> List[Dict[str, str]]:
    """
    Perform privacy-preserving web search using DuckDuckGo.
    Optionally bias results toward specific domains.
    """
    results_total = []
    subqueries = [f"{query} site:{domain}" for domain in site_filters] if site_filters else [query]

    with DDGS() as ddgs:
        for subquery in subqueries:
            for result in ddgs.text(subquery, max_results=max_results):
                results_total.append({
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "link": result.get("href", ""),
                    "source": subquery
                })
    return results_total


def gpt_stream(model: str, system_msg: str, user_msg: str, temperature: float = 0.3) -> str:
    """Stream GPT-4.1 output live to Colab console."""
    stream = client.chat.completions.create(
        model=model,
        temperature=temperature,
        stream=True,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
    )

    full_text = ""
    for event in stream:
        if event.choices and event.choices[0].delta and event.choices[0].delta.content:
            chunk = event.choices[0].delta.content
            print(chunk, end="", flush=True)
            full_text += chunk
    print()  # Final newline for clean output
    return full_text.strip()


# --- PLANNING ---
def generate_plan(user_question: str) -> Dict[str, object]:
    """Generate structured research plan before executing searches."""
    planning_prompt = f"""
You are a web research planner.

User Question:
{user_question}

Clarify the intent, break into 2–4 research steps,
propose search keywords and sources (domains),
and specify output style.
Add a note that searches use a privacy-preserving meta-search layer
to reduce tracking.

Return JSON: intent, steps, keywords, sources, output_style, safety_note
"""

    raw_plan = gpt_stream(
        MODEL_LIGHT,
        "Return JSON only.",
        planning_prompt,
        0.2
    )

    try:
        plan = json.loads(raw_plan)
    except json.JSONDecodeError:
        plan = {
            "intent": user_question,
            "steps": ["Gather background", "Find current developments", "Summarize with sources"],
            "keywords": [user_question],
            "sources": ["wikipedia.org", "reputable news", "gov/edu"],
            "output_style": "Executive summary + findings + citations",
            "safety_note": "Uses privacy-preserving search to minimize tracking."
        }

    for key in ["steps", "keywords", "sources"]:
        if not isinstance(plan.get(key), list):
            plan[key] = [plan.get(key)]

    return plan


def pretty_print_plan(plan: Dict[str, object]) -> None:
    """Display the generated plan for user confirmation."""
    print("\n----- PROPOSED PLAN -----")
    print("Intent:", plan["intent"])
    print("\nSteps:")
    for idx, step in enumerate(plan["steps"], 1):
        print(f"  {idx}. {step}")
    print("\nKeywords:", ", ".join(plan["keywords"]))
    print("Sources:", ", ".join(plan["sources"]))
    print("\nNote:", plan["safety_note"])
    print("\nEdit sources (comma-separated) or press Enter to accept.\n")


def confirm_plan(plan: Dict[str, object]) -> Dict[str, object]:
    """Allow user to edit or confirm the plan."""
    pretty_print_plan(plan)
    edit = input("Source override: ").strip()
    if edit:
        plan["sources"] = [source.strip() for source in edit.split(",") if source.strip()]
    print("\nFinal sources:")
    for source in plan["sources"]:
        print(f"  - {source}")
    print()
    return plan


# --- EXECUTION WITH STREAMING ---
def execute_plan(user_question: str, plan: Dict[str, object]) -> None:
    """Execute the multi-step search plan with streaming evaluations."""
    print("\n----- EXECUTION START -----\n")
    all_hits = []

    for step_index, step in enumerate(plan["steps"], 1):
        print(f"[Step {step_index}] {step}\n")
        step_results = []

        for keyword in plan["keywords"]:
            print(f"Searching for: {keyword}")
            hits = web_search(keyword, 6, plan["sources"])
            step_results.extend(hits)
            print(f"  Found {len(hits)} results.\n")

        snippets = "\n".join(
            f"Title: {hit['title']}\nSnippet: {hit['body']}\nURL: {hit['link']}\n"
            for hit in step_results[:10]
        )

        eval_prompt = f"""
You are evaluating search completeness for step "{step}".
Below are the retrieved snippets.

Snippets:
{snippets}

Determine:
1. Whether the information covers the topic adequately.
2. If not, suggest refined or expanded search queries.
3. Justify your judgment briefly.

Output a short reflection with reasoning and, if needed,
new queries to run next.
"""
        print("Evaluating coverage (streaming):\n")
        evaluation = gpt_stream(
            MODEL_REASONING,
            "You are a transparent research agent narrating reasoning step by step.",
            eval_prompt,
            0.4
        )

        new_queries = re.findall(r'["“](.*?)["”]', evaluation)
        if new_queries:
            print("\nDetected potential follow-up queries. Running expanded search...\n")
            for new_q in new_queries[:2]:
                hits_extra = web_search(new_q, 4, plan["sources"])
                print(f"  Expanded search '{new_q}' returned {len(hits_extra)} new hits.\n")
                step_results.extend(hits_extra)

        all_hits.extend(step_results)
        print("\n----- End of Step Evaluation -----\n")

    deduped = {hit["link"]: hit for hit in all_hits}
    evidence = "\n".join(
        f"[{i}] Title: {hit['title']}\nSnippet: {hit['body']}\nURL: {hit['link']}"
        for i, hit in enumerate(deduped.values(), 1)
    )

    final_prompt = f"""
You are a transparent analyst.

User Question: {user_question}
Intent: {plan['intent']}
Output Style: {plan['output_style']}

Instructions:
1. Provide an executive summary (3–6 concise bullets).
2. Then a section of detailed findings.
3. Then numbered sources.
4. Cite [n] numbers matching URLs in the evidence.

Evidence:
{evidence}
"""
    print("Synthesizing final answer (streaming):\n")
    gpt_stream(
        MODEL_REASONING,
        "You write well-structured, cited research briefings.",
        final_prompt,
        0.3
    )
    print("\n----- END -----\n")


# --- MAIN LOOP ---
def run_agent() -> None:
    """Interactive loop for repeated queries."""
    while True:
        question = input("Your question (or 'exit'): ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Session ended.")
            break
        plan = generate_plan(question)
        plan = confirm_plan(plan)
        execute_plan(question, plan)


# Uncomment to start automatically
run_agent()
