#!/usr/bin/env python3
"""
Conversational AI Agent – JSON Knowledge-Base with LLM-Based Relevance Scoring
===============================================================================
Loads an enriched JSON knowledge base (tree of titled nodes with instructions
and knowledge_facts), extracts text chunks, exhaustively scores each chunk
against the user query via an LLM call, assembles context (with hierarchy for
title-type chunks), and generates a conversational answer.

All interactions are logged to a timestamped file.
"""

import json
import os
import sys
import time
import re
import random
import datetime
import logging
import getpass
import concurrent.futures
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# ── Load API keys from .env ───────────────────────────────────────────────────
env_path = r"E:\may10\API_keys_for_Sander\api_keys_sander.env"
load_dotenv(env_path)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "json_input_file":  r"very small enriched_Mario_Pizza_small_feb27.json",
    
    "model":           "gemini-2.0-flash",# "gemini-2.5-flash",       # used for relevance, intent
    "disable_thinking_for_flash": True,           # option to not use thinking mode for 'model'
    
    "summary_model":    "gemini-2.0-flash",# "gemini-2.5-flash",       # used for KB summary
    "disable_thinking_for_summary_flash": True,  # option to not use thinking mode for 'summary_model'
    
    "answer_model":     "gemini-2.5-pro",         # used for final answer generation
    "disable_thinking_for_answer_flash": True,   # option to not use thinking mode for 'answer_model'
    "max_passes":       2,
    "pacing_delay":     0.5,          # seconds to wait if we hit a rate limit error (base delay)
    "max_concurrent_requests": 20,    # number of threads to use for parallel scoring

    # ── Role & Domain ─────────────────────────────────────────────────────
    "business_domain":  "Pizza Quick-Service Restaurant",
    "role":             "Order Taker / Receptionist",
    "role_goal":        "Order Receptionist",
    "business_name":    "Mario's Pizza",

    # ── Retrieval ─────────────────────────────
    "generate_pacing":           1.2,    # Time to wait (seconds) between scoring calls
    "relevance_score_threshold": 9,      # set  rule to have good relevant chunks 
    "min_relevant_chunks":       7, #20,     # hard floor for chunk retrieval 
    "min_relevance_score_floor": 5,      # absolute lowest score allowed to meet the hard floor
    "max_concurrent_requests":   30,     # For ThreadPoolExecutor (chunk scoring)
    "max_history_turns":         10,  # conversation turns kept in context


}

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

_SCRIPT_NAME = Path(__file__).stem          # e.g. "ai_agent"
_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"agent_session_{_timestamp}.log"

# Silence ALL third-party loggers by raising the root logger level
logging.basicConfig(level=logging.WARNING)

# Create our own logger with dedicated handlers (so our INFO messages get through)
logger = logging.getLogger("ai_agent")
logger.setLevel(logging.INFO)
logger.propagate = False   # don't send to root (avoids duplicates)

_fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
logger.addHandler(_sh)

# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE-BASE LOADING & CHUNK EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════


def load_knowledge_base(json_path: str):
    """
    Parse the enriched JSON file and return:
      nodes_lookup : dict[node_number -> node_dict]
      chunks       : list of chunk dicts
    Each chunk has keys: type, text, node_number, node (reference to full node).
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    nodes = data.get("nodes", [])
    nodes_lookup: dict[str, dict] = {}
    chunks: list[dict] = []

    for node in nodes:
        num = node["number"]
        nodes_lookup[num] = node

        # ── Title chunk ───────────────────────────────────────────────────
        title_text = node.get("title", "").strip()
        if title_text:
            chunks.append({
                "type":        "title",
                "text":        title_text,
                "node_number": num,
                "node":        node,
            })

        # ── Instruction chunks (one per sentence) ────────────────────────
        for sentence in node.get("instructions", {}).get("items", []):
            s = sentence.strip()
            if s:
                chunks.append({
                    "type":        "instruction",
                    "text":        s,
                    "node_number": num,
                    "node":        node,
                })

        # ── Knowledge-fact chunks (one per sentence) ─────────────────────
        for sentence in node.get("knowledge_facts", {}).get("items", []):
            s = sentence.strip()
            if s:
                chunks.append({
                    "type":        "knowledge_fact",
                    "text":        s,
                    "node_number": num,
                    "node":        node,
                })

    logger.info("Loaded %d nodes  →  %d chunks from %s", len(nodes), len(chunks), json_path)
    return nodes_lookup, chunks


# ═══════════════════════════════════════════════════════════════════════════════
# DEBUG DUMP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def dump_full_kb_text(nodes: list[dict], output_dir: Path) -> Path:
    """
    Write every node (number, title, instructions, knowledge_facts) as
    human-readable text for debugging.  Returns the output path.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"kb_full_dump_{_SCRIPT_NAME}_{ts}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        for node in nodes:
            indent = "  " * (node.get("depth", 1) - 1)
            f.write(f"{indent}[{node['number']}] {node.get('title', '')}\n")

            for item in node.get("instructions", {}).get("items", []):
                f.write(f"{indent}  INSTRUCTION: {item}\n")
            for item in node.get("knowledge_facts", {}).get("items", []):
                f.write(f"{indent}  FACT: {item}\n")
            f.write("\n")

    logger.info("Full KB text dump saved → %s", out_path)
    return out_path


def dump_titles_text(nodes: list[dict], output_dir: Path) -> tuple[Path, str]:
    """
    Write only the hierarchical titles (indented by depth) to a .txt file.
    Returns (output_path, titles_text_string).
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"kb_titles_{_SCRIPT_NAME}_{ts}.txt"

    lines: list[str] = []
    for node in nodes:
        indent = "  " * (node.get("depth", 1) - 1)
        lines.append(f"{indent}{node['number']}. {node.get('title', '')}")

    titles_text = "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(titles_text)

    logger.info("Titles text dump saved → %s", out_path)
    return out_path, titles_text


# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _parent_number(num: str) -> str | None:
    """Return the parent node number (e.g. '3.1.2' → '3.1'), or None for root."""
    parts = num.split(".")
    if len(parts) <= 1:
        return None
    return ".".join(parts[:-1])


def get_ancestor_titles(node_number: str, nodes_lookup: dict) -> list[str]:
    """
    Walk up the hierarchy collecting title strings from root down.
    E.g. for node '3.1.2.1':  ['Menu Offerings', 'Build-Your-Own Pizza',
                                'Crust Options', 'Regular Crust – Included']
    """
    chain: list[str] = []
    current = node_number
    while current is not None:
        node = nodes_lookup.get(current)
        if node:
            chain.append(node.get("title", ""))
        current = _parent_number(current)
    chain.reverse()  # root → leaf order
    return chain


# ═══════════════════════════════════════════════════════════════════════════════
# LLM HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _init_client(model_name: str):
    """Initialise the google.genai Client and return (client, model_name).
       Checks GEMINI_API_KEY / GOOGLE_API_KEY env vars first; if absent,
       prompts the user interactively."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n⚠  No GEMINI_API_KEY or GOOGLE_API_KEY found in environment.")
        api_key = getpass.getpass("   Paste your Gemini API key here: ").strip()
        if not api_key:
            logger.error("No API key provided – cannot continue.")
            sys.exit(1)
        os.environ["GEMINI_API_KEY"] = api_key   # cache for the session

    client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialised for model '%s'", model_name)
    return client, model_name


import threading
USAGE_LOCK = threading.Lock()
USAGE_STATS = {
    "model": {"input": 0, "output": 0, "thinking": 0, "cost": 0.0},
    "summary_model": {"input": 0, "output": 0, "thinking": 0, "cost": 0.0},
    "answer_model": {"input": 0, "output": 0, "thinking": 0, "cost": 0.0}
}

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    # Approximate pricing per 1M tokens
    rates = {
        "gemini-2.5-flash": {"in": 0.075, "out": 0.30},
        "gemini-2.5-pro": {"in": 1.25, "out": 5.00},
        "gemini-2.0-flash": {"in": 0.075, "out": 0.30},
        "gemini-2.0-pro": {"in": 1.25, "out": 5.00},
        "gemini-1.5-flash": {"in": 0.075, "out": 0.30},
        "gemini-1.5-pro": {"in": 1.25, "out": 5.00}
    }
    rate = rates.get(model_name, {"in": 0.075, "out": 0.30})
    return (input_tokens / 1_000_000) * rate["in"] + (output_tokens / 1_000_000) * rate["out"]

def call_llm(client, model_name: str, prompt: str, usage_role: str = "model", return_usage: bool = False) -> str | tuple[str, dict]:
    """Send a prompt to the Gemini model with a custom exponential backoff retry."""
    max_retries = 5
    base_delay = 1.0
    
    generate_kwargs = {
        "model": model_name,
        "contents": prompt,
    }
    
    role_disable_thinking = False
    if usage_role == "model":
        role_disable_thinking = CONFIG.get("disable_thinking_for_flash", True)
    elif usage_role == "summary_model":
        role_disable_thinking = CONFIG.get("disable_thinking_for_summary_flash", False)
    elif usage_role == "answer_model":
        role_disable_thinking = CONFIG.get("disable_thinking_for_answer_flash", False)
        
    if role_disable_thinking and "2.5-flash" in model_name.lower():
        generate_kwargs["config"] = {'thinking_config': {'thinking_budget': 0}}
        
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(**generate_kwargs)
            
            # Extract usage metadata
            meta = response.usage_metadata
            usage_info = {"in": 0, "out": 0, "thinking": 0, "cost": 0.0}
            if meta:
                in_toks = getattr(meta, 'prompt_token_count', 0) or 0
                out_toks = getattr(meta, 'candidates_token_count', 0) or 0
                thinking_toks = getattr(meta, 'thoughts_token_count', 0) or 0
                
                # Thinking tokens are billed at the same rate as output tokens
                call_cost = calculate_cost(model_name, in_toks, out_toks + thinking_toks)
                
                with USAGE_LOCK:
                    USAGE_STATS[usage_role]["input"] += in_toks
                    USAGE_STATS[usage_role]["output"] += out_toks
                    USAGE_STATS[usage_role]["thinking"] += thinking_toks
                    USAGE_STATS[usage_role]["cost"] += call_cost
                
                usage_info = {"in": in_toks, "out": out_toks, "thinking": thinking_toks, "cost": call_cost}
                
                if not return_usage:
                    msg = f"💰 [LLM Call - {usage_role} ({model_name})] Tokens: {in_toks} in, {out_toks} out, {thinking_toks} thinking | Cost: ${call_cost:.6f}"
                    print(msg)
                    logger.info(msg)
                
            if return_usage:
                return response.text.strip(), usage_info
            return response.text.strip()
        except Exception as exc:
            err_msg = str(exc).lower()
            if "503" in err_msg or "429" in err_msg or "too many requests" in err_msg or "unavailable" in err_msg:
                if attempt == max_retries - 1:
                    raise exc
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning("API error '%s' on attempt %d. Retrying in %.1fs...", str(exc).split()[0], attempt + 1, delay)
                time.sleep(delay)
            else:
                raise exc


# ═══════════════════════════════════════════════════════════════════════════════
# KB SUMMARY & INTENT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

KB_SUMMARY_PROMPT = """\
Below is a hierarchical list of titles from a knowledge base.
Write a concise summary (3-5 sentences) describing what this knowledge base \
covers, what topics and categories it contains, and what kind of questions \
it could answer.  Be factual and generic — do not invent information.

Titles:
{titles_text}

Summary:
"""


def generate_kb_summary(client, model_name: str, titles_text: str,
                        output_dir: Path) -> str:
    """
    Send the titles text to the LLM to produce a concise summary of the
    knowledge base.  Saves the summary to a .txt file and returns it.
    """
    prompt = KB_SUMMARY_PROMPT.format(titles_text=titles_text)
    summary = call_llm(client, model_name, prompt, usage_role="summary_model")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"kb_summary_{_SCRIPT_NAME}_{ts}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info("KB summary saved → %s", out_path)
    logger.info("KB Summary: %s", summary)
    return summary


INTENT_PROMPT_TEMPLATE = """\
You are an intent detector. Given a summary of a knowledge base, \
a user's query, and optionally prior conversation history, determine \
the user's specific intent.

Knowledge Base Summary:
{kb_summary}

{history_block}

User query: "{query}"

Respond with a single short sentence describing the user's intent. \
Be specific about what information the user is looking for. \
Do NOT restate the query; instead describe the underlying need.

Intent:
"""


def detect_query_intent(
    client, model_name: str, kb_summary: str, query: str,
    conversation_history: list[dict], pacing_delay: float
) -> str:
    """
    Detect the user's intent using the KB summary and (optionally)
    conversation history.  Returns a short intent description string.
    """
    if conversation_history:
        history_lines = []
        for turn in conversation_history[-10:]:
            role = turn["role"].upper()
            history_lines.append(f"{role}: {turn['content']}")
        history_block = "Conversation so far:\n" + "\n".join(history_lines)
    else:
        history_block = "(No prior conversation)"

    prompt = INTENT_PROMPT_TEMPLATE.format(
        kb_summary=kb_summary, query=query, history_block=history_block
    )
    intent = call_llm(client, model_name, prompt, usage_role="model")
    logger.info("Detected intent: %s", intent)
    time.sleep(pacing_delay)
    return intent


# ═══════════════════════════════════════════════════════════════════════════════
# RELEVANCE SCORING  (exhaustive LLM query per chunk)
# ═══════════════════════════════════════════════════════════════════════════════

RELEVANCE_PROMPT_TEMPLATE = """\
You are a STRICT relevance judge.

The user's detected intent is:
"{intent}"

Given this intent, decide whether the text snippet below DIRECTLY provides \
factual information that fulfils the intent.

Be very strict:
- Score 8-10: The snippet directly provides factual information that fulfils the intent.
- Score 5-7:  The snippet is somewhat related but does not directly fulfil the intent.
- Score 1-4:  The snippet is only tangentially related (same general domain).
- Score 0:    Completely unrelated.

Only mark "relevant": true if the snippet would be NECESSARY to compose \
a good answer that fulfils the intent.
Do NOT mark a snippet as relevant just because it shares keywords with the intent.

Original user query: "{query}"

Text snippet: "{chunk_text}"

Reply ONLY with a JSON object (no markdown fences) in this exact format:
{{"relevant": true, "score": 8}}

Where:
- "relevant" is true ONLY if the snippet directly helps fulfil the intent, false otherwise.
- "score" is an integer from 0 (completely irrelevant) to 10 (perfectly relevant).
"""


def _parse_relevance_json(raw: str) -> tuple[bool, int]:
    """Best-effort parse of the LLM relevance JSON."""
    # Strip markdown fences if present
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        relevant = bool(obj.get("relevant", False))
        score = int(obj.get("score", 0))
        return relevant, score
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback heuristics
        if '"relevant": true' in raw.lower() or '"relevant":true' in raw.lower():
            # Try to extract score
            m = re.search(r'"score"\s*:\s*(\d+)', raw)
            sc = int(m.group(1)) if m else 5
            return True, sc
        return False, 0


def score_chunk_relevance(
    client, model_name: str, query: str, chunk_text: str,
    intent: str, pacing_delay: float
) -> tuple[bool, int, dict]:
    """Send one LLM call to judge the relevance of a single chunk."""
    prompt = RELEVANCE_PROMPT_TEMPLATE.format(
        query=query, chunk_text=chunk_text, intent=intent
    )
    try:
        raw, usage = call_llm(client, model_name, prompt, usage_role="model", return_usage=True)
        relevant, score = _parse_relevance_json(raw)
    except Exception as exc:
        logger.warning("Relevance call failed for chunk [%.60s…]: %s", chunk_text, exc)
        relevant, score, usage = False, 0, {"in": 0, "out": 0, "thinking": 0, "cost": 0.0}

    time.sleep(pacing_delay)
    return relevant, score, usage


def find_relevant_chunks(
    client,
    model_name: str,
    query: str,
    intent: str,
    chunks: list[dict],
    pacing_delay: float,
    score_threshold: int,
    min_chunks: int = 20,
    min_score_floor: int = 2,
) -> tuple[list[dict], list[dict]]:
    """
    Scores every chunk against the detected intent.
    Returns a list of chunk dicts augmented with 'score' key, sorted descending.
    """
    scored: list[dict] = []
    evaluation_history: list[dict] = []

    def score_worker(chunk: dict) -> tuple[int, dict, bool, int, dict]:
        """Worker function for parallel execution."""
        idx = chunks.index(chunk)
        relevant, score, usage = score_chunk_relevance(
            client, model_name, query, chunk["text"], intent, pacing_delay
        )
        return idx, chunk, relevant, score, usage

    logger.info("── Relevance Scoring (%d chunks to evaluate in parallel) ──", len(chunks))
    
    max_workers = CONFIG.get("max_concurrent_requests", 10)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(score_worker, chunk) for chunk in chunks]
        
        # Use as_completed to process them as they finish, but we log the absolute index
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            completed_count += 1
            try:
                idx, chunk, relevant, score, usage = future.result()
                
                evaluation_history.append({
                    "chunk": chunk,
                    "relevant": relevant,
                    "score": score,
                    "usage": usage
                })
                
                use_str = ""
                if usage and usage["in"] > 0:
                    use_str = f" | Tokens: {usage['in']} in, {usage['out']} out | Cost: ${usage['cost']:.6f}"
                
                logger.info(
                    "  [%d/%d] score=%2d  relevant=%-5s  type=%-14s  text=%s%s",
                    completed_count, len(chunks), score, str(relevant), chunk["type"],
                    chunk["text"].replace("\n", " "), use_str
                )
                if relevant and score >= score_threshold:
                    entry = dict(chunk)
                    entry["score"] = score
                    scored.append(entry)
            except Exception as exc:
                logger.error("  Chunk scoring failed: %s", exc)

    logger.info("Evaluated all %d chunks.", len(chunks))

    # De-duplicate
    seen_texts: set[str] = set()
    unique: list[dict] = []
    for ev in evaluation_history:
        s = ev["chunk"]
        # We append a copy of the chunk with the score added, so it's uniform
        scored_entry = dict(s)
        scored_entry["score"] = ev["score"]
        
        if scored_entry["text"] not in seen_texts:
            seen_texts.add(scored_entry["text"])
            unique.append(scored_entry)

    unique.sort(key=lambda c: c["score"], reverse=True)
    
    # ── Apply the new policy ──────────────────────────────────────────────────
    # 1. Take all chunks with score >= score_threshold (9)
    # 2. But ensure we take AT LEAST min_chunks (20) overall
    
    final_selection = []
    for c in unique:
        if c["score"] >= score_threshold:
            final_selection.append(c)
        elif len(final_selection) < min_chunks and c["score"] >= min_score_floor:
            final_selection.append(c)
        else:
            break
            
    return final_selection, evaluation_history


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════


def build_context(relevant_chunks: list[dict], nodes_lookup: dict) -> str:
    """
    Assemble context text from relevant chunks following the rules:

    1) Title chunk  →  hierarchy path (ancestor titles) + instructions + knowledge_facts
    2) Instruction / knowledge_fact chunk  →  the single sentence as-is
    """
    context_blocks: list[str] = []

    idx = 1
    for rank, chunk in enumerate(relevant_chunks, 1):
        ctype = chunk["type"]
        node_number = chunk["node_number"]
        node = chunk["node"]
        score = chunk.get("score", "N/A")

        if ctype == "title":
            ancestors = get_ancestor_titles(node_number, nodes_lookup)
            hierarchy_path = " > ".join(ancestors)

            # 1) Include the hierarchy path
            block_title = (
                f"[Context #{idx} rank {rank} — Title hierarchy for node {node_number} (Score: {score})]\n"
                f"Path: {hierarchy_path}"
            )
            context_blocks.append(block_title)
            idx += 1

            # 2) Include all instructions for this node
            instructions = node.get("instructions", {}).get("items", [])
            if instructions:
                inst_lines = "\n".join(f"  {i}. {item}" for i, item in enumerate(instructions, 1))
                block_inst = (
                    f"[Context #{idx} rank {rank} — Instructions array for node {node_number} (Score: {score})]\n"
                    f"{inst_lines}"
                )
                context_blocks.append(block_inst)
                idx += 1

            # 3) Include all knowledge facts for this node
            facts = node.get("knowledge_facts", {}).get("items", [])
            if facts:
                fact_lines = "\n".join(f"  {i}. {item}" for i, item in enumerate(facts, 1))
                block_fact = (
                    f"[Context #{idx} rank {rank} — Knowledge Facts array for node {node_number} (Score: {score})]\n"
                    f"{fact_lines}"
                )
                context_blocks.append(block_fact)
                idx += 1

        else:
            block = (
                f"[Context #{idx} rank {rank} — {ctype} from node {node_number} (Score: {score})]\n"
                f"{chunk['text']}"
            )
            context_blocks.append(block)
            idx += 1

    return "\n\n".join(context_blocks)


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

MASTER_PROMPT_TEMPLATE = """\
You are **{business_name}** — a {role} for a {business_domain}.
Your goal is to serve as a helpful {role_goal}.

=== KNOWLEDGE CONTEXT (retrieved from the business knowledge base) ===
{context}
=== END KNOWLEDGE CONTEXT ===

=== CONVERSATION HISTORY ===
{history}
=== END CONVERSATION HISTORY ===

CUSTOMER'S CURRENT MESSAGE:
\"{query}\"

INSTRUCTIONS FOR YOUR RESPONSE:
1. Answer the customer's question using ONLY the knowledge context above.
2. Stay in character as a friendly, professional {role}.
3. If the customer asks for an item, service, or feature (e.g. drinks, salads, delivery to a specific area) that is NOT explicitly mentioned in the context, you MUST politely inform them that it is currently unavailable or you don't have it.
4. Do NOT use your general knowledge to assume what a {business_domain} typically sells. If it's not in the context, it doesn't exist for this business.
5. Keep answers concise but complete.
6. Do NOT invent prices, menu items, or policies not present in the context.
"""


def build_master_prompt(
    query: str,
    context: str,
    conversation_history: list[dict],
    config: dict,
) -> str:
    """Build the final prompt sent to the LLM for the answer."""
    # Format history
    history_lines: list[str] = []
    max_turns = config.get("max_history_turns", 10)
    for turn in conversation_history[-max_turns:]:
        role = turn["role"].upper()
        history_lines.append(f"{role}: {turn['content']}")
    history_str = "\n".join(history_lines) if history_lines else "(no prior conversation)"

    return MASTER_PROMPT_TEMPLATE.format(
        business_name=config["business_name"],
        role=config["role"],
        business_domain=config["business_domain"],
        role_goal=config["role_goal"],
        context=context if context else "(no relevant context found)",
        history=history_str,
        query=query,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTION LOGGER
# ═══════════════════════════════════════════════════════════════════════════════

DETAIL_LOG_DIR = Path(__file__).parent / "logs" / "interactions"
DETAIL_LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_interaction(
    query: str,
    relevant_chunks: list[dict],
    all_evaluations: list[dict],
    prompt: str,
    answer: str,
    turn_number: int,
):
    """Write a detailed log entry for a single interaction."""
    DETAIL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = DETAIL_LOG_DIR / f"turn_{turn_number:04d}_{ts}.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 80}\n")
        f.write(f"TURN {turn_number}  —  {ts}\n")
        f.write(f"{'=' * 80}\n\n")

        f.write(f"USER QUERY:\n{query}\n\n")

        f.write(f"{'─' * 40}\n")
        f.write(f"ALL CHUNK EVALUATIONS ({len(all_evaluations)}):\n")
        f.write(f"{'─' * 40}\n")
        total_evals = len(all_evaluations)
        for i, ev in enumerate(all_evaluations, 1):
            ch = ev["chunk"]
            use = ev.get("usage")
            use_str = f"Tokens: {use['in']} in, {use['out']} out | Cost: ${use['cost']:.6f}" if use else "Failed"
            f.write(f"\n  [{i}/{total_evals}] score={ev['score']} relevant={ev['relevant']} type={ch['type']} node={ch.get('node_number')} | {use_str}\n")
            f.write(f"      text: {ch['text']}\n")

        f.write(f"\n{'─' * 40}\n")
        f.write(f"RELEVANT CHUNKS SELECTED ({len(relevant_chunks)}):\n")
        f.write(f"{'─' * 40}\n")
        for i, ch in enumerate(relevant_chunks, 1):
            f.write(f"\n  [{i}] type={ch['type']}  node={ch['node_number']}  score={ch.get('score','?')}\n")
            f.write(f"      text: {ch['text']}\n")
            
            # Print associated instructions and facts if this chunk brought them along
            if ch.get("type") == "title" and "node" in ch:
                node = ch["node"]
                
                inst_items = node.get("instructions", {}).get("items", [])
                if inst_items:
                    f.write("      instructions:\n")
                    for it in inst_items:
                        f.write(f"        - {it}\n")
                        
                fact_items = node.get("knowledge_facts", {}).get("items", [])
                if fact_items:
                    f.write("      knowledge_facts:\n")
                    for it in fact_items:
                        f.write(f"        - {it}\n")


        f.write(f"\n{'─' * 40}\n")
        f.write(f"FULL PROMPT SENT TO LLM:\n")
        f.write(f"{'─' * 40}\n")
        f.write(prompt)
        f.write(f"\n\n{'─' * 40}\n")
        f.write(f"LLM ANSWER:\n")
        f.write(f"{'─' * 40}\n")
        f.write(answer)
        f.write("\n")
        
        f.write(f"{'─' * 40}\n")
        f.write(f"ACCUMULATED USAGE AT END OF THIS TURN:\n")
        f.write(f"{'─' * 40}\n")
        t_cost = 0.0
        for role_name in ["summary_model", "model", "answer_model"]:
            stats = USAGE_STATS[role_name]
            r_cost = stats["cost"]
            t_cost += r_cost
            f.write(f"  {role_name:13s} - In: {stats['input']:8d} | Out: {stats['output']:8d} | Think: {stats['thinking']:8d} | Cost: ${r_cost:.6f}\n")
        f.write(f"  {'TOTAL':13s}                                               | Cost: ${t_cost:.6f}\n")
        f.write(f"{'─' * 40}\n")

    logger.info("Interaction log saved → %s", log_path)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN  —  Interactive CLI Loop
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print(f"  {CONFIG['business_name']} — AI {CONFIG['role_goal']}")
    print(f"  Scoring model: {CONFIG['model']}")
    print(f"  Answer  model: {CONFIG['answer_model']}")
    print("=" * 60)

    # ── Load knowledge base ───────────────────────────────────────────────
    kb_path = Path(__file__).parent / CONFIG["json_input_file"]
    if not kb_path.exists():
        logger.error("Knowledge-base file not found: %s", kb_path)
        sys.exit(1)

    nodes_lookup, chunks = load_knowledge_base(str(kb_path))
    logger.info("Ready.  Total chunks available for retrieval: %d", len(chunks))

    # ── Step 0a: Dump full KB to .txt for debugging ───────────────────────
    with open(kb_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    all_nodes = raw_data.get("nodes", [])
    dump_full_kb_text(all_nodes, LOG_DIR)

    # ── Step 0b: Dump titles to .txt ──────────────────────────────────────
    _, titles_text = dump_titles_text(all_nodes, LOG_DIR)

    # ── Init LLM (two models: fast for scoring, strong for answers) ─────
    client, model_name = _init_client(CONFIG["model"])
    summary_client, summary_model = _init_client(CONFIG.get("summary_model", CONFIG["model"]))
    answer_client, answer_model = _init_client(CONFIG["answer_model"])

    # ── Step 0c: Generate KB summary from titles via LLM ─────────────────
    print(f"⏳ Generating knowledge-base summary from titles using {summary_model} …")
    kb_summary = generate_kb_summary(summary_client, summary_model, titles_text, LOG_DIR)
    print(f"✅ KB Summary ready.\n")

    # ── Conversation state ────────────────────────────────────────────────
    conversation_history: list[dict] = []
    turn_number = 0

    print("\nType your message (or 'exit' / 'quit' to end).\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        turn_number += 1
        logger.info("──── TURN %d ────", turn_number)
        logger.info("User: %s", user_input)

        # ── Step 1: Detect query intent ───────────────────────────────────
        print("⏳ Detecting query intent …")
        intent = detect_query_intent(
            client, model_name, kb_summary, user_input,
            conversation_history, CONFIG["pacing_delay"]
        )
        print(f"🎯 Intent: {intent}\n")

        # ── Step 2: Find relevant chunks using intent ─────────────────────
        print(f"⏳ Searching {len(chunks)} knowledge chunks for relevance …")
        t0 = time.time()

        relevant_chunks, all_evaluations = find_relevant_chunks(
            client=client,
            model_name=model_name,
            query=user_input,
            intent=intent,
            chunks=chunks,
            pacing_delay=CONFIG["generate_pacing"],
            score_threshold=CONFIG["relevance_score_threshold"],
            min_chunks=CONFIG.get("min_relevant_chunks", 20),
            min_score_floor=CONFIG.get("min_relevance_score_floor", 2),
        )

        elapsed = time.time() - t0
        print(f"✅ Found {len(relevant_chunks)} relevant chunks in {elapsed:.1f}s\n")

        # ── Step 2: Build context ─────────────────────────────────────────
        context = build_context(relevant_chunks, nodes_lookup)

        # ── Step 3: Build master prompt ───────────────────────────────────
        prompt = build_master_prompt(
            query=user_input,
            context=context,
            conversation_history=conversation_history,
            config=CONFIG,
        )

        # ── DEBUG: print the full prompt ──────────────────────────────────
        debug_msg = f"\n{'═' * 60}\nDEBUG — FULL MASTER PROMPT SENT TO LLM:\n{'═' * 60}\n{prompt}\n{'═' * 60}\n"
        print(debug_msg)
        logger.info("DEBUG — FULL MASTER PROMPT SENT TO LLM:\n%s", prompt)

        # ── Step 4: Get answer ────────────────────────────────────────────
        print(f"⏳ Generating answer with {answer_model} …")
        try:
            answer = call_llm(answer_client, answer_model, prompt, usage_role="answer_model")
        except Exception as exc:
            answer = f"(Error generating answer: {exc})"
            logger.error("Answer generation failed: %s", exc)

        print(f"\n{'─' * 50}")
        print(f"Agent: {answer}")
        print(f"{'─' * 50}\n")

        # ── Step 5: Update history ────────────────────────────────────────
        conversation_history.append({"role": "customer", "content": user_input})
        conversation_history.append({"role": "agent",    "content": answer})

        # ── Step 6: Log everything ────────────────────────────────────────
        log_interaction(
            query=user_input,
            relevant_chunks=relevant_chunks,
            all_evaluations=all_evaluations,
            prompt=prompt,
            answer=answer,
            turn_number=turn_number,
        )

        logger.info("Agent answer: %s", answer)
        
        # ── Step 7: Print accumulated usage ───────────────────────────────
        usage_lines = [
            f"\n{'━' * 60}",
            "📊 ACCUMULATED CONVERSATION USAGE & COST",
            f"{'━' * 60}"
        ]
        t_cost = 0.0
        for role_name in ["summary_model", "model", "answer_model"]:
            stats = USAGE_STATS[role_name]
            r_cost = stats["cost"]
            t_cost += r_cost
            usage_lines.append(f"  {role_name:13s} - In: {stats['input']:8d} | Out: {stats['output']:8d} | Think: {stats['thinking']:8d} | Cost: ${r_cost:.6f}")
        usage_lines.append(f"  {'TOTAL':13s}                                               | Cost: ${t_cost:.6f}")
        usage_lines.append(f"{'━' * 60}\n")
        
        usage_block = "\n".join(usage_lines)
        print(usage_block)
        
        # Log it so it appears in the main agent_session logs as well
        logger.info("ACCUMULATED CONVERSATION USAGE & COST:")
        for line in usage_lines[3:-1]:  # Skip the borders and header
            logger.info("  " + line.strip())


if __name__ == "__main__":
    main()
