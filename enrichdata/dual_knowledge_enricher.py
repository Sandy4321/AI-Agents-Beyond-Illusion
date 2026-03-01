#!/usr/bin/env python3
"""
Dual-Knowledge Enrichment System  v4 — GENERIC + CONTEXT-AWARE
================================================================
SEQUENTIAL (debug-friendly). No asyncio.

Step 1: Reads the ENTIRE business file → extracts a business summary via LLM.
Step 2: For each numbered line, passes the FULL file + summary as context.
        The LLM extracts REAL specifics from the data — no hardcoded examples.

    Layer 1 — 7 Actionable Instructions  (Strategy or Tactics)
    Layer 2 — 5 Knowledge Facts          (extracted from file data)

100% GENERIC template — works for ANY business, not just pizza.

Usage:
    python dual_knowledge_enricher.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load API key from .env file
env_path = r"E:\may10\API_keys_for_Sander\api_keys_sander.env"
load_dotenv(env_path)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    "input_file":       r"very small original Mario Pizza feb26.txt",
    "output_file":      r"very small enriched_Mario_Pizza_small_feb26.txt",
    "json_output_file": r"very small enriched_Mario_Pizza_small_feb26.json",
    "model":            "gemini-2.5-flash",  # e.g., gemini-2.0-flash, gemini-2.5-flash, gpt-4o-mini
    "max_passes":       2,
    "pacing_delay":     1.0,  # seconds to wait between node processing to avoid 429 errors (15 RPM limit)
    "dry_run":          False,
    "num_instructions": 2,#7,     # Layer 1 — Actionable Instructions (Strategy or Tactics)
    "num_facts":        2,#5,     # Layer 2 — Knowledge Facts (extracted from file data)

    # ── Role & Domain (generic — change for any business) ─────────────
    "business_domain":  "Pizza Quick-Service Restaurant",
    "role":             "Order Taker / Receptionist",
    "summary_goal":     "Order Receptionist",
    "business_name":    "Mario's Pizza",
}


# ═══════════════════════════════════════════════════════════════════════════
# 1.  UTILS & API RETRY LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def with_retries(func, *args, **kwargs):
    """Execute an API call with exponential backoff for 429 / Rate Limit errors."""
    max_retries = 6
    base_delay = 5.0  # seconds

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource" in err_str and "exhausted" in err_str or "rate limit" in err_str:
                if attempt == max_retries - 1:
                    print(f"\n❌ Max retries reached: {e}")
                    raise
                
                delay = base_delay * (2 ** attempt)
                print(f"\n  ⚠️ Rate limit hit (429). Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise

# ═══════════════════════════════════════════════════════════════════════════
# 2.  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HeaderNode:
    number: str
    title: str
    depth: int
    raw_line: str
    children_text: str = ""
    is_leaf: bool = True
    line_index: int = 0


@dataclass
class EnrichedNode:
    number: str
    title: str
    depth: int
    is_leaf: bool
    facts: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    refinement_passes: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# 3.  HEADER PARSER — ENRICH EVERY NUMBERED LINE
# ═══════════════════════════════════════════════════════════════════════════

HEADER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.+)$")


def parse_headers(text: str) -> list[HeaderNode]:
    """Every numbered line becomes a header to enrich."""
    lines = text.splitlines()
    nodes: list[HeaderNode] = []
    body_lines: list[str] = []

    for line_idx, line in enumerate(lines):
        m = HEADER_RE.match(line)
        if m:
            number, title = m.group(1), m.group(2).strip()
            if number.count(".") == 0 and len(number) > 1:
                body_lines.append(line)
                continue
            if nodes:
                nodes[-1].children_text = "\n".join(body_lines).strip()
            body_lines = []
            depth = number.count(".") + 1
            nodes.append(HeaderNode(
                number=number, title=title, depth=depth,
                raw_line=line.strip(), line_index=line_idx,
            ))
            continue
        body_lines.append(line)

    if nodes and body_lines:
        nodes[-1].children_text = "\n".join(body_lines).strip()

    for i, node in enumerate(nodes):
        node.is_leaf = True
        for j in range(i + 1, len(nodes)):
            if nodes[j].depth > node.depth:
                node.is_leaf = False
                break
            if nodes[j].depth <= node.depth:
                break

    return nodes


# ═══════════════════════════════════════════════════════════════════════════
# 4.  BUSINESS SUMMARY EXTRACTION (Step 0 — runs once)
# ═══════════════════════════════════════════════════════════════════════════

def build_summary_prompt(full_text: str, cfg: dict) -> str:
    """Prompt to extract STRATEGIC BUSINESS ESSENCE — agnostic to industry."""
    return textwrap.dedent(f"""\
        You are a cross-industry business analyst and AI systems architect.
        
        Your task is to analyze a structured business description document and extract a concise but information-dense Business Essence Summary.
        
        CRITICAL GOAL: The ultimate goal of this summary is to condition an AI agent or a new human employee acting as a "{cfg['summary_goal']}".
        
        The summary must capture the operational DNA of the business tailored to this specific goal, so that downstream AI systems can reason correctly about customer interactions in this context.
        
        Do not rewrite the original document.
        Do not enrich yet.
        Only extract the structured business essence.
        
        INPUT
        You will receive a structured hierarchical business description.
        
        🔷 OUTPUT FORMAT (STRICT STRUCTURE REQUIRED)
        
        Return the output in the following structured format:
        
        1. Business Classification
        Primary Business Type: (e.g., Retail, Service, Healthcare, Rental, Hybrid, Subscription, etc.)
        Transaction Model: (One-time purchase / Appointment-based / Rental duration / Membership / Mixed)
        Physical / Digital / Hybrid:
        Geographic Constraints:
        Customer Interaction Channels: (In-person, phone, online, delivery, etc.)
        
        2. Revenue Model Structure
        Core Revenue Drivers:
        Secondary Revenue Drivers:
        Upsell Potential Areas:
        Price Tiering Structure (if present):
        Minimum Order / Eligibility Rules (if present):
        
        3. Operational Model
        Capacity Constraints:
        Time Constraints:
        Dependency Constraints:
        Peak Load Sensitivity:
        Resource Allocation Model:
        Fulfillment Model (if applicable):
        
        4. Customer Lifecycle Pattern
        Typical Entry Point:
        Data Required From Customer:
        Decision Friction Points:
        Repeat Customer Signals:
        Retention Opportunities:
        Complaint / Risk Exposure Areas:
        
        5. Risk & Compliance Surface
        Identity Verification Needs:
        Payment Risk Factors:
        Eligibility Checks:
        Safety / Liability Risks:
        Refund / Dispute Exposure:
        
        6. Behavioral Economics Signals
        Price Sensitivity Level:
        Urgency Level:
        Emotional Drivers:
        Trust Requirements:
        Convenience Sensitivity:
        
        7. Business Complexity Score (1–10)
        Explain briefly why.
        
        8. AI Interaction Implications for {cfg['summary_goal']}
        Based on this business structure and the goal of a {cfg['summary_goal']}, an AI agent interacting with customers must prioritize:
        (bullet list of critical reasoning capabilities required)
        
        Examples:
        - Slot availability validation
        - Quantity verification
        - Identity checks
        - Address validation
        - Cross-sell intelligence
        - Policy enforcement
        - Eligibility verification
        - Complaint resolution logic
        
        🔷 RULES
        Stay neutral and industry-agnostic.
        Do not hallucinate missing policies.
        If information is absent, state "Not specified".
        Keep output concise but dense.
        Think in systems, not descriptions.
        
        🔷 FINAL INSTRUCTION
        Now analyze the following business document and extract the Business Essence Summary:
        
        ═══════════════════════════════════════
        BUSINESS DOCUMENT:
        ═══════════════════════════════════════
        {full_text}
        ═══════════════════════════════════════
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  PROMPT ENGINEERING — 100% GENERIC (no hardcoded specifics)
# ═══════════════════════════════════════════════════════════════════════════

def build_generator_prompt(
    node: HeaderNode,
    cfg: dict,
    business_summary: str,
    rejected_facts: list[dict] | None = None,
    rejected_instructions: list[dict] | None = None,
) -> str:
    """
    Generic prompt — works for ANY business.
    All specifics come from the business_summary and the node itself.
    """

    node_type = "LEAF NODE (no sub-items)" if node.is_leaf else "CATEGORY NODE (has sub-items)"

    if node.is_leaf:
        instr_guidance = textwrap.dedent("""\
            This is a LEAF NODE — generate TACTICAL EXECUTION instructions.
            The employee is on the phone/counter RIGHT NOW with a customer.

            Based on the BUSINESS DATA provided, generate instructions that:
            - Tell the employee exactly what to SAY or DO regarding this specific option
            - Highlight critical constraints, prerequisites, or eligibility rules from the data
            - Provide specific upsell/cross-sell scripts using REAL items/services and exact prices
            - Include time, duration, or capacity estimates to quote to the customer
            - Define the "kill switch": when a customer should be steered AWAY from this option
            - Address potential customer objections or friction points using facts from the data""")
    else:
        instr_guidance = textwrap.dedent("""\
            This is a CATEGORY NODE — generate STRATEGIC ROUTING instructions.
            The employee is on the phone/counter RIGHT NOW with a customer.

            Based on the BUSINESS DATA provided, generate instructions that:
            - Ask qualifying questions to narrow down the customer's specific needs
            - Route the customer based on their constraints (e.g., budget, time, eligibility)
            - Transition smoothly to the specific sub-items/services that ACTUALLY EXIST
            - Advise on handling different customer profiles (e.g., individual vs. group, novice vs. expert)
            - Provide decision-tree logic to help indecisive customers choose the right option""")

    if node.is_leaf:
        fact_guidance = textwrap.dedent("""\
            Generate facts by EXTRACTING real information from the business data:
            - Exact pricing structures, tiers, or fees extracted FROM the file
            - Technical specifications, features, or included components
            - Time, duration, or operational capacity data relevant to this specific item
            - Practical knowledge that helps the employee confidently answer detailed customer questions
            - How this item/service interacts with or depends on other offerings""")
    else:
        fact_guidance = textwrap.dedent("""\
            Generate facts by EXTRACTING real information from the business data:
            - What specific sub-items/services fall under this category (from the file)
            - Price ranges, baselines, or comparative costs across this category
            - General policies, rules, or requirements that apply universally to this category
            - How options in this category compare to each other (the trade-offs)
            - Contextual information that helps the employee navigate this area of the business""")

    # Context from children text
    context_block = ""
    if node.children_text:
        context_block = f"\n\nDIRECT CONTENT under this header:\n{node.children_text}"

    # Refinement
    refinement_block = ""
    if rejected_facts or rejected_instructions:
        parts = ["\n\n--- REFINEMENT REQUIRED ---"]
        parts.append("The following items were REJECTED. Rewrite them using "
                      "REAL data from the business document.")
        if rejected_facts:
            parts.append("\nREJECTED FACTS:")
            for rf in rejected_facts:
                parts.append(f'  ✗ "{rf["text"]}" — Reason: {rf["reason"]}')
        if rejected_instructions:
            parts.append("\nREJECTED INSTRUCTIONS:")
            for ri in rejected_instructions:
                parts.append(f'  ✗ "{ri["text"]}" — Reason: {ri["reason"]}')
        refinement_block = "\n".join(parts)

    # Build dynamic JSON example placeholders
    instr_example = ", ".join(f'"instr{i+1}"' for i in range(cfg.get("num_instructions", 7)))
    fact_example = ", ".join(f'"fact{i+1}"' for i in range(cfg.get("num_facts", 5)))

    prompt = textwrap.dedent(f"""\
        You are an expert business trainer with 19 years of experience.
        You are training a new "{cfg["role"]}" (Goal: {cfg["summary_goal"]})
        at {cfg["business_name"]} ({cfg["business_domain"]}).

        ═══════════════════════════════════════
        BUSINESS SUMMARY (extracted from actual business data):
        ═══════════════════════════════════════
        {business_summary}



        ═══════════════════════════════════════
        CURRENT NODE TO ENRICH:
        ═══════════════════════════════════════
        Header: [{node.number}] {node.title}
        Level: {node.depth} (1=top, 2=sub-category, 3+=specific item)
        Type: {node_type}
        {context_block}

        ═══════════════════════════════════════════════════════════
        GENERATE EXACTLY {cfg['num_instructions']} INSTRUCTIONS
        ═══════════════════════════════════════════════════════════
        {instr_guidance}

        CRITICAL RULES:
        - Every instruction is what the EMPLOYEE says/does FACING A CUSTOMER.
        - NEVER write about staff training, meetings, or back-office tasks.
        - All items, prices, policies, and constraints MUST come from the business data above.
        - Do NOT invent products, prices, or facts not in the document.
        - Start every instruction with an imperative verb (Ask, Mention, Warn, Suggest, Confirm, etc.)

        ═══════════════════════════════════════════════════════════
        GENERATE EXACTLY {cfg['num_facts']} KNOWLEDGE FACTS
        ═══════════════════════════════════════════════════════════
        {fact_guidance}

        CRITICAL RULES:
        - Every fact MUST be extracted from or directly supported by the business data above.
        - Include REAL prices, REAL items, and REAL policies from the document.
        - Do NOT generate generic facts that could apply to any business.
        - Do NOT invent statistics, metrics, or operational data not in the document.

        {refinement_block}

        Respond in EXACTLY this JSON format, no markdown fencing:
        {{
            "instructions": [{instr_example}],
            "facts": [{fact_example}]
        }}
    """)
    return prompt


def build_critic_prompt(
    node: HeaderNode, cfg: dict, facts: list[str], instructions: list[str],
    business_summary: str,
) -> str:
    """Critic prompt — also gets business data to verify factual accuracy."""
    facts_block = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(facts))
    instr_block = "\n".join(f"  {i+1}. {inst}" for i, inst in enumerate(instructions))
    node_type = "LEAF" if node.is_leaf else "CATEGORY"

    prompt = textwrap.dedent(f"""\
        You are a ruthless quality critic. You have the ACTUAL business data below.
        You are grading this for the specific goal: {cfg["summary_goal"]}.
        Review the generated training material and REJECT anything that:
        - Contains invented facts NOT in the business data
        - Is NOT customer-facing (staff training, meetings, internal ops)
        - Is too vague or generic (could apply to any business)
        - Uses passive voice or doesn't start with an imperative verb (instructions)
        - Mentions products, prices, or policies NOT in the actual data

        ═══════════════════════════════════════
        BUSINESS SUMMARY (source of truth):
        ═══════════════════════════════════════
        {business_summary}

        ═══════════════════════════════════════
        NODE: [{node.number}] {node.title}  (Type: {node_type})
        ═══════════════════════════════════════

        === INSTRUCTIONS TO REVIEW ===
        {instr_block}

        === FACTS TO REVIEW ===
        {facts_block}

        For each item, output ACCEPT or REJECT with a reason.

        Respond in EXACTLY this JSON format, no markdown fencing:
        {{
            "instruction_verdicts": [
                {{"index": 0, "verdict": "ACCEPT|REJECT", "reason": "..."}},
                ...
            ],
            "fact_verdicts": [
                {{"index": 0, "verdict": "ACCEPT|REJECT", "reason": "..."}},
                ...
            ]
        }}
    """)
    return prompt


# ═══════════════════════════════════════════════════════════════════════════
# 6.  NODE ENRICHER — SEQUENTIAL (debug-friendly)
# ═══════════════════════════════════════════════════════════════════════════

class NodeEnricher:
    """Sequential: Generator → Critic → Refine. No async."""

    def __init__(self, client, cfg: dict, business_summary: str, full_text: str):
        self.client = client
        self.cfg = cfg
        self.model = cfg["model"]
        self.max_passes = cfg["max_passes"]
        self.dry_run = cfg["dry_run"]
        self.business_summary = business_summary
        self.full_text = full_text

    def llm_call(self, prompt: str, temperature: float = 0.7) -> str:
        """Synchronous LLM call. Set breakpoint here."""
        if self.dry_run:
            return self._mock_response(prompt)

        system_instruction = "You are an expert business trainer. Respond ONLY with valid JSON. No markdown fencing."

        def do_call():
            if self.model.startswith("gemini"):
                from google.genai import types
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=temperature,
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                    )
                )
                return response.text.strip()
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content.strip()
        
        return with_retries(do_call)

    def _mock_response(self, prompt: str) -> str:
        if '"instruction_verdicts"' in prompt or "REJECTION CRITERIA" in prompt:
            return json.dumps({
                "instruction_verdicts": [
                    {"index": i, "verdict": "ACCEPT", "reason": "OK"} for i in range(self.cfg.get("num_instructions", 7))
                ],
                "fact_verdicts": [
                    {"index": i, "verdict": "ACCEPT", "reason": "OK"} for i in range(self.cfg.get("num_facts", 5))
                ],
            })
        else:
            return json.dumps({
                "instructions": [f"[Mock instruction {i+1} — will be real with API]" for i in range(self.cfg.get("num_instructions", 7))],
                "facts": [f"[Mock fact {i+1} — will be real with API]" for i in range(self.cfg.get("num_facts", 5))],
            })

    @staticmethod
    def parse_json(raw: str) -> dict:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
            
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as first_err:
            # --- Attempt 1: Fix invalid backslash escapes (e.g. \', \J) ---
            def fix_escape(match):
                val = match.group(0)
                if val in ('\\\\', '\\"', '\\/', '\\b', '\\f', '\\n', '\\r', '\\t'):
                    return val
                if val.startswith('\\u') and len(val) == 6:
                    return val
                if val == "\\'":
                    return "'"
                return "\\" + val

            fixed = re.sub(r'\\u[0-9a-fA-F]{4}|\\.', fix_escape, cleaned)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

            # --- Attempt 2: Repair truncated JSON (unterminated strings/brackets) ---
            # The LLM response may have been cut off by max_output_tokens.
            repaired = fixed
            # Close any unterminated string by finding if we have an odd number of unescaped quotes
            in_string = False
            i = 0
            while i < len(repaired):
                ch = repaired[i]
                if ch == '\\' and in_string:
                    i += 2  # skip escaped char
                    continue
                if ch == '"':
                    in_string = not in_string
                i += 1
            if in_string:
                repaired += '"'

            # Close any open brackets/braces
            open_stack = []
            in_str = False
            j = 0
            while j < len(repaired):
                ch = repaired[j]
                if ch == '\\' and in_str:
                    j += 2
                    continue
                if ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch in ('{', '['):
                        open_stack.append(ch)
                    elif ch == '}' and open_stack and open_stack[-1] == '{':
                        open_stack.pop()
                    elif ch == ']' and open_stack and open_stack[-1] == '[':
                        open_stack.pop()
                j += 1

            # Close remaining open brackets in reverse order
            for opener in reversed(open_stack):
                repaired += ']' if opener == '[' else '}'

            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

            # If all repairs failed, print the raw response for debugging
            print(f"\n  ⚠️ JSON parse failed. Raw LLM response (first 500 chars):\n{raw[:500]}")
            raise first_err

    def generate(self, node: HeaderNode,
                 rejected_facts=None, rejected_instructions=None
    ) -> tuple[list[str], list[str]]:
        """Generate facts + instructions. Set breakpoint here."""
        prompt = build_generator_prompt(
            node, self.cfg, self.business_summary,
            rejected_facts, rejected_instructions,
        )
        raw = self.llm_call(prompt, temperature=0.7)
        try:
            data = self.parse_json(raw)
        except json.JSONDecodeError:
            print("    🔄 Retrying generate due to JSON parse error...")
            raw = self.llm_call(prompt, temperature=0.7)
            data = self.parse_json(raw)
        nf = self.cfg.get("num_facts", 5)
        ni = self.cfg.get("num_instructions", 7)
        return data.get("facts", [])[:nf], data.get("instructions", [])[:ni]

    def critique(self, node: HeaderNode, facts: list[str], instructions: list[str]
    ) -> tuple[list[dict], list[dict]]:
        """Critique generated content. Set breakpoint here."""
        prompt = build_critic_prompt(
            node, self.cfg, facts, instructions, self.business_summary,
        )
        raw = self.llm_call(prompt, temperature=0.2)
        try:
            data = self.parse_json(raw)
        except json.JSONDecodeError:
            print("    🔄 Retrying critique due to JSON parse error...")
            raw = self.llm_call(prompt, temperature=0.2)
            data = self.parse_json(raw)

        rejected_facts = []
        for v in data.get("fact_verdicts", []):
            if v.get("verdict", "").upper() == "REJECT":
                idx = v["index"]
                if idx < len(facts):
                    rejected_facts.append({"index": idx, "text": facts[idx], "reason": v.get("reason", "")})

        rejected_instructions = []
        for v in data.get("instruction_verdicts", []):
            if v.get("verdict", "").upper() == "REJECT":
                idx = v["index"]
                if idx < len(instructions):
                    rejected_instructions.append({"index": idx, "text": instructions[idx], "reason": v.get("reason", "")})

        return rejected_facts, rejected_instructions

    def enrich_node(self, node: HeaderNode) -> EnrichedNode:
        """Full pipeline for one node. Set breakpoint here."""
        result = EnrichedNode(
            number=node.number, title=node.title,
            depth=node.depth, is_leaf=node.is_leaf,
        )

        facts, instructions = self.generate(node)

        for pass_num in range(self.max_passes):
            rejected_facts, rejected_instructions = self.critique(node, facts, instructions)
            if not rejected_facts and not rejected_instructions:
                break
            result.refinement_passes = pass_num + 1
            new_facts, new_instructions = self.generate(
                node, rejected_facts=rejected_facts or None,
                rejected_instructions=rejected_instructions or None,
            )
            if rejected_facts:
                rejected_idx = {rf["index"] for rf in rejected_facts}
                ni = 0
                for i in range(len(facts)):
                    if i in rejected_idx and ni < len(new_facts):
                        facts[i] = new_facts[ni]; ni += 1
            if rejected_instructions:
                rejected_idx = {ri["index"] for ri in rejected_instructions}
                ni = 0
                for i in range(len(instructions)):
                    if i in rejected_idx and ni < len(new_instructions):
                        instructions[i] = new_instructions[ni]; ni += 1

        result.facts = facts
        result.instructions = instructions
        return result


# ═══════════════════════════════════════════════════════════════════════════
# 7.  OUTPUT FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════

def build_enriched_file(original_text, nodes, enriched):
    lines = original_text.splitlines()
    output_lines = []
    header_line_map = {}
    for node in nodes:
        if node.number in enriched:
            header_line_map[node.line_index] = enriched[node.number]
    for idx, line in enumerate(lines):
        output_lines.append(line)
        if idx in header_line_map:
            en = header_line_map[idx]
            label = "LEAF" if en.is_leaf else "CATEGORY"
            output_lines.append("")
            output_lines.append(f"[INSTRUCTIONS]  ({label})")
            for i, instr in enumerate(en.instructions, 1):
                output_lines.append(f"  {i}. {instr}")
            output_lines.append("")
            output_lines.append("[KNOWLEDGE FACTS]")
            for i, fact in enumerate(en.facts, 1):
                output_lines.append(f"  {i}. {fact}")
            output_lines.append("")
    return "\n".join(output_lines)


def format_json(enriched_nodes, cfg):
    data = {
        "system": "Dual-Knowledge Enrichment System v4",
        "business": cfg["business_name"],
        "domain": cfg["business_domain"],
        "target_role": cfg["role"],
        "total_nodes": len(enriched_nodes),
        "nodes": [
            {
                "number": n.number, "title": n.title, "depth": n.depth,
                "node_type": "leaf" if n.is_leaf else "category",
                "instructions": {"items": n.instructions},
                "knowledge_facts": {"items": n.facts},
                "meta": {"refinement_passes": n.refinement_passes},
            }
            for n in enriched_nodes
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# 8.  MAIN — SEQUENTIAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    cfg = CONFIG

    # ── Read input ────────────────────────────────────────────────────
    input_path = Path(cfg["input_file"])
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    full_text = input_path.read_text(encoding="utf-8")
    print(f"📄 Read {len(full_text):,} chars from {input_path.name}")

    # ── Parse headers ─────────────────────────────────────────────────
    nodes = parse_headers(full_text)
    if not nodes:
        print("❌ No headers found.")
        sys.exit(1)

    categories = [n for n in nodes if not n.is_leaf]
    leaves = [n for n in nodes if n.is_leaf]
    print(f"🔍 Parsed {len(nodes)} headers: {len(categories)} categories, {len(leaves)} leaves")
    for n in nodes:
        indent = "  " * (n.depth - 1)
        tag = "📂" if not n.is_leaf else "🍕"
        print(f"   {indent}{tag} [{n.number}] {n.title}")

    # ── Initialize LLM client ─────────────────────────────────────────
    client = None
    if not cfg["dry_run"]:
        if cfg["model"].startswith("gemini"):
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("❌ GEMINI_API_KEY not found. Check your .env file.")
                sys.exit(1)
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
            except ImportError:
                print("❌ google-genai not installed. Run: pip install google-genai")
                sys.exit(1)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("❌ OPENAI_API_KEY not found. Check: " + env_path)
                sys.exit(1)
            if OpenAI is None:
                print("❌ openai not installed. Run: pip install openai")
                sys.exit(1)
            client = OpenAI(api_key=api_key)

    mode = "🧪 DRY-RUN" if cfg["dry_run"] else f"🤖 LIVE ({cfg['model']})"
    print(f"\n{mode} | role=\"{cfg['role']}\"")
    print("─" * 65)

    # ══════════════════════════════════════════════════════════════════
    # STEP 0: Extract business summary from the FULL file
    # ══════════════════════════════════════════════════════════════════
    print("\n📋 Step 0: Extracting business summary from file...")
    if cfg["dry_run"]:
        business_summary = "[DRY-RUN] Mock business summary."
    else:
        summary_prompt = build_summary_prompt(full_text, cfg)
        system_instruction = "You are a business analyst. Respond with plain text."
        
        def do_summary_call():
            if cfg["model"].startswith("gemini"):
                from google.genai import types
                response = client.models.generate_content(
                    model=cfg["model"],
                    contents=summary_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                        max_output_tokens=3000,
                    )
                )
                return response.text.strip()
            else:
                response = client.chat.completions.create(
                    model=cfg["model"],
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": summary_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=3000,
                )
                return response.choices[0].message.content.strip()
        
        business_summary = with_retries(do_summary_call)

    print(f"   ✅ Summary extracted ({len(business_summary)} chars)")
    print(f"   Preview: {business_summary[:200]}...")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Enrich each node SEQUENTIALLY
    # ══════════════════════════════════════════════════════════════════
    enricher = NodeEnricher(
        client=client, cfg=cfg,
        business_summary=business_summary,
        full_text=full_text,
    )

    print(f"\n🔧 Enriching {len(nodes)} nodes sequentially...")
    print("─" * 65)

    enriched_list: list[EnrichedNode] = []
    for i, node in enumerate(nodes):
        tag = "📂 CATEGORY" if not node.is_leaf else "🍕 LEAF"
        print(f"\n  [{i+1}/{len(nodes)}] {tag}: [{node.number}] {node.title}")

        enriched = enricher.enrich_node(node)  # <-- breakpoint here
        enriched_list.append(enriched)

        print(f"    ✅ {len(enriched.instructions)} instructions, "
              f"{len(enriched.facts)} facts"
              f"{f', {enriched.refinement_passes} refinements' if enriched.refinement_passes else ''}")
        
        # Pacing delay to avoid exhausting RPM
        if cfg.get("pacing_delay", 0) > 0 and i < len(nodes) - 1:
            time.sleep(cfg["pacing_delay"])

    # ── Sort and build output ─────────────────────────────────────────
    enriched_list.sort(key=lambda n: [int(x) for x in n.number.split(".")])
    enriched_map = {n.number: n for n in enriched_list}

    enriched_text = build_enriched_file(full_text, nodes, enriched_map)
    out_path = Path(cfg["output_file"])
    out_path.write_text(enriched_text, encoding="utf-8")
    print(f"\n📝 Enriched file → {out_path}  ({len(enriched_text):,} chars)")

    json_content = format_json(enriched_list, cfg)
    json_path = Path(cfg["json_output_file"])
    json_path.write_text(json_content, encoding="utf-8")
    print(f"📦 JSON output   → {json_path}  ({len(json_content):,} chars)")

    total_instr = sum(len(n.instructions) for n in enriched_list)
    total_facts = sum(len(n.facts) for n in enriched_list)
    total_refines = sum(n.refinement_passes for n in enriched_list)

    print(f"\n{'═' * 65}")
    print(f"  ✅ ENRICHMENT COMPLETE")
    print(f"     Nodes:         {len(enriched_list)} ({len(categories)} cat, {len(leaves)} leaf)")
    print(f"     Instructions:  {total_instr}")
    print(f"     Facts:         {total_facts}")
    print(f"     Refinements:   {total_refines}")
    print(f"{'═' * 65}")


if __name__ == "__main__":
    main()
