# orchestration.py — Refactored for Microsoft Agent Framework (agent_framework)
#
# AG2 → MAF mapping:
#   ConversableAgent (orchestrator)         → Agent (orchestrator) via OpenAIChatCompletionClient
#   AutoPattern + initiate_group_chat       → WorkflowBuilder with explicit edge routing
#   OnCondition + StringLLMCondition        → LLM-driven router executor (select_agent)
#   handoffs.add_llm_conditions(conditions) → WorkflowBuilder.add_edge() calls (PROPAGATOR-ready)
#   UserProxyAgent                          → removed; entry via workflow.run()
#   apply_safeguard_policy                  → inline safeguard check in router executor
#   max_rounds=10                           → max_iterations guard in router loop
#   result.chat_history parsing             → direct ctx.yield_output() from leaf agents
#   LLMConfig.from_json                     → OpenAIChatCompletionClient (reads from env)

import asyncio
import json
from typing_extensions import Never

from agent_framework import Agent, WorkflowBuilder, executor, WorkflowContext
from agent_framework.openai import OpenAIChatCompletionClient

from adviceAgent import build_advice_agent
from dbAgent import build_db_agent, DatabaseConnection
from diagnosisAgent import build_diagnosis_agent
from priceAgent import build_price_agent
from statsAgent import build_stat_agent
from utils import load_prompts, load_safeguards
from messageCleanser import OutputCleanser

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

prompts = load_prompts()
safeguards = load_safeguards()
client = OpenAIChatCompletionClient(model="gpt-4o")  # swap model/endpoint as needed

# ---------------------------------------------------------------------------
# Agent instantiation
# NOTE: Each build_*_agent() factory is defined in the refactored agent files.
# ---------------------------------------------------------------------------

advice_agent    = build_advice_agent(prompts["advice"])
db_agent        = build_db_agent(prompts["db"], DatabaseConnection())
diagnosis_agent = build_diagnosis_agent(prompts["diagnosis"])
price_agent     = build_price_agent(prompts["price"])
stat_agent      = build_stat_agent(prompts["stats"])

# Orchestrator: LLM-based router that selects the right specialist
orchestrator_agent = client.as_agent(
    name="orchestrator",
    instructions=prompts["orchestrator"]["instructions"],
    default_options={"temperature": 0.0},
)

# Registry maps agent name → agent instance (PROPAGATOR can inspect this dict directly)
AGENT_REGISTRY: dict[str, Agent] = {
    "AdviceAgent":    advice_agent,
    "DBAgent":        db_agent,
    "DiagnosisAgent": diagnosis_agent,
    "PriceAgent":     price_agent,
    "StatAgent":      stat_agent,
}

# ---------------------------------------------------------------------------
# Safeguard helper
# Replaces: apply_safeguard_policy(agents=[...], policy="safeguards.json", ...)
# ---------------------------------------------------------------------------

def _passes_safeguards(message: str) -> bool:
    """
    Inline safeguard check against loaded safeguards.json policy.
    Extend this with an LLM-based policy call if your safeguards require it.
    """
    blocked_patterns = safeguards.get("blocked_patterns", [])
    message_lower = message.lower()
    return not any(pattern.lower() in message_lower for pattern in blocked_patterns)


# ---------------------------------------------------------------------------
# Router executor
# Replaces: AutoPattern + OnCondition + StringLLMCondition + handoffs
#
# This is the node PROPAGATOR should instrument — it's the explicit decision
# point where the graph edge is selected at runtime. You can add_edge() calls
# around this executor in WorkflowBuilder to wire in PROPAGATOR observers.
# ---------------------------------------------------------------------------

@executor(id="router")
async def router_executor(
    task: str,
    ctx: WorkflowContext[Never, str],
) -> None:
    output_cleanser = OutputCleanser()
    max_iterations = 10
    current_task = task

    for _ in range(max_iterations):

        # Safeguard gate (mirrors apply_safeguard_policy on orchestratorAgent)
        if not _passes_safeguards(current_task):
            await ctx.yield_output(
                "Your request was flagged as potentially violating our safety regulations. "
                "Please try again with a different prompt."
            )
            return

        # Orchestrator selects which specialist to route to
        # Returns a JSON blob: {"agent": "<AgentName>", "refined_task": "<...>"}
        routing_response = await orchestrator_agent.run(
            f"Given the following user message, return a JSON object with:\n"
            f"  'agent': one of {list(AGENT_REGISTRY.keys())}\n"
            f"  'refined_task': the message rewritten for that specialist\n\n"
            f"Message: {current_task}"
        )

        try:
            routing = json.loads(routing_response.text)
            selected_name = routing["agent"]
            refined_task  = routing.get("refined_task", current_task)
        except (json.JSONDecodeError, KeyError):
            # Orchestrator returned something unparseable — surface the error
            await ctx.yield_output(
                "I'm sorry, there was an error in the response. Please try again."
            )
            return

        if selected_name not in AGENT_REGISTRY:
            await ctx.yield_output(
                "I'm sorry, there was an error in the response. Please try again."
            )
            return

        # Dispatch to selected specialist agent
        # NOTE: This is the explicit graph edge — PROPAGATOR hooks live here.
        selected_agent = AGENT_REGISTRY[selected_name]
        specialist_result = await selected_agent.run(refined_task)
        raw_reply = specialist_result.text or ""

        if not raw_reply:
            await ctx.yield_output(
                "I'm sorry, there was an error in the response. Please try again."
            )
            return

        clean_reply = output_cleanser.cleanOutput(raw_reply)

        print(f"ROUTING → {selected_name}")
        print(f"REPLY: {raw_reply}")
        print(f"CLEAN REPLY: {clean_reply}")

        await ctx.yield_output(clean_reply)
        return  # single-turn orchestration; remove return for multi-turn loops

    # Exceeded max_iterations
    await ctx.yield_output(
        "I'm sorry, there was an error in the response. Please try again."
    )


# ---------------------------------------------------------------------------
# Workflow graph (explicit — PROPAGATOR-ready)
#
# WorkflowBuilder exposes the graph as Python — add edges here to wire in
# PROPAGATOR instrumentation without reverse-engineering hidden routing.
#
# Example PROPAGATOR hook (add before .build()):
#   builder.add_edge("router", "propagator_observer")
# ---------------------------------------------------------------------------

def build_workflow():
    builder = WorkflowBuilder(start_executor=router_executor)
    # PROPAGATOR: add_edge() calls go here as you build the instrumentation layer
    return builder.build()


workflow = build_workflow()


# ---------------------------------------------------------------------------
# Public interface (called by userMessageBuilder.py → orchestrate())
# ---------------------------------------------------------------------------

async def orchestrate_async(full_message_with_context: str) -> str:
    result = await workflow.run(full_message_with_context)
    return result


def orchestrate(full_message_with_context: str) -> str:
    """
    Synchronous wrapper so userMessageBuilder.py can call orchestrate()
    without changes. MAF is async-native; this bridges the gap.
    """
    return asyncio.run(orchestrate_async(full_message_with_context))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    message = (
        "I am a 55 year old pregnant woman who smokes, "
        "can you give me some healthcare advice?"
    )
    result = orchestrate(message)
    print(result)

"""
A few decisions worth knowing about:

AutoPattern + initiate_group_chat is replaced by the router_executor — a single explicit node that asks the orchestrator LLM to pick a specialist, then dispatches to it. This is the node PROPAGATOR should instrument; the edge selection is right there in plain Python, not buried in AG2's pattern internals.

apply_safeguard_policy becomes _passes_safeguards() — currently a string-match implementation. If your safeguards.json policy requires an LLM eval pass, that's a one-function swap.

orchestrate() is kept synchronous via asyncio.run() so userMessageBuilder.py doesn't need to change. Once you're ready to go fully async end-to-end, drop the wrapper.

AGENT_REGISTRY is exposed as a plain dict — PROPAGATOR can iterate it, inspect edges, or inject observers without touching any hidden internals.

adviceAgent and diagnosisAgent aren't in the files you've sent yet — their build_* factories follow the same pattern as priceAgent and statsAgent. They'll slot right in when those files come through.
"""