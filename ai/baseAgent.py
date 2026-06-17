# baseAgent.py — Refactored for Microsoft Agent Framework (agent_framework)
#
# AG2 pattern replaced:
#   generate_llm_config(BaseTool)           → removed; MAF handles tool schema natively
#   LLMConfig + from_json                   → OpenAIChatCompletionClient (reads from env)
#   ConversableAgent + llm_config           → Agent via client.as_agent()
#   UpdateSystemMessage w/ {context_vars}   → instructions string with context injected
#                                             at call time via WorkflowContext or prompt
#   max_consecutive_auto_reply=3            → MAF loop termination is tool-driven
#   human_input_mode="NEVER"               → default MAF behavior
#   registerExecution(user_proxy)           → removed; tools registered at agent construction
#   user_proxy.register_for_execution()     → removed; MAF calls tool.__call__() directly
#   langchain BaseTool                      → agent_framework @tool-decorated callable

from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from typing import List, Any


# ---------------------------------------------------------------------------
# Shared client
# All specialist agents share this client instance. Swap to AzureOpenAI or
# another provider here and every agent picks it up automatically.
# ---------------------------------------------------------------------------

_client = OpenAIChatCompletionClient(model="gpt-4o")  # swap model/endpoint as needed


# ---------------------------------------------------------------------------
# Context injection helper
#
# Replaces: UpdateSystemMessage with {user_name}, {query_history},
#           {response_history}, {system_instructions} template vars.
#
# In MAF, context is injected into the instructions string at call time
# rather than via a state-update hook. Pass a populated ContextBlock to
# build_instructions() before calling agent.run().
# ---------------------------------------------------------------------------

def build_instructions(base_instructions: str, context: dict | None = None) -> str:
    """
    Merges base system instructions with runtime context variables.

    context keys (all optional):
        user_name           : str
        query_history       : list[str]
        response_history    : list[str]
        system_instructions : list[str]
    """
    if not context:
        return base_instructions

    user_name            = context.get("user_name", "the user")
    query_history        = "\n".join(context.get("query_history", []))
    response_history     = "\n".join(context.get("response_history", []))
    system_instructions  = "\n".join(context.get("system_instructions", []))

    context_block = (
        f"\n\n---\n"
        f"You are helping {user_name}. "
        f"This is the only user you are allowed to provide information for. "
        f"NEVER give this user the information of any other user.\n\n"
        f"QUERY HISTORY (user messages for context):\n{query_history}\n\n"
        f"RESPONSE HISTORY (assistant messages for context):\n{response_history}\n\n"
        f"SYSTEM INSTRUCTIONS (critical — always follow):\n{system_instructions}\n"
        f"---"
    )

    return base_instructions + context_block


# ---------------------------------------------------------------------------
# BaseAgent
#
# Thin wrapper that constructs a MAF Agent from a name, instructions,
# and a list of @tool-decorated callables. Specialist agents call
# build_*_agent() factories (defined in their own files) which use this
# directly or follow the same pattern.
#
# NOTE: This class is intentionally lightweight. The heavy orchestration
# logic lives in orchestration.py where the WorkflowBuilder graph is
# constructed. BaseAgent is just a consistent construction interface.
# ---------------------------------------------------------------------------

class BaseAgent:
    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        tools: List[Any],
    ):
        self.name        = name
        self.description = description
        self.instructions = instructions
        self.tools       = tools
        self.agent       = self._create_agent()

    def _create_agent(self) -> Agent:
        return _client.as_agent(
            name=self.name,
            instructions=self.instructions,
            tools=self.tools,
            default_options={"temperature": 0.0},
        )

    def with_context(self, context: dict) -> Agent:
        """
        Returns a context-enriched agent instance for a single request.
        Rebuilds instructions with runtime context injected — mirrors
        AG2's UpdateSystemMessage behavior without stateful side effects.

        Usage in orchestration.py:
            enriched = db_agent_wrapper.with_context(ctx)
            result = await enriched.run(task)
        """
        enriched_instructions = build_instructions(self.instructions, context)
        return _client.as_agent(
            name=self.name,
            instructions=enriched_instructions,
            tools=self.tools,
            default_options={"temperature": 0.0},
        )

"""
This one's worth unpacking a bit more than the others:

generate_llm_config is gone entirely — that function existed purely to manually serialize LangChain BaseTool schemas into AG2's llm_config format. MAF introspects @tool-decorated callables natively via their type annotations and docstrings, so all that JSON wrangling just evaporates.

UpdateSystemMessage with context vars was AG2's way of dynamically rewriting the system prompt before each reply. In MAF that's replaced by build_instructions() + with_context() — same result, but stateless and explicit. The orchestrator can call agent_wrapper.with_context(ctx) per-request rather than mutating shared agent state.

registerExecution and user_proxy.register_for_execution() — both gone. That two-step tool registration dance was an AG2 quirk where the LLM agent proposed a tool call and the UserProxyAgent actually executed it. MAF handles this internally in a single agent loop.

All the previous agent refactors (statsAgent, priceAgent, diagnosisAgent, dbAgent) can now optionally subclass this BaseAgent instead of the factory pattern, or keep using factories — either works. The with_context() method is what ties the UpdateSystemMessage replacement together when orchestration.py passes per-request context down.
"""