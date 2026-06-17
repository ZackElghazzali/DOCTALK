# priceAgent.py — Refactored for Microsoft Agent Framework (agent_framework)
# Agent specializing in medication price lookup from CostPlusDrugs
# uses webParseTool to search for medications using DrugPriceLookupTool

from agent_framework import Agent, WorkflowBuilder, executor, WorkflowContext
from agent_framework.openai import OpenAIChatCompletionClient
from typing_extensions import Never
from tools.webParseTool import DrugPriceLookupTool
from baseAgent import BaseAgent


# ---------------------------------------------------------------------------
# Agent factory
#
# AG2 pattern replaced:
#   ConversableAgent + UserProxyAgent + registerExecution()  →
#   Agent with tools passed directly at construction
# ---------------------------------------------------------------------------

def build_price_agent(prompts: dict) -> Agent:
    """
    Factory that constructs the PriceAgent as a MAF Agent.

    NOTE: DrugPriceLookupTool must be a @tool-decorated callable in
    tools/webParseTool.py for MAF to register it correctly.
    """
    client = OpenAIChatCompletionClient(model="gpt-4o")  # swap model as needed

    agent = client.as_agent(
        name="PriceAgent",
        instructions=prompts["instructions"],
        tools=[DrugPriceLookupTool()],
        default_options={"temperature": 0.0},
    )

    return agent


# ---------------------------------------------------------------------------
# Workflow wiring (replaces UserProxyAgent + initiate_chat)
# ---------------------------------------------------------------------------

def build_price_workflow(price_agent: Agent):

    @executor(id="price_executor")
    async def price_executor(task: str, ctx: WorkflowContext[Never, str]) -> None:
        result = await price_agent.run(task)
        await ctx.yield_output(result.text)

    workflow = (
        WorkflowBuilder(start_executor=price_executor)
        .build()
    )

    return workflow

"""
Same notes as statsAgent.py apply:

UserProxyAgent is fully dropped — no proxy needed in MAF, tools are called directly by the agent during its run loop.

BaseAgent import is removed here since MAF's Agent is the base — if your BaseAgent carries any custom logic beyond AG2 boilerplate, flag it when that file comes through and we'll sort out what to preserve.

DrugPriceLookupTool() needs the @tool decorator in webParseTool.py just like pyTool.
"""