# adviceAgent.py — Refactored for Microsoft Agent Framework (agent_framework)
# Agent specializing in healthcare advice
# uses adviceApiTool and webParseTool (HTMLParserTool) for grounded responses

from agent_framework import Agent, WorkflowBuilder, executor, WorkflowContext
from agent_framework.openai import OpenAIChatCompletionClient
from typing_extensions import Never
from tools.adviceApiTool import AdviceAPITool
from tools.webParseTool import HTMLParserTool


# ---------------------------------------------------------------------------
# Agent factory
#
# Mirrors the pattern established in priceAgent, diagnosisAgent, dbAgent.
# BaseAgent subclassing dropped — MAF Agent is the base.
#
# NOTE: AdviceAPITool and HTMLParserTool must be @tool-decorated callables
# in their respective tool files for MAF to register them correctly.
# ---------------------------------------------------------------------------

def build_advice_agent(prompts: dict) -> Agent:
    """
    Factory that constructs the AdviceAgent as a MAF Agent.
    """
    client = OpenAIChatCompletionClient(model="gpt-4o")  # swap model as needed

    agent = client.as_agent(
        name="AdviceAgent",
        instructions=prompts["instructions"],
        tools=[AdviceAPITool(), HTMLParserTool()],
        default_options={"temperature": 0.0},
    )

    return agent


# ---------------------------------------------------------------------------
# Workflow wiring (replaces UserProxyAgent + initiate_chat)
# ---------------------------------------------------------------------------

def build_advice_workflow(advice_agent: Agent):

    @executor(id="advice_executor")
    async def advice_executor(task: str, ctx: WorkflowContext[Never, str]) -> None:
        result = await advice_agent.run(task)
        await ctx.yield_output(result.text)

    workflow = (
        WorkflowBuilder(start_executor=advice_executor)
        .build()
    )

    return workflow


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    from utils import load_prompts

    async def main():
        prompts = load_prompts()
        advice_agent = build_advice_agent(prompts["advice"])
        workflow = build_advice_workflow(advice_agent)

        result = await workflow.run(
            "I am a 55 year old pregnant woman who smokes, "
            "can you give me some healthcare advice?"
        )
        print(result)

    asyncio.run(main())

"""
Clean and consistent with every other agent in the set. The last missing piece in orchestration.py was build_advice_agent — this completes it. All five specialist agents now have their factories defined and AGENT_REGISTRY in orchestration.py is fully wired up.
"""