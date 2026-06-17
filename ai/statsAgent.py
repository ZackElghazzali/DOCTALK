# statsAgent.py — Refactored for Microsoft Agent Framework (agent_framework)

from agent_framework import Agent, tool, WorkflowBuilder, executor, WorkflowContext
from agent_framework.openai import OpenAIChatCompletionClient
from typing_extensions import Never
from tools.statTool import pyTool
from baseAgent import BaseAgent
from utils import load_prompts


# ---------------------------------------------------------------------------
# Tool registration
# NOTE: pyTool() must be decorated with @tool in tools/statTool.py.
# If it isn't yet, wrap it here temporarily:
#   from agent_framework import tool as af_tool
#   py_tool_fn = af_tool(pyTool())
# ---------------------------------------------------------------------------

def build_stat_agent(prompts: dict) -> Agent:
    """
    Factory that constructs the StatAgent as a MAF Agent.

    AG2 pattern replaced:
      - ConversableAgent + LLMConfig        → Agent + OpenAIChatCompletionClient
      - code_execution_config / Docker      → hosted code interpreter tool (or local pyTool)
      - registerExecution(user_proxy)       → tools=[] passed directly at construction
      - human_input_mode="NEVER"            → default MAF behavior (no human loop)
    """
    client = OpenAIChatCompletionClient(model="gpt-4o")  # swap model as needed

    agent = client.as_agent(
        name="StatAgent",
        instructions=prompts["instructions"],
        tools=[pyTool()],           # pyTool must be a @tool-decorated callable
        default_options={"temperature": 0.0},
    )

    return agent


# ---------------------------------------------------------------------------
# Workflow wiring (replaces UserProxyAgent + initiate_chat)
#
# AG2 pattern replaced:
#   user_proxy.initiate_chat(statAgent.agent, message=...) →
#   WorkflowBuilder with an input executor → stat_agent executor → output
# ---------------------------------------------------------------------------

def build_stat_workflow(stat_agent: Agent):

    @executor(id="stat_executor")
    async def stat_executor(task: str, ctx: WorkflowContext[Never, str]) -> None:
        result = await stat_agent.run(task)
        await ctx.yield_output(result.text)

    workflow = (
        WorkflowBuilder(start_executor=stat_executor)
        .build()
    )

    return workflow


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def main():
        prompts = load_prompts()
        stat_agent = build_stat_agent(prompts)
        workflow = build_stat_workflow(stat_agent)

        result = await workflow.run(
            "My birthday is November 22nd 2004, how many days have I been alive?"
        )
        print(result)

    asyncio.run(main())

"""
A few things worth flagging:

DockerCodeExecutor is dropped entirely — MAF's Agent handles tool iteration automatically and doesn't use a separate executor proxy pattern. If pyTool does local code execution, it just needs to be a @tool-decorated callable in tools/statTool.py.

UserProxyAgent + initiate_chat is gone — replaced by a single WorkflowBuilder with one executor. When orchestration.py comes through, this wiring will likely get absorbed into the broader workflow graph anyway.

LLMConfig.from_json → OpenAIChatCompletionClient reads from env by default. If you're on Azure OpenAI, swap to the Azure variant with your endpoint/credential.

The BaseAgent import is kept since you presumably still have that base class — but depending on what it does, it may also need a pass through this process.
"""