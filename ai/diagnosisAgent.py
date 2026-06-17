# diagnosisAgent.py — Refactored for Microsoft Agent Framework (agent_framework)
# Agent specializing in medical condition diagnosis information
# uses diagnosisApiTool to query the diagnosis API

from agent_framework import Agent, WorkflowBuilder, executor, WorkflowContext
from agent_framework.openai import OpenAIChatCompletionClient
from typing_extensions import Never
from tools.diagnosisApiTool import DiagnosisAPITool


# ---------------------------------------------------------------------------
# Agent factory
#
# AG2 pattern replaced:
#   ConversableAgent + LLMConfig + UserProxyAgent + registerExecution() →
#   Agent with tools passed directly at construction
# ---------------------------------------------------------------------------

def build_diagnosis_agent(prompts: dict) -> Agent:
    """
    Factory that constructs the DiagnosisAgent as a MAF Agent.

    NOTE: DiagnosisAPITool must be a @tool-decorated callable in
    tools/diagnosisApiTool.py for MAF to register it correctly.
    """
    client = OpenAIChatCompletionClient(model="gpt-4o")  # swap model as needed

    agent = client.as_agent(
        name="DiagnosisAgent",
        instructions=prompts["instructions"],
        tools=[DiagnosisAPITool()],
        default_options={"temperature": 0.0},
    )

    return agent


# ---------------------------------------------------------------------------
# Workflow wiring (replaces UserProxyAgent + initiate_chat)
# ---------------------------------------------------------------------------

def build_diagnosis_workflow(diagnosis_agent: Agent):

    @executor(id="diagnosis_executor")
    async def diagnosis_executor(task: str, ctx: WorkflowContext[Never, str]) -> None:
        result = await diagnosis_agent.run(task)
        await ctx.yield_output(result.text)

    workflow = (
        WorkflowBuilder(start_executor=diagnosis_executor)
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
        diagnosis_agent = build_diagnosis_agent(prompts["diagnosis"])
        workflow = build_diagnosis_workflow(diagnosis_agent)

        result = await workflow.run(
            "Could you tell me about the medical condition called lupus?"
        )
        print(result)

    asyncio.run(main())

"""
Nothing surprising here — UserProxyAgent, LLMConfig, and registerExecution() all gone, DiagnosisAPITool passes straight through as a tool. The only thing worth a second look when diagnosisApiTool.py comes through is whether DiagnosisAPITool() returns a callable or a class instance — MAF needs it decorated with @tool
"""