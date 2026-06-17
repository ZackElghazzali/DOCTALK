# dbAgent.py — Refactored for Microsoft Agent Framework (agent_framework)
# Database agent that queries and manages patient records
# extends no base class — MAF Agent is the base

from agent_framework import Agent, WorkflowBuilder, executor, WorkflowContext
from agent_framework.openai import OpenAIChatCompletionClient
from typing_extensions import Never
from tools.dbTool import (
    DatabaseConnection,
    Patient,
    MedicalHistory,
    Appointment,
    Prescription,
    QueryPatientInfoTool,
    QueryMedicalHistoryTool,
    QueryAppointmentsTool,
    QueryPrescriptionsTool,
    UpdatePatientRecordTool,
    AddPatientRecordTool,
    DeletePatientRecordTool,
)


# ---------------------------------------------------------------------------
# Agent factory
#
# AG2 pattern replaced:
#   ConversableAgent + LLMConfig + UserProxyAgent + registerExecution() →
#   Agent with tools passed directly at construction
#
# NOTE: All Tool classes imported from tools/dbTool.py must be
# @tool-decorated callables for MAF to register them correctly.
# Each tool that takes db_connection or models at init should be
# instantiated here and passed as a bound callable.
# ---------------------------------------------------------------------------

def build_db_agent(prompts: dict, db_connection: DatabaseConnection) -> Agent:
    """
    Factory that constructs the DBAgent as a MAF Agent.
    db_connection is injected at construction so each tool gets
    a consistent connection reference across the agent's lifetime.
    """
    client = OpenAIChatCompletionClient(model="gpt-4o")  # swap model as needed

    models = {
        "patients":        Patient,
        "medical_history": MedicalHistory,
        "appointments":    Appointment,
        "prescriptions":   Prescription,
    }

    tools = [
        QueryPatientInfoTool(db_connection),
        QueryMedicalHistoryTool(db_connection),
        QueryAppointmentsTool(db_connection),
        QueryPrescriptionsTool(db_connection),
        UpdatePatientRecordTool(db_connection, models),
        AddPatientRecordTool(db_connection, models),
        DeletePatientRecordTool(db_connection, models),
    ]

    agent = client.as_agent(
        name="DBAgent",
        instructions=prompts["instructions"],
        tools=tools,
        default_options={"temperature": 0.0},
    )

    return agent


# ---------------------------------------------------------------------------
# Workflow wiring (replaces UserProxyAgent + initiate_chat)
# ---------------------------------------------------------------------------

def build_db_workflow(db_agent: Agent):

    @executor(id="db_executor")
    async def db_executor(task: str, ctx: WorkflowContext[Never, str]) -> None:
        result = await db_agent.run(task)
        await ctx.yield_output(result.text)

    workflow = (
        WorkflowBuilder(start_executor=db_executor)
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
        db_agent = build_db_agent(prompts["db"], DatabaseConnection())
        workflow = build_db_workflow(db_agent)

        result = await workflow.run(
            "Pull up the medical history for patient ID 1042."
        )
        print(result)

    asyncio.run(main())

"""
One thing worth flagging here specifically: the models dict that maps table names to SQLAlchemy/ORM model classes is passed into the write tools (Update, Add, Delete). That pattern carries over cleanly — just make sure those tool classes are @tool-decorated in dbTool.py while still accepting db_connection and models in their __init__. The @tool decorator in MAF wraps the __call__ method, not the constructor, so the dependency injection pattern survives intact.
"""