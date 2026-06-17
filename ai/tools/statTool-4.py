import tempfile
import os
from pathlib import Path
from typing import Any
from agent_framework import tool
from agent_framework.coding import DockerCodeExecutor, CodeBlock


class DockerSandbox(DockerCodeExecutor):
    """
    Sandboxed Docker executor with resource limits and cleanup.
    AG2 equivalent: autogen.coding.DockerCommandLineCodeExecutor
    The constructor args and sandbox constraints are identical —
    only the import path changes from autogen.coding to agent_framework.coding.
    """

    def __init__(self, timeout: int = 10):
        self.temp_dir = tempfile.TemporaryDirectory()
        super().__init__(
            work_dir=Path("/executor"),
            bind_dir=Path(os.environ["HOST_WORKDIR"]),
            timeout=timeout,
            container_create_kwargs={
                "network_mode": "none",
                "mem_limit": "256m",
                "cpu_quota": 50000,
                "read_only": False,
                "security_opt": ["no-new-privileges:true"],
                "cap_drop": ["ALL"],
                "user": "nobody",
            },
            auto_remove=True,
        )

    def execute_code_blocks(self, code_blocks):
        result = super().execute_code_blocks(code_blocks)
        for f in Path("/executor").glob("tmp_code_*"):
            f.unlink(missing_ok=True)
        return result

    def __exit__(self, *args):
        super().__exit__(*args)
        self.temp_dir.cleanup()


class pyTool:
    """
    Executes Python code inside a sandboxed Docker container.
    AG2 equivalent: pyTool(BaseTool) with pyInput(BaseModel) args_schema.
    In MAF, parameters are declared directly on __call__ — no separate input model needed.
    """

    @tool
    def __call__(self, code: str) -> str:
        """
        Execute Python code in a sandboxed Docker container.
        You MUST use print() to output results.

        Args:
            code: The Python code to execute. Use print() for all output.
        """
        try:
            code = code.replace("\\n", "\n")
            code_block = CodeBlock(code=code, language="python")
            with DockerSandbox() as executor:
                result = executor.execute_code_blocks([code_block])
            if result.exit_code != 0:
                return f"Error (exit {result.exit_code}): {result.output}"
            return result.output if result.output else "Code executed successfully but produced no output"
        except Exception as e:
            return f"Execution error: {str(e)}"


if __name__ == "__main__":
    stat_tool = pyTool()
    result = stat_tool(
        code=(
            "def is_prime(n):\n"
            "    if n <= 1:\n"
            "        return False\n"
            "    for i in range(2, int(n**0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n\n"
            "primes = []\n"
            "i = 2\n"
            "while len(primes) < 15:\n"
            "    if is_prime(i):\n"
            "        primes.append(i)\n"
            "    i += 1\n"
            "print(primes[:15])"
        )
    )
    print(result)
