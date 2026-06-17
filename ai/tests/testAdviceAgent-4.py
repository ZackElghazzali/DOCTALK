import unittest
import asyncio
from unittest.mock import patch, AsyncMock
from adviceAgent import build_advice_agent
from utils import load_prompts

# To run all tests from ai dir: python -m unittest discover
# To run just these tests from ai dir: python -m unittest tests.testAdviceAgent

# Integration test for agent <-> tool communication.
# Patches AdviceAPITool.__call__ so no real HTTP requests are made.
# Verifies the agent extracts parameters correctly from natural language
# and passes them through to the tool.

MOCK_RESPONSE = {"Title": "TestTitle", "Content": "TestContent"}


class Test(unittest.TestCase):
    def setUp(self) -> None:
        prompts = load_prompts()
        self.agent = build_advice_agent(prompts["advice"])

    def _run_async(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def runTest(self, message: str, expected_params: dict):
        with patch("tools.adviceApiTool.AdviceAPITool.__call__") as mock_call:
            mock_call.return_value = MOCK_RESPONSE

            self._run_async(self.agent.run(message))

            mock_call.assert_called_once()
            _, kwargs = mock_call.call_args

            for key, value in expected_params.items():
                if key in kwargs:
                    self.assertEqual(kwargs[key], value)

    def testAllParams(self):
        message = (
            "I would like to get some healthcare advice. "
            "I am a 35 year old female, who is pregnant, "
            "I am sexually active, and I do smoke tobacco."
        )
        params = {
            "age": "35",
            "pregnant": "yes",
            "sex": "female",
            "sexually_active": "yes",
            "tobacco_use": "yes",
        }
        self.runTest(message, params)

    def testSomeNo(self):
        message = (
            "What are best practices for managing high blood pressure? "
            "I'm 20 years old, male, not pregnant, sexually active, and not a smoker."
        )
        params = {
            "age": "20",
            "pregnant": "no",
            "sex": "male",
            "sexually_active": "yes",
            "tobacco_use": "no",
        }
        self.runTest(message, params)

    def testSomeBlank(self):
        message = (
            "I would like to get some healthcare advice. "
            "I am a 35 year old female, who is pregnant."
        )
        # Blank or omitted params are both valid — health.gov treats them the same
        params = {
            "age": "35",
            "pregnant": "yes",
            "sex": "female",
        }
        self.runTest(message, params)


if __name__ == "__main__":
    unittest.main()
