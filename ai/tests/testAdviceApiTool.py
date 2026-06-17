import unittest
from tools.adviceApiTool import AdviceAPITool

# To run all tests from ai dir: python -m unittest discover
# To run just these tests from ai dir: python -m unittest tests/testAdviceApiTool.py

# All tests here have the same output, since the latter two tests are the same
# beside the blank/leftout parameters (which are implied by the API defaulting them).
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = AdviceAPITool()

    def testApi(self):
        result = self.tool(
            age="35",
            sex="female",
            pregnant="no",
            sexually_active="yes",
            tobacco_use="no",
        )
        self.assertEqual(result["Title"], "The Basics: Overview")

    def testApiBlank(self):
        # Empty strings behave the same as omitted — health.gov ignores blank params
        result = self.tool(
            age="35",
            sex="female",
            pregnant="no",
            sexually_active="",
            tobacco_use="",
        )
        self.assertEqual(result["Title"], "The Basics: Overview")

    def testApiLeftout(self):
        # Omitting optional params entirely — same result as blank
        result = self.tool(
            age="35",
            sex="female",
            pregnant="no",
        )
        self.assertEqual(result["Title"], "The Basics: Overview")


if __name__ == "__main__":
    unittest.main()
