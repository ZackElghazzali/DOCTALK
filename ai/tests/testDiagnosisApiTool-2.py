import unittest
from tools.diagnosisApiTool import DiagnosisAPITool

# To run all tests from ai dir: python -m unittest discover
# To run just this test from ai dir: python -m unittest tests/testDiagnosisApiTool.py

class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = DiagnosisAPITool()

    def testApi(self):
        actual = [
            4,
            ["2236", "8544", "9091", "2239"],
            None,
            [
                ["Cough"],
                ["Sputum production"],
                ["Whooping cough (pertussis)"],
                ["Coughing up blood"],
            ],
        ]
        result = self.tool(
            base_url="https://clinicaltables.nlm.nih.gov/api/conditions/v3/search?",
            terms="Cough",
        )
        self.assertEqual(result, actual)


if __name__ == "__main__":
    unittest.main()
