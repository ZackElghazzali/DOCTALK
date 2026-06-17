from typing import Optional
from agent_framework import tool
from tools.baseApiTool import BaseAPITool
import requests


class DiagnosisAPITool(BaseAPITool):

    base_url: str = "https://clinicaltables.nlm.nih.gov/api/conditions/v3/search?"

    @tool
    def __call__(
        self,
        terms: str,
        base_url: Optional[str] = None,
    ) -> dict:
        """
        Search the NLM clinical tables API for medical conditions matching a search term.

        Args:
            terms: The search string for which to find matching conditions (e.g. "Cough").
                   More than one partial word can be present — there is an implicit AND between them.
            base_url: API endpoint URL. Uses class-level default if not provided.
        """
        url = base_url or self.base_url
        params = {"terms": terms}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return self.parse_response(response)
        except requests.exceptions.RequestException as e:
            return {"Error": str(e)}


if __name__ == "__main__":
    api_tool = DiagnosisAPITool()
    result = api_tool(
        base_url="https://clinicaltables.nlm.nih.gov/api/conditions/v3/search?",
        terms="Cough",
    )
    print(result)
