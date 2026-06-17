from typing import Optional
from agent_framework import tool
from tools.baseApiTool import BaseAPITool
import requests


class AdviceAPITool(BaseAPITool):

    base_url: str = "https://odphp.health.gov/myhealthfinder/api/v4/myhealthfinder.json"

    def parse_response(self, response: requests.Response):
        data = response.json()
        try:
            return data["Result"]["Resources"]["All"]["Resource"][0]["Sections"]["section"][0]
        except Exception:
            return data

    @tool
    def __call__(
        self,
        age: Optional[str] = None,
        sex: Optional[str] = None,
        pregnant: Optional[str] = None,
        sexually_active: Optional[str] = None,
        tobacco_use: Optional[str] = None,
    ) -> dict:
        """
        Query the health.gov API for personalized healthcare advice.
        All parameters are optional — provide whichever are known.

        Args:
            age: Age of the person. Enter as a string (e.g. "35"), not an integer.
            sex: Sex of the person. Use "male" or "female".
            pregnant: Pregnancy status. Use "yes" or "no".
            sexually_active: Sexually active status. Use "yes" or "no".
            tobacco_use: Tobacco use status. Use "yes" or "no".
        """
        params = {k: v for k, v in {
            "age": age,
            "sex": sex,
            "pregnant": pregnant,
            "sexuallyActive": sexually_active,
            "tobaccoUse": tobacco_use,
        }.items() if v is not None and v != ""}

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self.parse_response(response)
        except requests.exceptions.RequestException as e:
            return {"Error": str(e)}


if __name__ == "__main__":
    api_tool = AdviceAPITool()
    result = api_tool(
        age="35",
        pregnant="yes",
        sex="female",
    )
    print(result)
