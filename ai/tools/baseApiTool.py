from typing import Dict, Any, Optional
from agent_framework import tool
import requests


class BaseAPITool:
    """
    Shared HTTP GET base for MAF API tools.
    Subclasses override parse_response() to reshape the API response.
    """

    base_url: str = ""

    def build_params(self, **kwargs) -> Dict[str, Any]:
        return {k: v for k, v in kwargs.items() if k != "base_url" and v is not None and v != ""}

    def parse_response(self, response: requests.Response) -> Any:
        return response.json()

    @tool
    def __call__(self, base_url: Optional[str] = None, **kwargs) -> Any:
        """
        Call an HTTP GET API endpoint.

        Args:
            base_url: API endpoint URL. Uses class-level default if not provided.
        """
        url = base_url or self.base_url
        params = self.build_params(**kwargs)
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return self.parse_response(response)
        except requests.exceptions.RequestException as e:
            return {"Error": str(e)}
