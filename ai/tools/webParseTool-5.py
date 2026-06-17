# web parse tool that currently looks up drug prices
# from costplusdrugs
# fetches complete drug DB, caches, and matches

from typing import Dict, Any, Optional, List, ClassVar
from agent_framework import tool
import requests
import copy
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from bs4 import BeautifulSoup, Comment


def calculate_similarity(str1: str, str2: str) -> float:
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


class DrugPriceLookupTool:
    """
    Searches Cost Plus Drugs for medication pricing.
    AG2 equivalent: DrugPriceLookupTool(BaseTool) with DrugPriceLookupArgs(BaseModel).
    DrugPriceLookupArgs is removed — parameters move directly onto __call__.
    _run/_arun collapse into a single @tool __call__.
    All caching, fuzzy matching, and formatting logic is unchanged.
    """

    API_URL: ClassVar[str] = "https://us-central1-costplusdrugs-publicapi.cloudfunctions.net/main"
    CACHE_DURATION_HOURS: ClassVar[int] = 24
    _cached_drugs: ClassVar[Optional[List[Dict[str, Any]]]] = None
    _cache_timestamp: ClassVar[Optional[datetime]] = None

    def fetch_drug_database(self) -> List[Dict[str, Any]]:
        if self._cached_drugs is not None and self._cache_timestamp is not None:
            cache_age = datetime.now() - self._cache_timestamp
            if cache_age < timedelta(hours=self.CACHE_DURATION_HOURS):
                print(f"using cached drug database ({len(self._cached_drugs)} drugs)")
                return self._cached_drugs

        print(f"fetching drug DB from {self.API_URL}...")
        try:
            response = requests.get(self.API_URL, timeout=30)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and "results" in data:
                drugs = data["results"]
            elif isinstance(data, list):
                drugs = data
            else:
                raise ValueError("Unexpected API response format")

            DrugPriceLookupTool._cached_drugs = drugs
            DrugPriceLookupTool._cache_timestamp = datetime.now()
            print(f"successfully fetched {len(drugs)} drugs")
            return drugs

        except requests.RequestException as e:
            print(f"error fetching drug db: {e}")
            if self._cached_drugs is not None:
                print(f"using possible old cached data as fallback.. ({len(self._cached_drugs)} drugs)")
                return self._cached_drugs
            raise

    def search_drugs(self, query: str, drugs: List[Dict[str, Any]], threshold: float = 0.6) -> List[Dict[str, Any]]:
        query_lower = query.lower().strip()
        matches = []

        for drug in drugs:
            med_name = drug.get("medication_name", "").lower()
            brand_name = drug.get("brand_name", "").lower()

            med_similarity = calculate_similarity(query_lower, med_name)
            brand_similarity = calculate_similarity(query_lower, brand_name) if brand_name else 0.0
            similarity = max(med_similarity, brand_similarity)

            if query_lower in med_name or query_lower in brand_name:
                similarity = max(similarity, 0.9)
            if similarity >= threshold:
                matches.append({"drug": drug, "similarity": similarity})

        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return [match["drug"] for match in matches]

    def format_drug_info(self, drug: Dict[str, Any]) -> Dict[str, Any]:
        unit_price = drug.get("unit_billing_price", "$0.00").replace("$", "")
        quantity = int(drug.get("medisapn_quantity", 1))

        try:
            price_per_unit = float(unit_price)
            total_price = price_per_unit * quantity
        except (ValueError, TypeError):
            price_per_unit = 0.0
            total_price = 0.0

        return {
            "medication_name": drug.get("medication_name", "Unknown"),
            "brand_name": drug.get("brand_name", "N/A"),
            "generic": drug.get("brand_generic", "Unknown") == "Generic",
            "form": drug.get("form", "Unknown"),
            "strength": drug.get("strength", "Unknown"),
            "quantity": quantity,
            "pack_size": drug.get("medisapn_pack_size", "Unknown"),
            "pack_units": drug.get("medisapn_pack_size_units", "ea"),
            "price_per_unit": f"${price_per_unit}",
            "total_price": f"${total_price}",
            "ndc": drug.get("ndc", "Unknown"),
            "url": drug.get("url", ""),
            "insurance_eligible": drug.get("insurance_eligible", "Unknown"),
            "auto_refill": drug.get("auto_refill", False),
        }

    @tool
    def __call__(self, drug_name: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search for medication prices from Cost Plus Drugs.
        Provide the drug name (e.g., "metformin", "lisinopril 10mg") and get pricing
        information including strength, form, quantity, and cost.
        Returns multiple results if there are different formulations.

        Args:
            drug_name: Name of the medication to search for (e.g. "metformin" or "lisinopril 10mg").
            max_results: Maximum number of results to return (default: 5).
        """
        print(f"searching for: {drug_name}...")
        try:
            drugs = self.fetch_drug_database()
            matches = self.search_drugs(drug_name, drugs, threshold=0.5)

            if not matches:
                return {
                    "found": False,
                    "query": drug_name,
                    "message": f"No medications found matching '{drug_name}'. Try a different spelling/generic name",
                    "results": [],
                }

            matches = matches[:max_results]
            formatted_results = [self.format_drug_info(drug) for drug in matches]
            print(f"found {len(formatted_results)} matches")

            return {
                "found": True,
                "query": drug_name,
                "count": len(formatted_results),
                "results": formatted_results,
            }

        except Exception as e:
            print(f"drug lookup error: {e}")
            return {
                "found": False,
                "query": drug_name,
                "error": str(e),
                "message": "Failed to fetch drug prices. API might be unavailable",
                "results": [],
            }


class HTMLParserTool:
    """
    Parses and extracts content from any HTML webpage.
    AG2 equivalent: HTMLParserTool(BaseTool) with HTMLParserArgs(BaseModel).
    HTMLParserArgs removed — parameters move directly onto __call__.
    _run/_arun collapse into a single @tool __call__.
    All parsing, metadata extraction, and hidden content logic is unchanged.
    """

    @tool
    def __call__(
        self,
        url: str,
        extract_text: bool = True,
        extract_links: bool = False,
        extract_images: bool = False,
        extract_scripts: bool = False,
        extract_hidden: bool = False,
    ) -> Dict[str, Any]:
        """
        Parse and extract content from any HTML webpage.
        Retrieves text content, article titles, headings, and paragraphs.
        Useful for reading anything HTML on the web.

        Args:
            url: URL of the webpage to parse (e.g. "http://example.com/blog").
            extract_text: Extract all visible text content from the page (default: True).
            extract_links: Extract all hyperlinks from the page (default: False).
            extract_images: Extract all image sources from the page (default: False).
            extract_scripts: Extract all script tags and their content (ATTACK VECTOR) (default: False).
            extract_hidden: Extract hidden, invisible content like comments (ATTACK VECTOR) (default: False).
        """
        print(f"[parser] fetching URL: {url}")

        try:
            response = requests.get(url, timeout=30, headers={
                "User-Agent": "Mozilla/5.0 (HealthcareBot/1.0; +http://example.com/bot)"
            })
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            result = {
                "success": True,
                "url": url,
                "timestamp": datetime.now().isoformat(),
            }

            result["metadata"] = self._extract_metadata(soup)
            if extract_text:
                result["text_content"] = self._extract_text(soup)
            if extract_links:
                result["links"] = self._extract_links(soup)
            if extract_images:
                result["images"] = self._extract_images(soup)
            if extract_scripts:
                scripts = self._extract_scripts(soup)
                result["scripts"] = scripts
                result["script_count"] = len(scripts)
            if extract_hidden:
                hidden = self._extract_hidden_content(soup)
                result["hidden_content"] = hidden

            print(f"[parser] successfully parsed. extracted "
                  f"{len(result.get('text_content', {}).get('paragraphs', []))} paragraphs, "
                  f"{result.get('script_count', 0)} scripts")
            return result

        except requests.RequestException as e:
            print(f"[parser] request error: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "error_type": "request_error",
                "message": f"failed to fetch page from {url}; server may be unreachable",
            }
        except Exception as e:
            print(f"[parser] parsing error: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "error_type": "parse_error",
                "message": "failed to parse HTML. page structure may not be valid",
            }

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        metadata = {}
        title = soup.find("title")
        if title:
            metadata["title"] = title.get_text(strip=True)
        description = soup.find("meta", attrs={"name": "description"})
        if description and description.get("content"):
            metadata["description"] = description.get("content")
        keywords = soup.find("meta", attrs={"name": "keywords"})
        if keywords and keywords.get("content"):
            metadata["keywords"] = keywords.get("content")
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            metadata["og_title"] = og_title.get("content")
        author = soup.find("meta", attrs={"name": "author"})
        if author and author.get("content"):
            metadata["author"] = author.get("content")
        return metadata

    def _extract_text(self, soup: BeautifulSoup) -> Dict[str, Any]:
        text_content = {}
        soup_text = copy.copy(soup)
        for element in soup_text(["script", "style", "meta", "link"]):
            element.decompose()
        for element in soup_text.select("[hidden]"):
            element.decompose()
        for element in soup_text.find_all(style=True):
            style = element["style"].replace(" ", "").lower()
            if any(k in style for k in ["display:none", "visibility:hidden", "opacity:0"]):
                element.decompose()
        for element in soup_text.find_all(class_=True):
            classes = " ".join(element["class"]).lower()
            if any(k in classes for k in ["hidden", "invisible", "d-none", "hide"]):
                element.decompose()
        for element in soup_text.find_all(style=True):
            style = element["style"].replace(" ", "").lower()
            if "position:absolute" in style and any(k in style for k in ["left:-", "top:-", "right:-", "bottom:-"]):
                element.decompose()
        text_content["full_text"] = soup_text.get_text(separator=" ", strip=True)
        return text_content

    def _extract_links(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        links = []
        for a_tag in soup.find_all("a", href=True):
            link_data = {"href": a_tag["href"], "text": a_tag.get_text(strip=True)}
            if a_tag.get("title"):
                link_data["title"] = a_tag["title"]
            links.append(link_data)
        return links

    def _extract_images(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        images = []
        for img_tag in soup.find_all("img"):
            img_data = {}
            if img_tag.get("src"):
                img_data["src"] = img_tag["src"]
            if img_tag.get("alt"):
                img_data["alt"] = img_tag["alt"]
            if img_tag.get("title"):
                img_data["title"] = img_tag["title"]
            if img_data:
                images.append(img_data)
        return images

    def _extract_scripts(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        scripts = []
        for idx, script_tag in enumerate(soup.find_all("script")):
            script_data = {"index": idx, "type": script_tag.get("type", "text/javascript")}
            if script_tag.get("src"):
                script_data["source"] = "external"
                script_data["src"] = script_tag.get("src")
                script_data["content"] = None
            else:
                script_data["source"] = "inline"
                content = script_tag.string
                script_data["content"] = content.strip() if content else script_tag.get_text(strip=True)
            other_attrs = {k: v for k, v in script_tag.attrs.items() if k not in ["src", "type"]}
            if other_attrs:
                script_data["attributes"] = other_attrs
            scripts.append(script_data)
        return scripts

    def _extract_hidden_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        hidden = {}

        comment_list = []
        for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith("<!--")):
            comment_text = comment.strip().lstrip("<!--").rstrip("-->").strip()
            if comment_text:
                comment_list.append(comment_text)
        hidden["comments"] = comment_list
        hidden["comment_count"] = len(comment_list)

        hidden_elements = []
        for element in soup.find_all(style=True):
            style = element["style"].lower()
            if any(k in style for k in ["display:none", "display: none", "visibility:hidden",
                                         "visibility: hidden", "opacity:0", "opacity: 0"]):
                text = element.get_text(strip=True)
                if text:
                    hidden_elements.append({"tag": element.name, "text": text, "style": element["style"]})
        for element in soup.find_all(class_=True):
            classes = " ".join(element["class"]).lower()
            if any(k in classes for k in ["hidden", "invisible", "d-none", "hide"]):
                text = element.get_text(strip=True)
                if text:
                    hidden_elements.append({"tag": element.name, "text": text, "class": " ".join(element["class"])})
        for element in soup.select("[hidden]"):
            print(element)
            text = element.get_text(strip=True)
            if text:
                hidden_elements.append({"tag": element.name, "text": text, "attr": element["hidden"]})
        if hidden_elements:
            hidden["hidden_elements"] = hidden_elements
            hidden["hidden_element_count"] = len(hidden_elements)

        data_attrs = []
        for element in soup.find_all(lambda tag: any(attr.startswith("data-") for attr in tag.attrs)):
            data_dict = {attr: element[attr] for attr in element.attrs if attr.startswith("data-")}
            if data_dict:
                data_attrs.append({
                    "tag": element.name,
                    "attributes": data_dict,
                    "text_preview": element.get_text(strip=True)[:100],
                })
        if data_attrs:
            hidden["data_attributes"] = data_attrs
            hidden["data_attribute_count"] = len(data_attrs)

        custom_meta = []
        for meta in soup.find_all("meta"):
            if not meta.get("charset") and meta.get("name") not in ["description", "keywords", "author", "viewport"]:
                meta_data = {}
                if meta.get("name"):
                    meta_data["name"] = meta["name"]
                if meta.get("content"):
                    meta_data["content"] = meta["content"]
                if meta.get("property"):
                    meta_data["property"] = meta["property"]
                if meta_data:
                    custom_meta.append(meta_data)
        if custom_meta:
            hidden["custom_meta_tags"] = custom_meta

        offscreen_elements = []
        for element in soup.find_all(style=True):
            style = element["style"].lower()
            if any(k in style for k in ["left:-", "top:-", "position:absolute"]):
                text = element.get_text(strip=True)
                if text:
                    offscreen_elements.append({"tag": element.name, "text": text[:200], "style": element["style"]})
        if offscreen_elements:
            hidden["offscreen_elements"] = offscreen_elements

        return hidden if hidden else None
