"""JSON extraction from LLM responses (mirror of rag_notices.data.parsing)."""
import json
import re
from typing import Any, Dict


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from an LLM response (with or without markdown fences).

    Returns the parsed dict with an added 'parsed': True flag, or {'parsed': False}
    if parsing fails.
    """
    try:
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", response).strip()
        parsed_data = json.loads(cleaned)
        parsed_data["parsed"] = True
        return parsed_data
    except json.JSONDecodeError:
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group(0))
                parsed_data["parsed"] = True
                return parsed_data
        except json.JSONDecodeError:
            pass
        return {"parsed": False}
