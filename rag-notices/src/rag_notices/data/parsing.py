import json
import re
from typing import Dict, Any


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM response (with or without markdown tags)
    
    Args:
        response: Raw LLM response string
    
    Returns:
        Dict with parsed content and 'parsed': True if successful,
        or {'parsed': False} if parsing fails
    """
    try:
        # Step 1: Remove markdown code blocks (```json and ```)
        cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', response).strip()
        
        # Step 2: Parse the JSON
        parsed_data = json.loads(cleaned)
        
        # Step 3: Add 'parsed' flag
        parsed_data['parsed'] = True
        
        return parsed_data
        
    except json.JSONDecodeError:
        # Step 4: Fallback - try to find JSON between braces
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group(0))
                parsed_data['parsed'] = True
                return parsed_data
        except json.JSONDecodeError:
            pass
        
        # Step 5: If all parsing attempts fail
        return {'parsed': False}
