import json
import os
from typing import Dict, Any, Optional


def load_config(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    
    with open(config_path, 'r') as f:
        return json.load(f)


def ensure_api_keys(config: Dict[str, Any]) -> bool:
    missing = []
    if not config.get('openai', {}).get('api_key'):
        missing.append('OpenAI API key')
    if not config.get('tinyfish', {}).get('api_key'):
        missing.append('Tinyfish API key')
    
    if missing:
        print(f"Warning: Missing API keys: {', '.join(missing)}")
        return False
    return True
