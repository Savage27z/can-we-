import json

from live.state import LiveState, compute_current_state

from . import deepseek_client
from .prompt import SYSTEM_PROMPT, build_user_prompt


def narrate_state(state: LiveState) -> str:
    user_prompt = build_user_prompt(json.dumps(state.to_dict(), indent=2))
    return deepseek_client.chat(SYSTEM_PROMPT, user_prompt)


def narrate_pair(pair: str) -> str:
    return narrate_state(compute_current_state(pair))
