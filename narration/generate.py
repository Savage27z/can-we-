import json

from live.state import LiveState, compute_current_state

from . import deepseek_client
from .prompt import SYSTEM_PROMPT, build_user_prompt


def _facts(state: LiveState) -> dict:
    facts = state.to_dict()
    for setup in facts["active_setups"]:
        # The trade plan (entry, stop, target) is computed and rendered by live.plan and
        # shown above this report. Keeping it from the model means it cannot restate or
        # reword a number the trader will act on.
        setup.pop("plan", None)
    return facts


def narrate_state(state: LiveState) -> str:
    user_prompt = build_user_prompt(json.dumps(_facts(state), indent=2))
    return deepseek_client.chat(SYSTEM_PROMPT, user_prompt)


def narrate_pair(pair: str) -> str:
    return narrate_state(compute_current_state(pair))
