import json

from live.plan import when
from live.state import LiveState, compute_current_state

from . import deepseek_client
from .prompt import SYSTEM_PROMPT, build_user_prompt

# Times the model may quote. They are handed over already formatted ("Wed 10 Jun 21:00 UTC"),
# as the news layer does with its "when" text, because given an ISO string the model copies it
# into the report as it is. fvg_start_time stays raw: it is diagnostic and never shown.
_SETUP_TIMES = ("sweep_time", "confirm_time")


def _facts(state: LiveState) -> dict:
    facts = state.to_dict()
    facts["as_of"] = when(facts["as_of"])
    for setup in facts["active_setups"]:
        # The trade plan (entry, stop, target) is computed and rendered by live.plan and
        # shown above this report. Keeping it from the model means it cannot restate or
        # reword a number the trader will act on.
        setup.pop("plan", None)
        for name in _SETUP_TIMES:
            if setup.get(name):
                setup[name] = when(setup[name])
    return facts


def narrate_state(state: LiveState) -> str:
    user_prompt = build_user_prompt(json.dumps(_facts(state), indent=2))
    return deepseek_client.chat(SYSTEM_PROMPT, user_prompt)


def narrate_pair(pair: str) -> str:
    return narrate_state(compute_current_state(pair))
