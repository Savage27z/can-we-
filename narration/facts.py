"""Turns a LiveState into the JSON handed to a DeepSeek model. Shared by the /analysis
narration report and the chat Q&A layer, so a fact reaching either model is formatted
identically — times pre-rendered, so the model never does time arithmetic and can only
copy what it's given, never compute or restate it in a different form.
"""
from live.plan import when
from live.state import LiveState

_SETUP_TIMES = ("sweep_time", "confirm_time")


def state_facts(state: LiveState, include_plan: bool = False,
                include_rejections: bool = False) -> dict:
    """`include_plan`: the narrated report keeps entry/stop/target/rr from the model because a
    separate deterministic block already renders them above it — the model restating them
    would risk a reworded number the trader might act on differently. Chat has no such block,
    so it is allowed to quote them (still only from this JSON, never invented).
    `include_rejections`: whether recent no-trade outcomes are included; the narrated report's
    fixed format has no section for them, so they are left out by default."""
    facts = state.to_dict()
    facts["as_of"] = when(facts["as_of"])
    for setup in facts["active_setups"]:
        if not include_plan:
            setup.pop("plan", None)
        for name in _SETUP_TIMES:
            if setup.get(name):
                setup[name] = when(setup[name])
    if include_rejections:
        for rejection in facts.get("recent_rejections", []):
            rejection["sweep_time"] = when(rejection["sweep_time"])
            if rejection.get("confirm_time"):
                rejection["confirm_time"] = when(rejection["confirm_time"])
    else:
        facts.pop("recent_rejections", None)
    return facts
