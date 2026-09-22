"""Free-text chat: /scan and /analysis answer fixed questions; this answers whatever was
actually typed, using the same real, already-computed numbers — never inventing a new one,
and declining anything shaped like advice, per narration/chat_prompt.py's system prompt.
"""
import json
import logging

from narration import chat_prompt, deepseek_client
from narration.facts import state_facts
from live.state import LiveState

from .scan import rejection_reason

log = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 500  # keeps a runaway paste from ballooning the prompt


def _context_json(results: dict) -> str:
    facts = {}
    for pair, result in results.items():
        if isinstance(result, LiveState):
            pair_facts = state_facts(result, include_plan=True, include_rejections=True)
            # Plain English, not the raw outcome code (e.g. "skipped_rollover"): the model
            # should never have to translate or explain the engine's internal vocabulary.
            for rejection, raw in zip(pair_facts["recent_rejections"], result.recent_rejections):
                rejection["reason"] = rejection_reason(raw)
            facts[pair] = pair_facts
        else:
            # A pair whose refresh failed is named, not silently dropped, so the model can
            # truthfully say it has no current data for that one rather than staying quiet.
            facts[pair] = {"error": "could not refresh this pair's data right now"}
    return json.dumps(facts, indent=2)


def answer(question: str, results: dict) -> str:
    question = question.strip()[:MAX_QUESTION_LENGTH]
    prompt = chat_prompt.build_user_prompt(question, _context_json(results))
    return deepseek_client.chat(chat_prompt.SYSTEM_PROMPT, prompt)
