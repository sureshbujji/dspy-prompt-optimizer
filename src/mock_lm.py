"""Deterministic mock LM implementing dspy's LM interface.

This module exists because dspy programs need an LM to run, and I wanted this
repo to work fully offline with no API key. The mock subclasses ``dspy.LM``
(the real one, from the installed ``dspy`` package) and overrides only
``forward()`` -- the single method dspy's ``BaseLM`` requires subclasses to
implement.

Mock LM interface contract (dspy 2.6.x, ``dspy/clients/base_lm.py``):

    forward(self, prompt=None, messages=None, **kwargs) -> response

where ``response`` must look like an OpenAI response object:

    response.choices            # list; each choice has .message.content (str)
                                #   and .finish_reason (str)
    response.usage              # mapping with token counts (dict(response.usage)
                                #   must work -- dspy calls dict() on it)
    response.model              # str, the model name

dspy's ``BaseLM.__call__`` then turns ``forward()``'s response into
``[choice.message.content, ...]``, and the ChatAdapter parses the
``[[ ## field ## ]]`` markers out of that string into the signature's output
fields. No network, no key, no litellm involved.

Honest-simulation design
------------------------
The mock behaves like a *weak but improvable* model, so that running a real
dspy teleprompter produces a real, measurable accuracy lift:

* Zero-shot (no demos in the prompt): a crude keyword heuristic -- e.g. any
  mention of "crash"/"data loss" -> ``critical``, any "error" -> ``high``,
  otherwise ``medium``; category by a similarly small keyword list. On the
  held-out set this scores ~40% (it is deliberately underfit).
* Few-shot (demos present): the mock pairs each assistant demo message with
  the user message before it, scores each demo by content-word overlap with
  the query (stopwords removed), and returns the winning demo's labels.
  Ties break toward the earliest demo; zero overlap falls back to the weak
  heuristic. With 16 in-context demos this scores ~85%.

This is a *simulation* of in-context learning, not the real thing: a real LM
generalizes from demos via its weights; this mock matches keywords
deterministically. It is still useful exactly because it lets the *dspy
machinery* -- signatures, adapters, teleprompters, metrics -- run end to end
offline, which is what this repo is actually demonstrating.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import dspy

# dspy's ChatAdapter delimits fields with [[ ## name ## ]] (see
# dspy/adapters/chat_adapter.py: field_header_pattern).
FIELD_RE = re.compile(r"\[\[ ## (\w+) ## \]\]")

_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been being has have had "
    "hasnt dont isnt not no yes it its this that these those as at by from their our "
    "your you we they he she his her him them will would can could should may might "
    "do does did done app apps get gets got after when where which who whom whose "
    "than then there here also just only into out over under again once than so such "
    "all any both each few more most other some than too very via per within without "
    "during peak hours latest update".split()
)


def _parse_fields(content: str) -> Dict[str, str]:
    """Split a dspy field-block message into {field_name: value}."""
    matches = list(FIELD_RE.finditer(content))
    fields: Dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        fields[match.group(1)] = content[match.end() : end].strip()
    return fields


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOPWORDS}


class _MockMessage:
    def __init__(self, content: str, role: str = "assistant") -> None:
        self.content = content
        self.role = role


class _MockChoice:
    def __init__(self, content: str) -> None:
        self.message = _MockMessage(content)
        self.finish_reason = "stop"
        self.index = 0


class _MockResponse:
    """Minimal OpenAI-shaped response: .choices / .usage / .model."""

    def __init__(self, content: str, model: str) -> None:
        self.choices = [_MockChoice(content)]
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.model = model


class WeakMockLM(dspy.LM):
    """Deterministic stand-in for a weak chat LM.

    Conforms to dspy 2.6.x's LM protocol by overriding ``forward()`` (see
    module docstring). Fully deterministic: identical inputs always produce
    identical outputs, with no randomness and no network calls.
    """

    def __init__(self, **kwargs: Any) -> None:
        # cache=False: our forward() never touches litellm, so dspy's
        # request cache would only add overhead.
        super().__init__(model="mock/weak-bug-lm", cache=False, **kwargs)

    # -- dspy LM protocol -------------------------------------------------
    def forward(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> _MockResponse:
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]

        demos, pending, query = [], None, {}
        # ChatAdapter sends: system, (user, assistant)* demos, user query.
        for msg in messages[1:]:
            fields = _parse_fields(msg.get("content", ""))
            if msg.get("role") == "user":
                pending = fields
            elif (
                msg.get("role") == "assistant"
                and "severity" in fields
                and pending is not None
            ):
                demo = dict(pending)
                demo.update(fields)
                demos.append(demo)
                pending = None
        if pending is not None:
            query = pending

        if demos:
            severity, category = self._predict_with_demos(demos, query)
        else:
            severity, category = self._weak_heuristic(query)

        completion = (
            "[[ ## severity ## ]]\n"
            f"{severity}\n\n"
            "[[ ## category ## ]]\n"
            f"{category}\n\n"
            "[[ ## completed ## ]]"
        )
        return _MockResponse(completion, self.model)

    # -- simulated model behavior ------------------------------------------
    @staticmethod
    def _weak_heuristic(query: Dict[str, str]) -> tuple:
        """Deliberately underfit keyword rules: the 'weak model' baseline."""
        text = (query.get("bug_title", "") + " " + query.get("bug_description", "")).lower()
        if "crash" in text or "data loss" in text:
            severity = "critical"
        elif "error" in text:
            severity = "high"
        else:
            severity = "medium"
        if "slow" in text or "freeze" in text or "timeout" in text or "memory" in text:
            category = "performance"
        elif (
            "button" in text
            or "layout" in text
            or "color" in text
            or "font" in text
            or "align" in text
            or "overlap" in text
        ):
            category = "ui"
        elif "typo" in text or "docs" in text or "help" in text:
            category = "documentation"
        elif "auth" in text or "token" in text or "password" in text:
            category = "security"
        else:
            category = "functional"
        return severity, category

    @staticmethod
    def _predict_with_demos(
        demos: List[Dict[str, str]], query: Dict[str, str]
    ) -> tuple:
        """Simulated in-context learning: copy labels from the most similar demo."""
        query_words = _content_words(
            query.get("bug_title", "") + " " + query.get("bug_description", "")
        )

        def overlap(demo: Dict[str, str]) -> int:
            demo_words = _content_words(
                demo.get("bug_title", "") + " " + demo.get("bug_description", "")
            )
            return len(query_words & demo_words)

        scored = sorted(
            ((overlap(d), i) for i, d in enumerate(demos)),
            key=lambda t: (-t[0], t[1]),
        )
        best_score, best_idx = scored[0]
        if best_score == 0:
            return WeakMockLM._weak_heuristic(query)
        best = demos[best_idx]
        return best["severity"], best["category"]
