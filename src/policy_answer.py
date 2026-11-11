"""A narrow numerical guard for the compatibility generator, not full grounding."""

import re
import unicodedata

from src.verification import numbers


def _digits(text):
    return "".join(str(unicodedata.digit(char)) if char.isdecimal() else char for char in text)


def supported_numbers(answer, evidence):
    # List ordinals are formatting, not policy quantities. Require whitespace so
    # decimal amounts such as '5.7' cannot be mistaken for a numbered list.
    content = re.sub(r"(?m)^\s*(?:[-*]\s*)?\d+[.)]\s+", "", answer)
    return numbers(_digits(content)).issubset(numbers(_digits("\n".join(evidence))))


def generate_policy_answer(chain, payload, evidence):
    for attempt in range(2):
        arguments = dict(payload)
        if attempt:
            arguments["question"] += (
                "\n\nRe-answer the original question using only the supplied policy evidence. "
                "The preceding response included numerical claims absent from that evidence. "
                "Do not repeat or invent them. If the answer is absent, say verified policy information is unavailable."
            )
        answer = chain.invoke(arguments).content
        if isinstance(answer, str) and answer.strip() and supported_numbers(answer, evidence):
            return answer
    return None
