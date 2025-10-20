"""Text-formatting utilities used by the prompt builder."""

from __future__ import annotations

import math
from typing import Any, List, Optional

from .constants import LANGUAGE_FRIENDLY_NAMES
from .parsers import format_age, is_nanlike


def truncate_text(text: str, limit: int = 160) -> str:
    """
    Trim ``text`` to ``limit`` characters, adding an ellipsis when truncation occurs.

    :param text: Raw text to condense.
    :type text: str
    :param limit: Maximum number of characters to allow before shortening.
    :type limit: int
    :returns: Condensed text, suffixed with ``...`` when the original exceeded ``limit``.
    :rtype: str
    """

    text = text.strip()
    if limit is None:
        return text
    if limit <= 3 or len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def clean_text(value: Any, *, limit: Optional[int] = None) -> str:
    """
    Convert ``value`` to normalised text, collapsing list-like inputs when present.

    :param value: Arbitrary value to format as human-friendly text.
    :type value: Any
    :param limit: Optional character limit applied after normalisation.
    :type limit: int, optional
    :returns: Cleaned string representation or an empty string when nothing meaningful remains.
    :rtype: str
    """

    if is_nanlike(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [clean_text(v) for v in value]
        parts = [p for p in parts if p]
        if not parts:
            return ""
        text = "; ".join(parts)
        return truncate_text(text, limit or len(text))
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            value = int(value)
    text = str(value).strip()
    if not text:
        return ""
    if limit:
        text = truncate_text(text, limit)
    return text


def human_join(parts: List[str]) -> str:
    """
    Join ``parts`` into a grammatically friendly string.

    :param parts: Text fragments to combine.
    :type parts: List[str]
    :returns: Items joined with commas and ``and`` following English style.
    :rtype: str
    """

    cleaned = [p.strip() for p in parts if p and p.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def with_indefinite_article(phrase: str) -> str:
    """
    Prefix ``phrase`` with an appropriate indefinite article.

    :param phrase: The phrase to augment.
    :type phrase: str
    :returns: The phrase with ``a`` or ``an`` prepended when needed.
    :rtype: str
    """

    text = phrase.strip()
    if not text:
        return text
    lowered = text.lower()
    prefixes = ("a ", "an ", "the ", "another ", "this ", "that ", "someone", "somebody", "none")
    if lowered.startswith(prefixes):
        return text
    if lowered[0] in "aeiou":
        return f"an {text}"
    return f"a {text}"


def describe_age_fragment(value: Any) -> Optional[str]:
    """
    Produce a natural-language age fragment derived from ``value``.

    :param value: Raw age input as text or numeric data.
    :type value: Any
    :returns: Age description suitable for insertion into a sentence, or ``None`` when unknown.
    :rtype: Optional[str]
    """

    age_text = format_age(value)
    if not age_text:
        return None
    cleaned = age_text.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    result: Optional[str] = None
    if cleaned.isdigit():
        result = f"{cleaned}-year-old"
    elif cleaned.endswith("+"):
        base = cleaned[:-1].strip()
        if base.isdigit():
            result = f"{cleaned} years old"
    elif "-" in cleaned:
        range_parts = [part.strip() for part in cleaned.split("-") if part.strip()]
        if range_parts and all(part.isdigit() for part in range_parts):
            if len(range_parts) == 2:
                start, end = range_parts
                result = f"between {start} and {end} years old"
    elif lowered.endswith("year-old") or lowered.endswith("years old"):
        result = cleaned
    if result is None:
        result = f"{cleaned} years old"
    return result


def describe_gender_fragment(value: Any) -> Optional[str]:
    """
    Produce a normalised gender fragment derived from ``value``.

    :param value: Raw gender input as text or coded data.
    :type value: Any
    :returns: Gender description ready for sentence use, or ``None`` when unknown.
    :rtype: Optional[str]
    """

    text = clean_text(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"male", "man", "m"}:
        return "man"
    if lowered in {"female", "woman", "f"}:
        return "woman"
    if lowered in {"non-binary", "nonbinary", "non binary"}:
        return "non-binary person"
    if "prefer" in lowered and "say" in lowered:
        return "someone who prefers not to state their gender"
    return text


def normalize_language_text(value: Any) -> str:
    """
    Convert language codes or informal labels into human-friendly names.

    :param value: Raw language identifier.
    :type value: Any
    :returns: Normalised language name or an empty string when nothing is provided.
    :rtype: str
    """

    text = clean_text(value)
    if not text:
        return ""
    lowered = text.lower()
    return LANGUAGE_FRIENDLY_NAMES.get(lowered, text)


# Compatibility aliases for legacy imports.
_truncate_text = truncate_text
_human_join = human_join
_with_indefinite_article = with_indefinite_article
_describe_age_fragment = describe_age_fragment
_describe_gender_fragment = describe_gender_fragment
_normalize_language_text = normalize_language_text

__all__ = [
    "clean_text",
    "describe_age_fragment",
    "describe_gender_fragment",
    "human_join",
    "normalize_language_text",
    "truncate_text",
    "with_indefinite_article",
]
