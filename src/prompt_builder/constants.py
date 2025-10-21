"""Shared constant maps used by the prompt builder."""

from __future__ import annotations

from typing import Dict, Set

TRUE_STRINGS: Set[str] = {"1", "true", "t", "yes", "y"}
FALSE_STRINGS: Set[str] = {"0", "false", "f", "no", "n"}

YT_FREQ_MAP = {
    "0": "rarely",
    "1": "occasionally",
    "2": "a few times a month",
    "3": "weekly",
    "4": "several times a week",
    "5": "daily",
    "6": "multiple times per day",
}

LANGUAGE_FRIENDLY_NAMES = {
    "en": "English",
    "en-us": "English",
    "en_us": "English",
    "english": "English",
    "es": "Spanish",
    "es-mx": "Spanish",
    "es_mx": "Spanish",
    "spanish": "Spanish",
    "fr": "French",
    "fr-fr": "French",
    "fr-ca": "French",
    "fr_ca": "French",
}

GUN_FIELD_LABELS: Dict[str, str] = {
    "right_to_own_importance": "Right-to-own importance",
    "assault_ban": "Supports assault weapons ban",
    "handgun_ban": "Supports handgun ban",
    "concealed_safe": "Believes concealed carry is safe",
    "stricter_laws": "Supports stricter gun laws",
    "gun_index": "Gun index",
    "gun_index_2": "Gun index (alt)",
    "gun_enthusiasm": "Gun enthusiasm",
    "gun_importance": "Gun importance",
    "gun_priority": "Gun policy priority",
    "gun_policy": "Gun policy stance",
    "gun_identity": "Gun identity",
}

MIN_WAGE_FIELD_LABELS: Dict[str, str] = {
    "minwage_text_r_w1": "Minimum wage stance (wave 1, inferred)",
    "minwage_text_r_w2": "Minimum wage stance (wave 2, inferred)",
    "minwage_text_r_w3": "Minimum wage stance (wave 3, inferred)",
    "minwage_text_w1": "Minimum wage stance (wave 1, survey)",
    "minwage_text_w2": "Minimum wage stance (wave 2, survey)",
    "mw_index_w1": "Minimum wage support index (wave 1)",
    "mw_index_w2": "Minimum wage support index (wave 2)",
    "minwage15_w1": "$15 minimum wage support (wave 1)",
    "minwage15_w2": "$15 minimum wage support (wave 2)",
    "mw_support_w1": "Supports wage increase (wave 1)",
    "mw_support_w2": "Supports wage increase (wave 2)",
    "minwage_importance": "Minimum wage importance",
    "minwage_priority": "Minimum wage priority",
}

__all__ = [
    "FALSE_STRINGS",
    "GUN_FIELD_LABELS",
    "LANGUAGE_FRIENDLY_NAMES",
    "MIN_WAGE_FIELD_LABELS",
    "TRUE_STRINGS",
    "YT_FREQ_MAP",
]
