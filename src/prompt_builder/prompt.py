"""Top-level prompt construction entry points."""

# pylint: disable=too-many-lines

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence

from .formatters import clean_text, human_join
from .parsers import as_list_json, format_count, format_yes_no, is_nanlike
from .profiles import ProfileRender, render_profile, synthesize_viewer_sentence
from .shared import first_non_nan_value, load_selected_survey_row
from .value_maps import format_field_value
from .video_stats import lookup_video_stats


def _last_index(xs: Any, val: Any) -> Optional[int]:
    """Return the last index of ``val`` within ``xs`` when ``xs`` is a list.

    :param xs: Sequence potentially containing ``val``.
    :type xs: Any
    :param val: Value to locate.
    :type val: Any
    :returns: Zero-based index or ``None`` when not found.
    :rtype: Optional[int]
    """
    if not isinstance(xs, list) or val is None:
        return None
    idx = None
    for i, v in enumerate(xs):
        if v == val:
            idx = i
    return idx


def _selected_row(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the ``selected_survey_row`` mapping from ``ex`` when available.

    :param ex: Dataset example potentially containing inline or serialised survey metadata.
    :type ex: Dict[str, Any]
    :returns: Normalised dictionary representation of the selected survey row.
    :rtype: Dict[str, Any]
    """
    return load_selected_survey_row(ex)


def build_user_prompt(ex: Dict[str, Any], max_hist: int = 12) -> str:
    """
    Assemble the user prompt used for recommendation tasks.

    :param ex: Cleaned interaction row with profile, history, and slate fields.
    :type ex: Dict[str, Any]
    :param max_hist: Maximum number of prior videos to include in the history section.
    :type max_hist: int
    :returns: Natural-language prompt describing the viewer, viewing context, and options.
    :rtype: str
    """

    show_ids = os.getenv("GRAIL_SHOW_IDS", "0") == "1"
    profile = render_profile(ex)
    selected = _selected_row(ex)

    lines: List[str] = []

    viewer_line = _viewer_summary_line(ex, selected, profile)
    if viewer_line:
        lines.append(f"VIEWER {viewer_line}")
    else:
        lines.append("VIEWER (profile information unavailable)")

    viewpoint_line = _initial_viewpoint_line(ex, selected)
    if viewpoint_line:
        lines.append(f"Initial Viewpoint: {viewpoint_line}")

    lines.append("")

    current_line = _current_video_line(ex, show_ids)
    if current_line:
        lines.append(f"CURRENTLY WATCHING {current_line}")
    else:
        lines.append("CURRENTLY WATCHING (current video unavailable)")

    lines.append("")
    lines.append("RECENTLY WATCHED (NEWEST LAST)")
    history_lines = _history_lines(ex, show_ids, max_hist)
    if history_lines:
        lines.extend(history_lines)
    else:
        lines.append("(no recently watched videos available)")

    survey_lines = _survey_lines(ex)
    if survey_lines:
        lines.append("")
        lines.append("SURVEY HIGHLIGHTS")
        lines.extend(survey_lines)

    option_lines = _options_lines(ex, show_ids)
    lines.append("")
    lines.append("OPTIONS")
    if option_lines:
        lines.extend(option_lines)
    else:
        lines.append("(no recommendation options available)")

    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _viewer_summary_line(
    ex: Dict[str, Any],
    selected: Dict[str, Any],
    profile: ProfileRender,
) -> str:
    """
    Build a compact viewer descriptor suitable for the ``VIEWER`` heading.

    :param ex: Primary dataset row with viewer and survey metadata.
    :type ex: Dict[str, Any]
    :param selected: Normalised ``selected_survey_row`` mapping extracted from ``ex``.
    :type selected: Dict[str, Any]
    :param profile: Full profile rendering returned by
        :func:`prompt_builder.profiles.render_profile`.
    :type profile: ProfileRender
    :returns: Single sentence summarising identity and location, or an empty string.
    :rtype: str
    """

    merged: Dict[str, Any] = dict(selected)
    merged.update(ex)
    raw = synthesize_viewer_sentence(merged).strip()
    fragments: List[str]
    if raw and raw.lower() not in {"(no profile provided)", "no profile provided"}:
        fragments = _normalize_viewer_fragments(raw)
    else:
        fragments = _fragments_from_profile(profile)

    location = _location_fragment(profile)
    if location:
        fragments.append(location)

    seen: set[str] = set()
    cleaned_fragments: List[str] = []
    for fragment in fragments:
        cleaned = fragment.strip().rstrip(".")
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned_fragments.append(cleaned)

    sentence = "; ".join(cleaned_fragments)
    return f"{sentence}." if sentence else ""


def _fragments_from_profile(profile: ProfileRender) -> List[str]:
    """
    Derive fallback fragments from an existing :class:`ProfileRender`.

    :param profile: Rendered profile containing ordered sentences.
    :type profile: ProfileRender
    :returns: List containing the first sentence stripped of punctuation, or empty
        when unavailable.
    :rtype: List[str]
    """

    for sentence in profile.sentences:
        cleaned = sentence.strip()
        if cleaned:
            return [cleaned.rstrip(".")]
    return []


def _normalize_viewer_fragments(raw: str) -> List[str]:
    """
    Split a synthesised viewer sentence into semicolon-ready fragments.

    :param raw: Raw viewer sentence produced by :func:`synthesize_viewer_sentence`.
    :type raw: str
    :returns: Ordered list of descriptive fragments prioritising identity elements.
    :rtype: List[str]
    """

    text = raw.strip().rstrip(".").replace(",\n", ", ").replace("\n", " ")
    parts = [part.strip() for part in text.split(", ") if part.strip()]
    if not parts:
        return []

    fragments: List[str] = []
    age = parts[0]
    idx = 1
    gender = None
    race = None
    if idx < len(parts) and _looks_like_gender_fragment(parts[idx]):
        gender = parts[idx]
        idx += 1
    if idx < len(parts) and _looks_like_race_fragment(parts[idx]):
        race = parts[idx]
        idx += 1

    identity_bits: List[str] = []
    if age:
        identity_bits.append(age)
    if race and gender:
        identity_bits.append(f"{race} {gender}")
    else:
        if race:
            identity_bits.append(race)
        if gender:
            identity_bits.append(gender)
    if identity_bits:
        fragments.append(", ".join(identity_bits))
    else:
        fragments.append(age)

    fragments.extend(parts[idx:])
    return fragments


def _looks_like_gender_fragment(text: str) -> bool:
    """
    Heuristically determine whether ``text`` refers to gender.

    :param text: Fragment extracted from a viewer sentence.
    :type text: str
    :returns: ``True`` when gender-related keywords are detected.
    :rtype: bool
    """

    lowered = text.lower()
    gender_keywords = [
        "man",
        "woman",
        "non-binary",
        "nonbinary",
        "trans",
        "prefers not to state",
        "gender",
    ]
    return any(keyword in lowered for keyword in gender_keywords)


def _looks_like_race_fragment(text: str) -> bool:
    """
    Heuristically determine whether ``text`` refers to race or ethnicity.

    :param text: Fragment extracted from a viewer sentence.
    :type text: str
    :returns: ``True`` when race-related keywords are detected.
    :rtype: bool
    """

    lowered = text.lower()
    race_keywords = [
        "white",
        "black",
        "african",
        "asian",
        "hispanic",
        "latino",
        "latinx",
        "pacific",
        "alaska",
        "native",
        "multiracial",
        "mixed",
        "american indian",
        "race",
    ]
    return any(keyword in lowered for keyword in race_keywords)


def _location_fragment(profile: ProfileRender) -> str:
    """
    Extract a location fragment from profile sentences when available.

    :param profile: Rendered profile containing viewer sentences.
    :type profile: ProfileRender
    :returns: Fragment such as ``\"lives in Seattle, Washington\"`` or ``\"\"`` if none found.
    :rtype: str
    """

    for sentence in profile.sentences:
        stripped = sentence.strip()
        lowered = stripped.lower()
        if lowered.startswith("they live in "):
            fragment = stripped[12:].strip().rstrip(".")
            return f"lives in {fragment}"
        if lowered.startswith("they reside in "):
            fragment = stripped[14:].strip().rstrip(".")
            return f"resides in {fragment}"
    return ""


def _initial_viewpoint_line(
    ex: Dict[str, Any],
    selected: Dict[str, Any],
) -> str:
    """
    Construct a concise description of the viewer's initial stance on the active issue.

    :param ex: Dataset example containing issue identifier and survey responses.
    :type ex: Dict[str, Any]
    :param selected: Normalised ``selected_survey_row`` mapping.
    :type selected: Dict[str, Any]
    :returns: Natural-language stance description, or an empty string when the issue is unknown.
    :rtype: str
    """

    issue_raw = first_non_nan_value(ex, selected, "issue") or ex.get("topic")
    issue = str(issue_raw or "").strip().lower()
    if issue == "minimum_wage":
        return _minimum_wage_viewpoint(ex, selected)
    if issue == "gun_control":
        return _gun_control_viewpoint(ex, selected)
    return ""


def _first_yes_no(
    ex: Dict[str, Any],
    selected: Dict[str, Any],
    keys: Sequence[str],
) -> Optional[str]:
    """
    Retrieve the first yes/no style response across ``keys``.

    :param ex: Dataset example consulted before fallback values.
    :type ex: Dict[str, Any]
    :param selected: Normalised ``selected_survey_row`` mapping.
    :type selected: Dict[str, Any]
    :param keys: Ordered field names searched for boolean responses.
    :type keys: Sequence[str]
    :returns: Lowercase ``\"yes\"`` or ``\"no\"`` when found, otherwise ``None``.
    :rtype: Optional[str]
    """

    for key in keys:
        value = first_non_nan_value(ex, selected, key)
        if value is None:
            continue
        verdict = format_yes_no(value)
        if verdict:
            return verdict.lower()
    return None


def _minimum_wage_viewpoint(
    ex: Dict[str, Any],
    selected: Dict[str, Any],
) -> str:
    """
    Summarise the viewer's stance on minimum wage policy.

    :param ex: Dataset example with minimum wage survey responses.
    :type ex: Dict[str, Any]
    :param selected: Normalised ``selected_survey_row`` mapping.
    :type selected: Dict[str, Any]
    :returns: Sentence describing support level or preferred wage, or ``\"\"`` if indeterminate.
    :rtype: str
    """

    def yes_no_phrase(keys: Sequence[str], yes: str, no: str) -> Optional[str]:
        """
        Map yes/no style responses across ``keys`` to canned phrases.

        :param keys: Ordered field names examined for boolean responses.
        :type keys: Sequence[str]
        :param yes: Phrase returned when a ``\"yes\"`` response is found.
        :type yes: str
        :param no: Phrase returned when a ``\"no\"`` response is found.
        :type no: str
        :returns: Matching phrase or ``None`` when no definitive response exists.
        :rtype: Optional[str]
        """
        verdict = _first_yes_no(ex, selected, keys)
        return {"yes": yes, "no": no}.get(verdict or "")

    def first_formatted(keys: Sequence[str], template: str) -> Optional[str]:
        """
        Format the first non-missing value across ``keys`` using ``template``.

        :param keys: Candidate field names to inspect.
        :type keys: Sequence[str]
        :param template: ``str.format`` template that receives the formatted value.
        :type template: str
        :returns: Rendered string or ``None`` when no values are available.
        :rtype: Optional[str]
        """
        for key in keys:
            value = first_non_nan_value(ex, selected, key)
            if value is None:
                continue
            formatted = format_field_value(key, value)
            if formatted:
                return template.format(value=formatted)
        return None

    candidates = (
        yes_no_phrase(
            ("minwage15_w2", "minwage15_w1"),
            "Supports a $15 minimum wage",
            "Opposes a $15 minimum wage",
        ),
        yes_no_phrase(
            ("mw_support_w2", "mw_support_w1"),
            "Supports raising the minimum wage",
            "Opposes raising the minimum wage",
        ),
        first_formatted(
            ("minwage_text_r_w2", "minwage_text_r_w1", "minwage_text_w2", "minwage_text_w1"),
            "Preferred minimum wage is about {value}",
        ),
        first_formatted(
            ("mw_index_w2", "mw_index_w1"),
            "Minimum wage support score is {value}",
        ),
    )
    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def _gun_control_viewpoint(
    ex: Dict[str, Any],
    selected: Dict[str, Any],
) -> str:
    """
    Summarise the viewer's stance on gun control policy.

    :param ex: Dataset example with gun policy responses.
    :type ex: Dict[str, Any]
    :param selected: Normalised ``selected_survey_row`` mapping.
    :type selected: Dict[str, Any]
    :returns: Sentence describing support level for specific gun policies, or ``\"\"`` if unknown.
    :rtype: str
    """

    for key, (yes_phrase, no_phrase) in (
        ("stricter_laws", ("Supports stricter gun laws", "Opposes stricter gun laws")),
        ("handgun_ban", ("Supports a handgun ban", "Opposes a handgun ban")),
        (
            "assault_ban",
            ("Supports an assault weapons ban", "Opposes an assault weapons ban"),
        ),
        (
            "concealed_safe",
            ("Believes concealed carry is safe", "Believes concealed carry is unsafe"),
        ),
    ):
        verdict = _first_yes_no(ex, selected, (key,))
        if verdict == "yes":
            return yes_phrase
        if verdict == "no":
            return no_phrase

    for key in ("gun_policy", "gun_identity", "gun_priority"):
        value = first_non_nan_value(ex, selected, key)
        formatted = format_field_value(key, value) if value is not None else ""
        if formatted:
            return formatted[0].upper() + formatted[1:] if formatted else ""

    for key, template in (
        ("gun_importance", "Gun policy importance is {value}"),
        ("gun_index", "Gun regulation support score is {value}"),
        ("gun_index_2", "Gun regulation support score (alt) is {value}"),
        ("gun_enthusiasm", "Gun enthusiasm is {value}"),
    ):
        value = first_non_nan_value(ex, selected, key)
        formatted = format_field_value(key, value) if value is not None else ""
        if formatted:
            return template.format(value=formatted)
    return ""


def _current_video_line(ex: Dict[str, Any], show_ids: bool) -> str:
    """
    Summarise the current video being watched.

    :param ex: Dataset example containing ``current_video_*`` fields.
    :type ex: Dict[str, Any]
    :param show_ids: Whether to include the YouTube video id in the output.
    :type show_ids: bool
    :returns: Readable description of the active video or ``\"\"`` if no data exists.
    :rtype: str
    """

    title = clean_text(ex.get("current_video_title"), limit=160)
    current_id = clean_text(ex.get("current_video_id"))
    channel = clean_text(
        ex.get("current_video_channel") or ex.get("current_video_channel_title")
    )
    if not (title or channel or current_id):
        return ""
    descriptor = title or (f"video id {current_id}" if current_id else "a video")
    if channel:
        descriptor += f" (from {channel})"
    if current_id and (show_ids or not title):
        descriptor += f" [id {current_id}]"
    return descriptor


def _history_lines(ex: Dict[str, Any], show_ids: bool, max_hist: int) -> List[str]:
    """
    Format prior watch history as numbered lines.

    :param ex: Dataset example containing ``watch_history`` details.
    :type ex: Dict[str, Any]
    :param show_ids: Whether to append YouTube ids to each entry.
    :type show_ids: bool
    :param max_hist: Maximum number of history entries to include.
    :type max_hist: int
    :returns: Ordered list of numbered history lines, newest last.
    :rtype: List[str]
    """

    prior_entries = _prior_entries(ex)
    if not prior_entries:
        return []
    limit = max_hist if max_hist and max_hist > 0 else len(prior_entries)
    recent = prior_entries[-limit:]
    descriptors: List[str] = []
    for record in recent:
        if not isinstance(record, dict):
            continue
        descriptor = _watched_descriptor(record, show_ids)
        if descriptor:
            descriptors.append(descriptor)
    if not descriptors:
        return []
    return [f"{idx}. {value}" for idx, value in enumerate(descriptors, 1)]


def _survey_lines(ex: Dict[str, Any]) -> List[str]:
    """
    Produce sentence fragments highlighting survey responses.

    :param ex: Dataset example potentially containing ``survey_highlights`` metadata.
    :type ex: Dict[str, Any]
    :returns: List containing a single formatted survey highlight sentence, or
        empty when unavailable.
    :rtype: List[str]
    """

    sentence = _survey_highlights(ex).strip()
    if not sentence:
        return []
    prefix = "Survey highlights:"
    if sentence.lower().startswith(prefix.lower()):
        sentence = sentence[len(prefix) :].strip()
    return [sentence] if sentence else []


def _options_lines(ex: Dict[str, Any], show_ids: bool) -> List[str]:
    """
    Render recommendation options as numbered lines.

    :param ex: Dataset example containing ``slate_items_json`` data.
    :type ex: Dict[str, Any]
    :param show_ids: Whether to display YouTube ids alongside titles.
    :type show_ids: bool
    :returns: List of formatted option strings ready for inclusion in the prompt.
    :rtype: List[str]
    """

    items = as_list_json(ex.get("slate_items_json"))
    if not items:
        return []
    lines: List[str] = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        line = _option_line(index, item, show_ids)
        if line:
            lines.append(line)
    return lines


SURVEY_HIGHLIGHT_SPECS: Sequence[tuple[str, str]] = (
    ("pid1", "party identification is {value}"),
    ("pid2", "party lean is {value}"),
    ("ideo1", "ideology is {value}"),
    ("pol_interest", "political interest is {value}"),
    ("religpew", "religious affiliation is {value}"),
    ("freq_youtube", "watches YouTube {value}"),
    ("newsint", "{value}"),
)

MIN_WAGE_HIGHLIGHT_SPECS: Sequence[tuple[Sequence[str], str]] = (
    (("minwage_text_w2", "minwage_text_w1"), "preferred minimum wage target is {value}"),
    (("mw_support_w2", "mw_support_w1"), "minimum wage support score is {value}"),
    (("minwage15_w2", "minwage15_w1"), "$15 minimum wage support is {value}"),
)

GUN_HIGHLIGHT_SPECS: Sequence[tuple[Sequence[str], str]] = (
    (("gun_importance",), "gun policy importance is {value}"),
    (("gun_index",), "gun regulation support score is {value}"),
    (("gun_enthusiasm",), "{value}"),
)

_PERCENTAGE_WORD_FIELDS: set[str] = {
    "pol_interest",
    "mw_support_w1",
    "mw_support_w2",
    "minwage15_w1",
    "minwage15_w2",
}

_UNDER_TWENTY: dict[int, str] = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

_TENS: dict[int, str] = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def _int_to_words(number: int) -> Optional[str]:
    """
    Convert ``number`` (0–999) into a hyphenated English phrase.

    :param number: Integer value to verbalise.
    :type number: int
    :returns: Lowercase English words or ``None`` when the number is outside the supported range.
    :rtype: Optional[str]
    """

    if number < 0 or number > 999:
        return None
    if number < 20:
        return _UNDER_TWENTY[number]

    result: Optional[str]
    if number < 100:
        tens, remainder = divmod(number, 10)
        base = _TENS.get(tens * 10)
        if not base:
            return None
        result = base if remainder == 0 else f"{base}-{_UNDER_TWENTY[remainder]}"
    else:
        hundreds, remainder = divmod(number, 100)
        if hundreds not in _UNDER_TWENTY:
            return None
        result = f"{_UNDER_TWENTY[hundreds]} hundred"
        if remainder:
            tail = _int_to_words(remainder)
            if not tail:
                return None
            result = f"{result} {tail}"
    return result


def _percentage_to_words(value: str) -> Optional[str]:
    """
    Convert a percentage string into an approximate verbal description.

    :param value: Percentage text such as ``\"65%\"``.
    :type value: str
    :returns: Phrase like ``\"about sixty-five percent\"`` or ``None`` if parsing fails.
    :rtype: Optional[str]
    """

    text = value.strip()
    if not text.endswith("%"):
        return None
    numeric_part = text[:-1].replace(",", "").strip()
    try:
        amount = float(numeric_part)
    except ValueError:
        return None
    rounded = int(round(amount))
    word = _int_to_words(rounded)
    if not word:
        return None
    prefix = ""
    if not math.isclose(amount, rounded, abs_tol=0.05):
        prefix = "about "
    return f"{prefix}{word} percent"


def _highlight_value(field: str, raw_value: Any) -> str:
    """
    Convert ``raw_value`` into a highlight-ready description for ``field``.

    :param field: Dataset field name associated with the highlight.
    :type field: str
    :param raw_value: Original value retrieved from the interaction row.
    :type raw_value: Any
    :returns: Readable summary string or ``\"\"`` if the field should be skipped.
    :rtype: str
    """

    if field == "gun_enthusiasm":
        verdict = format_yes_no(raw_value, yes="yes", no="no")
        if verdict == "yes":
            return "identifies as enthusiastic about guns"
        if verdict == "no":
            return "does not identify as enthusiastic about guns"
    value = format_field_value(field, raw_value)
    if not value:
        return ""
    if field in _PERCENTAGE_WORD_FIELDS:
        descriptive = _percentage_to_words(value)
        if descriptive:
            return descriptive
    return value


def _survey_highlights(ex: Dict[str, Any]) -> str:
    """
    Build a sentence summarising the most relevant survey highlights.

    :param ex: Dataset example containing highlight candidate fields.
    :type ex: Dict[str, Any]
    :returns: Sentence prefixed with ``\"Survey highlights:\"`` or ``\"\"`` if nothing applies.
    :rtype: str
    """

    highlights: List[str] = []
    for field, template in SURVEY_HIGHLIGHT_SPECS:
        value = _highlight_value(field, ex.get(field))
        if not value:
            continue
        highlights.append(template.format(value=value))

    issue = str(ex.get("issue") or "").strip().lower()
    issue_specs = MIN_WAGE_HIGHLIGHT_SPECS if issue == "minimum_wage" else ()
    if issue == "gun_control":
        issue_specs = GUN_HIGHLIGHT_SPECS
    for fields, template in issue_specs:
        value: Optional[str] = None
        for field in fields:
            candidate = _highlight_value(field, ex.get(field))
            if candidate:
                value = candidate
                break
        if value:
            highlights.append(template.format(value=value))

    if not highlights:
        return ""
    return f"Survey highlights: {human_join(highlights)}."


def _option_line(position: int, item: Dict[str, Any], show_ids: bool) -> str:
    """
    Render a single entry in the recommendation slate.

    :param position: One-based index of the recommendation.
    :type position: int
    :param item: Slate item metadata including title, id, and channel.
    :type item: Dict[str, Any]
    :param show_ids: Whether to append the YouTube id when present.
    :type show_ids: bool
    :returns: Fully formatted option line ready for inclusion in the prompt.
    :rtype: str
    """

    title = clean_text(item.get("title"), limit=160)
    option_id = clean_text(item.get("id"))
    display_title = title or option_id or "(untitled)"
    channel = clean_text(
        item.get("channel_title")
        or item.get("channel")
        or item.get("channel_name")
    )
    duration_text = _format_duration(item)
    descriptors: List[str] = []
    if channel:
        descriptors.append(channel)
    if duration_text:
        descriptors.append(f"{duration_text} long")
    if option_id and (show_ids or not title):
        descriptors.append(f"id {option_id}")
    stats = lookup_video_stats(option_id) if option_id else {}
    if stats is None:
        stats = {}
    engagement = _option_engagement_summary(item, stats)
    line = f"{position}. {display_title}"
    if descriptors:
        line += f" ({', '.join(descriptors)})"
    if engagement:
        line += f" — Engagement: {engagement}"
    return line


def _prior_entries(ex: Dict[str, Any]) -> List[dict]:
    """
    Return watch-history entries that precede the current video.

    :param ex: Interaction row containing watch history payloads.
    :type ex: Dict[str, Any]
    :returns: List of entry dictionaries ordered chronologically.
    :rtype: List[dict]
    """
    vids = as_list_json(ex.get("watched_vids_json"))
    detailed = as_list_json(ex.get("watched_detailed_json"))
    current_id = clean_text(ex.get("current_video_id"))
    cur_idx = None
    if current_id:
        cur_idx = _last_index(vids, current_id)
        if cur_idx is None and isinstance(detailed, list):
            for index in range(len(detailed) - 1, -1, -1):
                entry = detailed[index]
                if isinstance(entry, dict) and clean_text(entry.get("id")) == current_id:
                    cur_idx = index
                    break
    if cur_idx is None and isinstance(vids, list) and vids:
        cur_idx = len(vids) - 1
    if isinstance(detailed, list) and cur_idx is not None and cur_idx > 0:
        return detailed[:cur_idx]
    return []


def _watched_descriptor(record: Dict[str, Any], show_ids: bool) -> str:
    """
    Produce a narrative description of a previously watched video.

    :param record: Dictionary containing watch metadata.
    :type record: Dict[str, Any]
    :param show_ids: Whether to include identifiers in the descriptor.
    :type show_ids: bool
    :returns: Descriptor string describing watch progress and metadata.
    :rtype: str
    """
    title = clean_text(
        record.get("title") or record.get("name") or record.get("video_title"),
        limit=160,
    )
    rid = clean_text(record.get("id"))
    channel = clean_text(record.get("channel_title") or record.get("channel"))
    watch_seconds_value = _extract_duration_seconds(
        record,
        (
            "watch_seconds",
            "watch_duration",
            "watch_time",
            "watch_ms",
        ),
    )
    total_length_value = _extract_duration_seconds(
        record,
        (
            "total_length",
            "total_duration",
            "duration_seconds",
            "length_seconds",
            "total_length_ms",
            "duration",
        ),
    )
    name = title or "(untitled)"
    descriptors: List[str] = []
    if watch_seconds_value is not None and watch_seconds_value > 0:
        watch_int = int(round(watch_seconds_value))
        if total_length_value is not None and total_length_value > 0:
            total_int = int(round(total_length_value))
            ratio = min(1.0, watch_seconds_value / max(total_length_value, 1e-6))
            descriptors.append(
                f"watched {watch_int}s of {total_int}s ({int(round(ratio * 100))}% complete)"
            )
        else:
            descriptors.append(f"watched for {watch_int}s")
    if channel:
        descriptors.append(f"from {channel}")
    if show_ids and rid:
        descriptors.append(f"id {rid}")
    if descriptors:
        return f"{name} ({', '.join(descriptors)})"
    return name


def _extract_duration_seconds(
    record: Dict[str, Any],
   keys: Sequence[str],
) -> Optional[float]:
    """
    Extract the first positive duration from ``record`` across ``keys``.

    :param record: Mapping containing raw duration values.
    :type record: Dict[str, Any]
    :param keys: Candidate field names inspected in order.
    :type keys: Sequence[str]
    :returns: Duration in seconds when a positive value is found, otherwise ``None``.
    :rtype: Optional[float]
    """

    for key in keys:
        if key not in record:
            continue
        raw_value = record.get(key)
        if raw_value is None:
            continue
        try:
            duration = float(raw_value)
        except (TypeError, ValueError):
            continue
        if duration <= 0:
            continue
        if key.endswith("_ms") and duration > 1000:
            duration /= 1000.0
        return duration
    return None


def _format_duration(item: Dict[str, Any]) -> str:
    """
    Convert raw duration metadata into a seconds string.

    :param item: Dictionary containing duration metadata.
    :type item: Dict[str, Any]
    :returns: Seconds string such as ``"120s"`` or an empty string when unavailable.
    :rtype: str
    """
    duration_raw = (
        item.get("length_seconds")
        or item.get("duration_seconds")
        or item.get("duration")
    )
    if duration_raw is None or not str(duration_raw).strip():
        return ""
    try:
        duration_val = float(duration_raw)
    except (TypeError, ValueError):
        return ""
    if duration_val <= 0:
        return ""
    return f"{int(round(duration_val))}s"


def _option_engagement_summary(item: Dict[str, Any], stats: Dict[str, Any]) -> str:
    """
    Summarise engagement metrics for a recommendation option.

    :param item: Slate item metadata containing inline engagement counts.
    :type item: Dict[str, Any]
    :param stats: Lookup statistics sourced from
        :func:`prompt_builder.video_stats.lookup_video_stats`.
    :type stats: Dict[str, Any]
    :returns: Comma-separated phrase describing views, likes, comments, and share
        counts when available.
    :rtype: str
    """

    def first_present(source: Dict[str, Any], keys: Sequence[str]) -> Any:
        """
        Return the first non-missing value from ``source`` matching ``keys``.

        :param source: Mapping containing potential metric values.
        :type source: Dict[str, Any]
        :param keys: Field names to probe in order.
        :type keys: Sequence[str]
        :returns: Raw value when found, otherwise ``None``.
        :rtype: Any
        """

        for key in keys:
            if key not in source:
                continue
            value = source.get(key)
            if is_nanlike(value):
                continue
            text = str(value).strip()
            if not text:
                continue
            return value
        return None

    metrics = [
        (
            "views",
            (
                "view_count",
                "views",
                "viewCount",
                "ViewCount",
                "viewer_count",
                "viewerCount",
                "Viewers",
            ),
            ("view_count", "views", "viewCount", "ViewCount", "viewer_count", "viewerCount"),
        ),
        (
            "likes",
            ("like_count", "likes", "likeCount", "LikeCount", "Likes"),
            ("like_count", "likes", "likeCount", "LikeCount"),
        ),
        (
            "comments",
            ("comment_count", "comments", "commentCount", "CommentCount", "Comments"),
            ("comment_count", "comments", "commentCount", "CommentCount"),
        ),
        (
            "shares",
            ("share_count", "shares", "shareCount", "ShareCount", "Shares"),
            ("share_count", "shares", "shareCount", "ShareCount"),
        ),
    ]
    parts: List[str] = []
    for label, item_keys, stat_keys in metrics:
        value = first_present(item, item_keys)
        if value is None and stats:
            value = first_present(stats, stat_keys)
            if value is None and stat_keys:
                value = stats.get(stat_keys[0])
        formatted = format_count(value)
        if formatted is None:
            continue
        parts.append(f"{label} {formatted}")
    return ", ".join(parts)


__all__ = [
    "build_user_prompt",
    "synthesize_viewer_sentence",
]
