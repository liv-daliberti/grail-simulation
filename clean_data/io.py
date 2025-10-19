"""Input/output primitives for ingesting capsule artefacts and metadata.

The functions in this module know how to locate capsule directories,
read CSV/RDS exports from the CodeOcean bundle, parse supplemental
metadata, and normalise the resulting structures for downstream use.
They form the foundation upon which the session parser reconstructs the
full interaction dataset.
"""

from __future__ import annotations

import ast
import csv
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from clean_data.helpers import (
    _coerce_session_value,
    _is_nanlike,
    _strip_session_video_id,
)

log = logging.getLogger("clean_grail")


def resolve_capsule_data_root(path: Path) -> Optional[Path]:
    """Locate the CodeOcean capsule root or ``data`` directory from a user path.

    :param path: Candidate path provided by the caller.
    :returns: Normalized capsule root if detected, otherwise ``None``.
    """
    if not path.exists():
        return None
    if (path / "platform session data" / "sessions.json").exists():
        return path
    if (path / "data" / "platform session data" / "sessions.json").exists():
        return path / "data"
    return None


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Read a CSV file into a dataframe, returning empty on failure.

    :param path: Absolute or relative CSV path.
    :returns: Dataframe containing the file contents, or empty when unavailable.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        log.error("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


# pylint: disable=too-many-return-statements
def read_survey_with_fallback(*candidates: Path) -> pd.DataFrame:
    """Return the first survey export that includes a ``urlid`` column.

    :param candidates: Ordered sequence of candidate paths.
    :returns: Dataframe of the first usable survey export, or empty when none exist.
    """
    fallback_df: Optional[pd.DataFrame] = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        df = read_csv_if_exists(candidate)
        columns = [str(c).strip().lower() for c in getattr(df, "columns", [])]
        if "urlid" in columns:
            return df
        if df.empty and fallback_df is None:
            fallback_df = df
        else:
            log.warning("Survey frame missing urlid column; columns=%s", list(df.columns))
    if fallback_df is not None:
        return fallback_df
    if candidates:
        log.warning("Survey file missing: %s", candidates[0])
    return pd.DataFrame()


def _escape_r_string(path: Path) -> str:
    """Escape a filesystem path so it can be embedded in inline R code.

    :param path: Path to convert.
    :returns: Escaped string representation safe for ``Rscript``.
    """

    text = str(path)
    text = text.replace("\\", "/")
    return text.replace("'", "\\'")


def read_rds_dataframe(path: Path) -> pd.DataFrame:
    """Convert an RDS file into a dataframe via ``Rscript``.

    :param path: Location of the ``.rds`` file.
    :returns: Parsed dataframe or empty dataframe when conversion fails.
    """
    if not path.exists():
        return pd.DataFrame()
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        path_str = _escape_r_string(path)
        tmp_str = _escape_r_string(tmp_path)
        r_code = (
            "options(warn=2);"
            f"d <- readRDS('{path_str}');"
            "if (!is.data.frame(d)) d <- as.data.frame(d);"
            f"write.csv(d, file='{tmp_str}', row.names=FALSE, na='')"
        )
        subprocess.run(
            ["Rscript", "-e", r_code],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        df = pd.read_csv(tmp_path, dtype=str)
        return df
    except subprocess.CalledProcessError as exc:
        log.warning("Rscript failed while loading %s: %s", path, exc)
        return pd.DataFrame()
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        log.warning("Failed to parse RDS %s: %s", path, exc)
        return pd.DataFrame()
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def maybe_literal_eval(value: Any) -> Any:
    """Convert stringified lists/dicts/bools from R exports into native objects.

    :param value: Raw value emitted by the R exports.
    :returns: Parsed Python value or the original value when parsing fails.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"na", "nan", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "inf":
        return float("inf")
    if lowered == "-inf":
        return float("-inf")
    if text.startswith("{") or text.startswith("["):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
    return text


def load_shorts_sessions(data_root: Path) -> List[Dict[str, Any]]:
    """Load Shorts (Study 4) interaction logs from the YTRecs RDS export.

    :param data_root: Capsule ``data`` directory.
    :returns: List of normalized session dictionaries.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    rds_path = data_root / "shorts" / "ytrecs_sessions_may2024.rds"
    df = read_rds_dataframe(rds_path)
    if df.empty:
        return []

    df = df.where(pd.notna(df), None)
    sessions: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        session: Dict[str, Any] = {}
        for col in df.columns:
            value = row[col]
            if value is None:
                session[col] = None
                continue
            if isinstance(value, str):
                parsed = maybe_literal_eval(value)
                session[col] = parsed
            else:
                session[col] = value

        list_like_keys = [
            "vids",
            "ignoredVideos",
            "optionalInstructions",
            "pauseInteractions",
            "playInteractions",
            "ratingEvents",
            "ratingResults",
            "replayInteractions",
            "saveList",
            "seekBackwardInteractions",
            "seekForwardInteractions",
            "skipInteractions",
            "thumbInteractions",
        ]
        for key in list_like_keys:
            value = session.get(key)
            if isinstance(value, str):
                parsed = maybe_literal_eval(value)
                session[key] = parsed if isinstance(parsed, list) else []
            elif value is None:
                session[key] = []
            elif not isinstance(value, list):
                session[key] = list(value) if value is not None else []

        dict_like_keys = [
            "displayOrders",
            "vidEndTimes",
            "vidStartTimes",
            "vidTotalLengths",
            "vidWatchTimes",
            "thumbStates",
        ]
        for key in dict_like_keys:
            value = session.get(key)
            if isinstance(value, str):
                parsed = maybe_literal_eval(value)
                session[key] = parsed if isinstance(parsed, dict) else {}
            elif value is None:
                session[key] = {}

        vids_value = session.get("vids")
        if isinstance(vids_value, (list, tuple)):
            session["vids"] = [
                str(v).strip() for v in vids_value if isinstance(v, str) and v.strip()
            ]
        else:
            session["vids"] = []

        for key in ("endTime", "startTime"):
            value = session.get(key)
            if isinstance(value, str) and value.strip().isdigit():
                session[key] = int(value.strip())

        percent_visible = session.get("percentVisible")
        if isinstance(percent_visible, str):
            try:
                session["percentVisible"] = float(percent_visible)
            except ValueError:
                session["percentVisible"] = None

        session_finished = session.get("sessionFinished")
        if isinstance(session_finished, str):
            session["sessionFinished"] = session_finished.strip().lower() == "true"

        any_watched = session.get("anyVideoWatched")
        if isinstance(any_watched, str):
            session["anyVideoWatched"] = any_watched.strip().lower() == "true"

        sessions.append(session)

    log.info("Loaded %d Shorts sessions from %s", len(sessions), rds_path)
    return sessions


TEXT_SUFFIXES = {
    "title": ["Title", "title", "Name"],
    "channel_title": ["Channel", "ChannelTitle", "channel"],
    "channel_id": ["ChannelID", "ChannelId", "channel_id"],
    "description": ["Description", "description"],
    "category": ["Cat", "Category", "category"],
    "caption": ["Caption", "caption"],
}

NUMERIC_SUFFIXES = {
    "view_count": ["ViewCount", "viewCount", "Views", "views"],
    "like_count": ["LikeCount", "likeCount", "Likes", "likes"],
    "dislike_count": ["DislikeCount", "dislikeCount", "Dislikes", "dislikes"],
    "favorite_count": ["FavoriteCount", "favoriteCount", "Favorites", "favorites"],
    "comment_count": ["CommentCount", "commentCount", "Comments", "comments"],
    "duration": ["Duration", "duration", "DurationSeconds", "duration_seconds"],
}


def assign_if_missing(info: Dict[str, Any], key: str, value: Any) -> None:
    """Populate ``info[key]`` when the destination field is empty or placeholder.

    :param info: Metadata dictionary that may contain an existing value.
    :param key: Target key within ``info``.
    :param value: Candidate value to assign.
    """
    if value is None:
        return
    if isinstance(value, str):
        sval = value.strip()
        if not sval or _is_nanlike(sval):
            return
        value = sval
    if key in info:
        existing = info[key]
        if isinstance(existing, str):
            stripped = existing.strip()
            if stripped and not stripped.startswith("(title missing"):
                return
        elif existing not in {None, "", 0}:
            return
    info[key] = value


def _find_prefixed_value(
    row: Dict[str, Any],
    prefix: str,
    suffix: str,
    allow_unprefixed: bool = False,
) -> Optional[Any]:
    """Return the first non-empty value matching a prefix/suffix combination.

    :param row: Row dictionary emitted by CSV readers.
    :param prefix: Column prefix to search for.
    :param suffix: Column suffix to search for.
    :param allow_unprefixed: When ``True``, fall back to bare suffix matches.
    :returns: Matching value or ``None`` when no candidate is found.
    """
    candidates = []
    if prefix:
        candidates.extend(
            [
                f"{prefix}{suffix}",
                f"{prefix}_{suffix}",
                f"{prefix}{suffix.lower()}",
                f"{prefix}_{suffix.lower()}",
                f"{prefix}{suffix.upper()}",
                f"{prefix}_{suffix.upper()}",
            ]
        )
    if allow_unprefixed or not prefix:
        candidates.extend([suffix, suffix.lower(), suffix.upper()])
    for cand in candidates:
        if cand in row and row[cand] not in (None, ""):
            return row[cand]
    return None


def _apply_metadata_fields(
    info: Dict[str, Any],
    row: Dict[str, Any],
    prefix: str,
    allow_unprefixed: bool = False,
) -> None:
    """Populate metadata fields using a prefix-based lookup table.

    :param info: Mutable video metadata dictionary.
    :param row: CSV row currently being processed.
    :param prefix: Column prefix applied to the supplemental fields.
    :param allow_unprefixed: When ``True``, also consider bare suffix columns.
    """
    for dest, suffixes in TEXT_SUFFIXES.items():
        for suffix in suffixes:
            val = _find_prefixed_value(row, prefix, suffix, allow_unprefixed)
            if val is not None:
                assign_if_missing(info, dest, str(val))
                break
    for dest, suffixes in NUMERIC_SUFFIXES.items():
        for suffix in suffixes:
            val = _find_prefixed_value(row, prefix, suffix, allow_unprefixed)
            if val is not None:
                assign_if_missing(info, dest, _coerce_session_value(val))
                break


def fill_metadata_field(
    info: Dict[str, Any],
    row: Dict[str, Any],
    key: str,
    *candidate_columns: str,
) -> None:
    """Populate ``info[key]`` from the first valid entry in ``candidate_columns``.

    :param info: Metadata dictionary receiving the value.
    :param row: Row containing potential field values.
    :param key: Target metadata key to populate.
    :param candidate_columns: Ordered list of column names to inspect.
    """
    if info.get(key):
        return
    for col in candidate_columns:
        if not col:
            continue
        value = row.get(col)
        if value is None:
            continue
        sval = str(value).strip()
        if not sval or _is_nanlike(sval):
            continue
        info[key] = sval
        return


def augment_metadata_from_supplemental(
    meta: Dict[str, Dict[str, Any]], base_dir: Path
) -> None:
    """Enrich the metadata index with supplemental CSV content.

    :param meta: Existing metadata mapping keyed by video id.
    :param base_dir: Capsule base directory containing the supplemental bundle.
    """
    sup_dir = base_dir / "supplemental" / "metadata and ratings"
    if not sup_dir.exists():
        return
    for csv_path in sup_dir.glob("*.csv"):
        try:
            with csv_path.open("r", encoding="utf-8-sig") as fp:
                reader = csv.DictReader(fp)
                if not reader.fieldnames:
                    continue
                for row in reader:
                    for prefix, id_keys, allow_unprefixed in [
                        ("origin", ["originID", "originId", "origin_id"], True),
                        ("rec", ["recID", "recId", "rec_id"], False),
                        ("video", ["video_id", "videoId", "videoID"], True),
                        ("", ["id", "ID"], True),
                    ]:
                        vid_raw = None
                        for key in id_keys:
                            if key in row and row[key]:
                                vid_raw = row[key]
                                break
                        if not vid_raw:
                            continue
                        vid = _strip_session_video_id(str(vid_raw))
                        if not vid:
                            continue
                        info = meta.setdefault(vid, {"id": vid})
                        _apply_metadata_fields(info, row, prefix, allow_unprefixed)
        except (OSError, UnicodeDecodeError, csv.Error, ValueError) as exc:
            log.warning("Failed to read supplemental metadata %s: %s", csv_path, exc)


def load_video_metadata(base_dir: Path) -> Dict[str, str]:
    """Load supplemental video titles from the metadata bundle.

    :param base_dir: Capsule base directory containing supplemental metadata.
    :returns: Mapping of canonical video id to title string.
    """
    # pylint: disable=too-many-branches
    meta: Dict[str, str] = {}
    meta_dir = base_dir / "supplemental" / "metadata and ratings"
    if not meta_dir.exists():
        return meta
    for csv_path in meta_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            log.warning("Failed to read metadata %s: %s", csv_path, exc)
            continue
        id_col = None
        for cand in ("originID", "originId", "video_id", "videoId", "id"):
            if cand in df.columns:
                id_col = cand
                break
        if not id_col:
            continue
        title_col = None
        for cand in ("title", "video_title", "name"):
            if cand in df.columns:
                title_col = cand
                break
        for _, row in df.iterrows():
            vid = row.get(id_col)
            if pd.isna(vid):
                continue
            vid_str = str(vid).strip()
            if not vid_str:
                continue
            base = _strip_session_video_id(vid_str)
            if title_col and not pd.isna(row.get(title_col)):
                title = str(row.get(title_col))
            else:
                title = ""
            meta.setdefault(base, title)
    return meta


def load_recommendation_tree_metadata(
    base_dir: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Scan recommendation tree CSVs to build a metadata index keyed by video id.

    :param base_dir: Capsule base directory containing the tree exports.
    :returns: Tuple of (metadata mapping, issue map) derived from the tree CSVs.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    tree_root = base_dir / "recommendation trees"
    if not tree_root.exists():
        return {}, {}

    meta: Dict[str, Dict[str, Any]] = {}
    issue_map: Dict[str, str] = {}

    issue_dirs = [
        ("gun_control", tree_root / "trees_gun"),
        ("minimum_wage", tree_root / "trees_wage"),
    ]

    for issue_name, folder in issue_dirs:
        if not folder.exists():
            continue
        for csv_path in folder.glob("*.csv"):
            try:
                with csv_path.open("r", encoding="utf-8-sig") as fp:
                    reader = csv.DictReader(fp)
                    for row in reader:
                        origin_raw = row.get("originId") or row.get("originID") or ""
                        base_vid = _strip_session_video_id(origin_raw)
                        if not base_vid:
                            continue
                        info = meta.setdefault(base_vid, {"id": base_vid})
                        issue_map.setdefault(base_vid, issue_name)
                        if origin_raw and origin_raw != base_vid:
                            raw_ids = info.setdefault("raw_ids", [])
                            if origin_raw not in raw_ids:
                                raw_ids.append(origin_raw)

                        fill_metadata_field(
                            info,
                            row,
                            "title",
                            "originTitle",
                            "title",
                            "video_title",
                        )
                        fill_metadata_field(
                            info,
                            row,
                            "channel_id",
                            "originChannelId",
                            "channel_id",
                        )
                        fill_metadata_field(
                            info,
                            row,
                            "channel_title",
                            "originChannel",
                            "channelTitle",
                        )
                        fill_metadata_field(
                            info,
                            row,
                            "description",
                            "originDescription",
                            "description",
                        )
                        fill_metadata_field(info, row, "duration", "originDuration", "duration")
                        _apply_metadata_fields(info, row, "origin", allow_unprefixed=True)
                        for thumb_key, col in [
                            ("thumbnail_default", "originThumbnailDefault"),
                            ("thumbnail_medium", "originThumbnailMedium"),
                            ("thumbnail_high", "originThumbnailHigh"),
                        ]:
                            if info.get(thumb_key):
                                continue
                            val = row.get(col)
                            if val is None:
                                continue
                            sval = str(val).strip()
                            if sval:
                                info[thumb_key] = sval
                        if "duration" in info:
                            dur_val = _coerce_session_value(info.get("duration"))
                            if dur_val is None:
                                info.pop("duration", None)
                            else:
                                info["duration"] = dur_val

                        if info.get("recs"):
                            continue  # keep only the first observed slate for this origin
                        rec_entries = []
                        rec_cols = [
                            (int(m.group(1)), col)
                            for col in row.keys()
                            for m in [re.match(r"rec(\d+)", str(col), re.I)]
                            if m is not None
                        ]
                        rec_cols.sort(key=lambda x: x[0])
                        if rec_cols:
                            for _, col in rec_cols:
                                raw_rec = row.get(col)
                                if raw_rec is None:
                                    continue
                                raw_str = str(raw_rec).strip()
                                if not raw_str or _is_nanlike(raw_str):
                                    continue
                                rec_base = _strip_session_video_id(raw_str)
                                if not rec_base:
                                    continue
                                already = {
                                    entry.get("id")
                                    for entry in rec_entries
                                    if isinstance(entry, dict)
                                }
                                if rec_base in already:
                                    continue
                                rec_entry: Dict[str, Any] = {"id": rec_base}
                                if raw_str != rec_base:
                                    rec_entry["raw_id"] = raw_str
                                rec_entries.append(rec_entry)
                            if rec_entries:
                                info["recs"] = rec_entries
                        _apply_metadata_fields(info, row, "origin", allow_unprefixed=True)
            except (OSError, UnicodeDecodeError, csv.Error, ValueError) as exc:
                log.warning("Failed to read tree CSV %s: %s", csv_path, exc)
                continue

    augment_metadata_from_supplemental(meta, base_dir)
    return meta, issue_map


__all__ = [
    "resolve_capsule_data_root",
    "read_csv_if_exists",
    "read_survey_with_fallback",
    "read_rds_dataframe",
    "maybe_literal_eval",
    "load_shorts_sessions",
    "assign_if_missing",
    "fill_metadata_field",
    "augment_metadata_from_supplemental",
    "load_video_metadata",
    "load_recommendation_tree_metadata",
]
