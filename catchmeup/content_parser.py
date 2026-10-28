
"""Parsing utilities for ordered content collections."""

import re


def _normalize_header(value: str) -> str:
    """Normalize a table header for reliable matching."""
    value = re.sub(r"\[[^\]]*\]", "", str(value))
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def _find_column(headers: list[str], candidates: set[str]) -> int | None:
    """Find a column by matching normalized header names."""
    normalized = [_normalize_header(header) for header in headers]

    for index, header in enumerate(normalized):
        if header in candidates:
            return index

    for index, header in enumerate(normalized):
        if any(candidate in header for candidate in candidates):
            return index

    return None


def normalized_index(cells: list[str], name: str) -> int | None:
    """Return the index of a cell matching a normalized column name."""
    normalized_name = _normalize_header(name)

    for index, cell in enumerate(cells):
        if _normalize_header(cell) == normalized_name:
            return index

    return None


def _clean_table_cell(value: str) -> str:
    """Clean common wiki/Markdown formatting from a table cell."""
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"\[[^\]]*\]", "", value)
    value = re.sub(r"''+", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _split_table_row(line: str) -> list[str]:
    """Split a Markdown pipe-separated table row into cells."""
    line = line.strip()

    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]

    return [_clean_table_cell(cell) for cell in line.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    """Identify Markdown table separator rows such as |---|---|."""
    if not cells:
        return False

    return all(
        bool(re.fullmatch(r":?-{3,}:?", cell.strip()))
        for cell in cells
    )



def parse_episode_list(text: str, work_id: str) -> list[dict]:
    """Parse a numbered episode list while preserving season order."""
    if not isinstance(text, str) or not text.strip():
        return []

    units = []
    seen = set()
    current_season = 1

    season_pattern = re.compile(r"^\s*season\s+(\d+)\b.*$", re.IGNORECASE)
    episode_patterns = [
        re.compile(
            r"^\s*episode\s+(\d+)\s*[:.)\-–]\s*(.+?)\s*$",
            re.IGNORECASE,
        ),
        re.compile(r"^\s*(\d+)\s*[.):\-–]\s*(.+?)\s*$"),
    ]

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        season_match = season_pattern.match(line)
        if season_match:
            current_season = int(season_match.group(1))
            continue

        match = None
        for pattern in episode_patterns:
            match = pattern.match(line)
            if match:
                break

        if not match:
            continue

        episode_number = int(match.group(1))
        title = match.group(2).strip()
        key = (current_season, episode_number)

        if episode_number < 1 or not title or key in seen:
            continue

        seen.add(key)
        position = len(units) + 1

        units.append({
            "work_id": work_id,
            "unit_id": f"s{current_season:02d}e{episode_number:02d}",
            "position": position,
            "label": f"S{current_season}E{episode_number}",
            "text": f"Season {current_season}, Episode {episode_number}: {title}",
            "season": current_season,
            "episode": episode_number,
        })

    return units



def parse_episode_table(text: str, work_id: str) -> list[dict]:
    """Parse Markdown pipe-separated episode tables.

    Supports headers such as Season, Episode, Title, and optional
    description or synopsis columns. Duplicate season/episode rows
    are ignored. Invalid or unrecognized tables return an empty list.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    lines = text.splitlines()
    units = []
    seen = set()
    headers = None

    season_candidates = {"season", "season number", "season no"}
    episode_candidates = {
        "episode",
        "episode number",
        "episode no",
        "episode #",
        "ep",
    }
    title_candidates = {
        "title",
        "episode title",
        "name",
        "episode name",
    }
    description_candidates = {
        "description",
        "summary",
        "synopsis",
        "plot",
        "plot summary",
        "overview",
    }

    for line in lines:
        stripped = line.strip()

        # Only consider pipe-separated table rows.
        if "|" not in stripped:
            continue

        cells = _split_table_row(stripped)

        if not cells or _is_separator_row(cells):
            continue

        season_idx = _find_column(cells, season_candidates)
        episode_idx = _find_column(cells, episode_candidates)
        title_idx = _find_column(cells, title_candidates)

        # A recognized header must identify episode and title columns.
        if episode_idx is not None and title_idx is not None:
            headers = cells
            headers_season_idx = season_idx
            headers_episode_idx = episode_idx
            headers_title_idx = title_idx
            headers_description_idx = _find_column(
                cells, description_candidates
            )
            continue

        if headers is None:
            continue

        # Data rows must have the required columns.
        required_indices = [
            headers_episode_idx,
            headers_title_idx,
        ]
        if headers_season_idx is not None:
            required_indices.append(headers_season_idx)

        if max(required_indices) >= len(cells):
            continue

        episode_value = cells[headers_episode_idx]
        title = cells[headers_title_idx]

        # Episode numbers may appear as "1", "1. Title", or "Episode 1".
        episode_match = re.search(r"\d+", episode_value)
        if not episode_match or not title:
            continue

        episode_number = int(episode_match.group())
        if episode_number < 1:
            continue

        season_number = 1
        if headers_season_idx is not None:
            season_value = cells[headers_season_idx]
            season_match = re.search(r"\d+", season_value)
            if not season_match:
                continue
            season_number = int(season_match.group())

        if season_number < 1:
            continue

        duplicate_key = (season_number, episode_number)
        if duplicate_key in seen:
            continue
        seen.add(duplicate_key)

        description = ""
        if (
            headers_description_idx is not None
            and headers_description_idx < len(cells)
        ):
            description = cells[headers_description_idx].strip()

        # Keep stable IDs and sequential positions across seasons.
        position = len(units) + 1
        unit_id = f"s{season_number:02d}e{episode_number:02d}"

        unit_text = f"Season {season_number}, Episode {episode_number}: {title}"
        if description:
            unit_text += f"\n\n{description}"

        units.append(
            {
                "work_id": work_id,
                "unit_id": unit_id,
                "position": position,
                "label": f"S{season_number}E{episode_number}",
                "text": unit_text,
                "season": season_number,
                "episode": episode_number,
            }
        )

    return units



def parse_episode_guide(text: str, work_id: str = "episode-guide") -> list[dict]:
    """Parse an episode guide without confusing bare SxxEyy IDs for titles.

    Supports Markdown tables, season/episode lists, and headings such as
    ``S02E03 — The Episode Title``. When a guide has a bare ID heading, a short
    following line can supply its title. Bare IDs are never accepted as titles.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # Prefer structured tables and season-grouped lists before heading scans.
    table_units = parse_episode_table(text, work_id)
    table_units = [
        unit for unit in table_units
        if unit.get("title") and not re.fullmatch(
            r"S\d{1,2}E\d{1,3}", str(unit.get("title", "")).strip(), re.I
        )
    ]
    if len(table_units) >= 2:
        normalized_table = _normalize_guide_units(table_units)
        if len(normalized_table) >= 2:
            return normalized_table

    list_units = parse_episode_list(text, work_id)
    list_units = [
        unit for unit in list_units
        if unit.get("text") and not re.search(
            r"Season\s+\d+\s*,?\s*Episode\s+\d+\s*:\s*S\d{1,2}E\d{1,3}\s*$",
            str(unit.get("text", "")), re.I
        )
    ]
    if len(list_units) >= 2:
        normalized_list = _normalize_guide_units(list_units)
        if len(normalized_list) >= 2:
            return normalized_list

    lines = text.splitlines()
    units = []
    seen = set()
    code_line = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:\[)?S(\d{1,2})\s*E(\d{1,3})(?:\])?"
        r"(?:\s*(?:[-:|—–])\s*(.*?))?\s*$", re.I
    )
    season_line = re.compile(r"^\s*Season\s+(\d+)\b", re.I)
    episode_line = re.compile(
        r"^\s*(?:Episode\s+)?(\d{1,3})\s*[:.)—–-]\s*(.+?)\s*$", re.I
    )
    current_season = 1

    def add(season: int, episode: int, title: str):
        title = re.sub(r"\s+", " ", title).strip(" \t-*#:")
        if season < 1 or episode < 1 or not title:
            return
        if re.fullmatch(r"S\d{1,2}E\d{1,3}", title, re.I):
            return
        if len(title) > 180 or title.lower() in {"episode guide", "episodes", "season", "title"}:
            return
        key = (season, episode)
        if key in seen:
            return
        seen.add(key)
        units.append({
            "work_id": work_id,
            "unit_id": f"s{season:02d}e{episode:02d}",
            "position": len(units) + 1,
            "label": f"S{season:02d}E{episode:02d}",
            "text": f"Season {season}, Episode {episode}: {title}",
            "season": season,
            "episode": episode,
            "title": title,
        })

    for index, raw_line in enumerate(lines):
        line = raw_line.strip().strip("| ")
        if not line:
            continue
        season_match = season_line.match(line)
        if season_match:
            current_season = int(season_match.group(1))
            continue

        match = code_line.match(line)
        if match:
            season, episode = int(match.group(1)), int(match.group(2))
            title = (match.group(3) or "").strip()
            if not title:
                # Guide pages commonly put the title directly under the ID.
                for following in lines[index + 1:index + 3]:
                    candidate = following.strip().strip("| -*#")
                    if not candidate:
                        continue
                    if code_line.match(candidate) or season_line.match(candidate):
                        break
                    if re.match(r"^(?:Episode\s+)?\d+\s*[:.)—–-]", candidate, re.I):
                        break
                    if len(candidate) <= 120 and not candidate.startswith(("http://", "https://")):
                        title = candidate
                    break
            add(season, episode, title)
            continue

        # Also support season headings followed by `Episode N: Title` lines.
        episode_match = episode_line.match(line)
        if episode_match:
            add(current_season, int(episode_match.group(1)), episode_match.group(2))

    if len(units) < 2:
        return []
    return _normalize_guide_units(units)


def _normalize_guide_units(units: list[dict]) -> list[dict]:
    """Normalize guide IDs, labels, title fields, and positions."""
    normalized = []
    seen = set()
    for unit in sorted(units, key=lambda item: (int(item.get("season", 1)), int(item.get("episode", item.get("position", 1))))):
        try:
            season = int(unit.get("season", 1))
            episode = int(unit.get("episode", unit.get("position", 1)))
        except (TypeError, ValueError):
            continue
        text = str(unit.get("text", "")).strip()
        title = str(unit.get("title", "")).strip()
        if not title:
            match = re.search(r"Season\s+\d+\s*,?\s*Episode\s+\d+\s*:\s*(.+)", text, re.I)
            title = match.group(1).splitlines()[0].strip() if match else ""
        if not title or re.fullmatch(r"S\d{1,2}E\d{1,3}", title, re.I):
            continue
        key = (season, episode)
        if season < 1 or episode < 1 or key in seen:
            continue
        seen.add(key)
        normalized.append({
            **unit,
            "unit_id": f"s{season:02d}e{episode:02d}",
            "label": f"S{season:02d}E{episode:02d}",
            "season": season,
            "episode": episode,
            "title": title,
            "text": text or f"Season {season}, Episode {episode}: {title}",
        })
    for position, unit in enumerate(normalized, 1):
        unit["position"] = position
    return normalized

def parse_ordered_content(
    text: str,
    work_id: str,
    content_type: str = "show",
) -> list[dict]:
    """Parse ordered units for shows, books, courses, podcasts, or other content.

    The parser recognizes structured tables and numbered headings/lists.
    It preserves source text associated with recognized headings when
    available. It does not invent missing summaries or source content.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    if not isinstance(work_id, str) or not work_id.strip():
        raise ValueError("work_id must be a non-empty string.")

    content_type = (
        content_type.strip().lower()
        if isinstance(content_type, str)
        else ""
    )

    if content_type not in {"show", "book", "course", "podcast", "other"}:
        raise ValueError(
            "content_type must be show, book, course, podcast, or other."
        )

    # Batch transcript imports contain explicit SxxExx headers. Parse these
    # first so incidental "Episode 1" headings inside transcript text cannot
    # collapse multiple transcripts into one unit.
    if content_type in {"show", "podcast"}:
        transcript_header = re.compile(
            r"^\s*(?:#+\s*)?S(\d{1,2})\s*E(\d{1,2})\b(?:\s*[:.)–—-]?\s*.*)?$",
            re.IGNORECASE,
        )
        transcript_units = []
        current = None
        seen_episodes = set()

        for line in text.splitlines():
            header_match = transcript_header.match(line)
            if header_match:
                season = int(header_match.group(1))
                episode = int(header_match.group(2))
                key = (season, episode)
                if season > 0 and episode > 0 and key not in seen_episodes:
                    if current is not None:
                        current["text"] = "\\n".join(current.pop("_body_lines")).strip()
                        transcript_units.append(current)
                    seen_episodes.add(key)
                    label = f"S{season:02d}E{episode:02d}"
                    current = {
                        "work_id": work_id.strip(),
                        "unit_id": label.lower(),
                        "position": len(transcript_units) + 1,
                        "label": label,
                        "season": season,
                        "episode": episode,
                        "_body_lines": [line.strip()],
                    }
                    continue

            if current is not None and line.strip():
                current["_body_lines"].append(line.strip())

        if current is not None:
            current["text"] = "\\n".join(current.pop("_body_lines")).strip()
            transcript_units.append(current)

        if len(transcript_units) >= 2:
            for index, unit in enumerate(transcript_units, start=1):
                unit["position"] = index
            return transcript_units

    # Combined transcript imports use explicit SxxEyy headers. Parse these
    # before generic "Episode 1" headings so episode numbers reset correctly
    # for each season and each transcript remains a separate unit.
    if content_type in {"show", "podcast"}:
        explicit_header = re.compile(
            r"^\s*(?:\[)?S(\d{1,2})\s*E(\d{1,2})(?:\])?\b(?:\s*[-:–—|]\s*(.*))?\s*$",
            re.IGNORECASE,
        )
        explicit_units = []
        current = None
        seen_codes = set()

        def finish_explicit():
            if current is not None:
                current["text"] = "\n".join(current.pop("_body_lines")).strip()
                explicit_units.append(current.copy())

        for line in text.splitlines():
            header = explicit_header.match(line.strip())
            if header:
                season, episode = int(header.group(1)), int(header.group(2))
                key = (season, episode)
                if season < 1 or episode < 1 or key in seen_codes:
                    continue
                finish_explicit()
                seen_codes.add(key)
                label = f"S{season}E{episode}"
                heading_title = (header.group(3) or "").strip()
                prefix = f"{label}: {heading_title}" if heading_title else label
                current = {
                    "work_id": work_id.strip(),
                    "unit_id": f"s{season:02d}e{episode:02d}",
                    "position": len(explicit_units) + 1,
                    "label": label if not heading_title else f"{label}: {heading_title}",
                    "season": season,
                    "episode": episode,
                    "_body_lines": [prefix],
                }
                continue
            if current is not None and line.strip():
                current["_body_lines"].append(line.strip())
        finish_explicit()
        if explicit_units:
            explicit_units.sort(key=lambda unit: (unit["season"], unit["episode"]))
            for index, unit in enumerate(explicit_units, start=1):
                unit["position"] = index
            return explicit_units

    # Shows and podcasts: try episode tables first.
    if content_type in {"show", "podcast"}:
        table_units = parse_episode_table(text, work_id)
        if table_units:
            return table_units

        episode_units = parse_episode_list(text, work_id)
        if episode_units:
            return episode_units

    # Map generic content types to accepted heading names.
    heading_words = {
        "book": r"chapter|part",
        "course": r"lesson|module|unit",
        "other": r"section|part|chapter",
        "show": r"episode",
        "podcast": r"episode",
    }
    heading_pattern = heading_words[content_type]

    heading_re = re.compile(
        rf"^\s*(?:{heading_pattern})\s+(\d+)"
        r"(?:\s*[:.)\-–]\s*|\s+)?(.*?)\s*$",
        re.IGNORECASE,
    )

    # Also support simple numbered lists such as "1. Introduction".
    numbered_re = re.compile(
        r"^\s*(\d+)\s*[.):\-–]\s*(\S.*?)\s*$"
    )

    units = []
    seen_positions = set()
    current_unit = None

    def finish_current() -> None:
        if current_unit is not None:
            current_unit["text"] = "\n".join(
                current_unit.pop("_body_lines")
            ).strip() or current_unit["label"]

            units.append(current_unit.copy())

    for line in text.splitlines():
        heading_match = heading_re.match(line)
        numbered_match = None

        if not heading_match and content_type in {"book", "course", "other"}:
            numbered_match = numbered_re.match(line)

        match = heading_match or numbered_match
        if match:
            number = int(match.group(1))
            title = (
                match.group(2).strip()
                if heading_match and match.lastindex >= 2
                else match.group(2).strip()
            )

            if number < 1 or number in seen_positions:
                continue

            seen_positions.add(number)
            finish_current()

            if heading_match:
                heading_type = re.match(
                    r"^\s*([A-Za-z]+)", line
                ).group(1).title()
            else:
                heading_type = {
                    "book": "Chapter",
                    "course": "Lesson",
                    "other": "Section",
                    "show": "Episode",
                    "podcast": "Episode",
                }[content_type]

            if not title:
                title = f"{heading_type} {number}"

            current_unit = {
                "work_id": work_id.strip(),
                "unit_id": f"{heading_type.lower()}-{number}",
                "position": number,
                "label": f"{heading_type} {number}: {title}",
                "_body_lines": [f"{heading_type} {number}: {title}"],
            }
            continue

        # Preserve content under the current heading for downstream retrieval.
        if current_unit is not None and line.strip():
            current_unit["_body_lines"].append(line.strip())

    
    finish_current()

    
    # Fallback: recognize one or multiple transcript sections.
    if not units and content_type in {"show", "podcast"}:
        transcript_units = []
        current = None
        seen_episodes = set()

        episode_id_re = re.compile(
            r"\bS(\d{1,2})\s*E(\d{1,2})\b",
            re.IGNORECASE,
        )

        for line in text.splitlines():
            match = episode_id_re.search(line)

            if match:
                season = int(match.group(1))
                episode = int(match.group(2))
                key = (season, episode)

                if season > 0 and episode > 0 and key not in seen_episodes:
                    if current is not None:
                        current["text"] = "\n".join(
                            current.pop("_body_lines")
                        ).strip()
                        transcript_units.append(current)

                    seen_episodes.add(key)
                    label = f"S{season:02d}E{episode:02d}"

                    current = {
                        "work_id": work_id.strip(),
                        "unit_id": label.lower(),
                        "position": len(transcript_units) + 1,
                        "label": label,
                        "season": season,
                        "episode": episode,
                        "_body_lines": [line.strip()],
                    }
                    continue

            if current is not None and line.strip():
                current["_body_lines"].append(line.strip())

        if current is not None:
            current["text"] = "\n".join(
                current.pop("_body_lines")
            ).strip()
            transcript_units.append(current)

        if transcript_units:
            units = transcript_units


    # Positions must be sequential for ingestion's validation.
    units.sort(key=lambda unit: unit["position"])

    for index, unit in enumerate(units, start=1):
        unit["position"] = index

    return units


   
def parse_episode_guide(text: str, work_id: str = "episode-guide") -> list[dict]:
    """Parse episode guides with explicit IDs, headings, or season tables."""
    import re

    parsed = []
    seen = set()
    current_season = None

    def add_episode(season, episode, title, content=""):
        if season is None or episode is None:
            return

        season, episode = int(season), int(episode)
        if season < 1 or episode < 1:
            return

        key = (season, episode)
        if key in seen:
            return

        title = re.sub(r"\s+", " ", str(title)).strip(" \t|:-—")
        # Convert Markdown links to their visible text.
        title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title)

        # Remove trailing broadcast dates.
        title = re.sub(
            r"\s*\(\d{1,2}\s+[A-Za-z]+\s+\d{4}\)\s*$",
            "",
            title,
        ).strip()
        if not title or re.fullmatch(
            r"S\d{1,2}E\d{1,3}",
            title,
            re.IGNORECASE,
        ):
            title = f"Season {season}, Episode {episode}"

        seen.add(key)
        parsed.append({
            "season": season,
            "episode": episode,
            "label": f"S{season:02d}E{episode:02d}",
            "title": title,
            "text": content or title,
        })

    lines = text.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Season heading, e.g. "Season 2"
        season_heading = re.search(
            r"\bseason\s+(\d{1,2})\b",
            stripped,
            re.IGNORECASE,
        )
        if season_heading:
            current_season = int(season_heading.group(1))

        # Explicit ID: S02E03 - Episode Title
        match = re.search(
            r"\bS(\d{1,2})\s*E(\d{1,3})\b\s*[-:—|.]?\s*(.*)",
            stripped,
            re.IGNORECASE,
        )
        if match:
            title = match.group(3).strip()
            add_episode(
                match.group(1),
                match.group(2),
                title,
                stripped,
            )
            continue

        # Heading: "Season 2, Episode 3: Episode Title"
        match = re.search(
            r"\bseason\s+(\d{1,2})\s*[,| -]*\s*"
            r"episode\s+(\d{1,3})\s*[:—|-]?\s*(.*)",
            stripped,
            re.IGNORECASE,
        )
        if match:
            add_episode(
                match.group(1),
                match.group(2),
                match.group(3),
                stripped,
            )
            continue

        # Table row: Season | Episode | Title
        cells = [
            cell.strip()
            for cell in stripped.strip("|").split("|")
        ]
        if len(cells) >= 3:
            numbers = re.findall(r"\b\d{1,3}\b", cells[0])
            episodes = re.findall(r"\b\d{1,3}\b", cells[1])

            if numbers and episodes:
                add_episode(
                    numbers[0],
                    episodes[0],
                    cells[2],
                    stripped,
                )
                continue

        # Table row under an existing season heading:
        # Episode | Title, e.g. "3 | The Episode Name"
        if current_season is not None and len(cells) >= 2:
            if re.fullmatch(r"\d{1,3}", cells[0]):
                if cells[1].lower() not in {
                    "title", "episode", "episode title"
                }:
                    add_episode(
                        current_season,
                        cells[0],
                        cells[1],
                        stripped,
                    )

    # Fallback: use the existing parser only if line parsing found nothing.
    if not parsed:
        units = parse_ordered_content(
            text,
            work_id=work_id,
            content_type="show",
        )
        for unit in units:
            label = str(unit.get("label", ""))
            content = str(unit.get("text", ""))
            match = re.search(
                r"\bS(\d{1,2})\s*E(\d{1,3})\b",
                label + " " + content,
                re.IGNORECASE,
            )
            if match:
                title = content.strip().splitlines()[0] if content.strip() else ""
                add_episode(match.group(1), match.group(2), title, content)

    return sorted(
        parsed,
        key=lambda item: (item["season"], item["episode"]),
    )
