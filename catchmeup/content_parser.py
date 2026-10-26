
import re


def parse_episode_list(
    text: str,
    work_id: str,
) -> list[dict]:
    """Parse season-grouped, numbered episode lists."""

    if not isinstance(text, str) or not text.strip():
        return []

    if not isinstance(work_id, str) or not work_id.strip():
        raise ValueError("work_id must be a non-empty string")

    season_pattern = re.compile(
        r"^\s*season\s+(\d+)\s*$",
        re.IGNORECASE,
    )
    episode_pattern = re.compile(
        r"^\s*(\d{1,3})[.)]\s+(.+?)\s*$"
    )

    units = []
    current_season = None
    seen_episodes = set()
    position = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()

        season_match = season_pattern.match(line)
        if season_match:
            current_season = int(season_match.group(1))
            continue

        if current_season is None:
            continue

        episode_match = episode_pattern.match(line)
        if not episode_match:
            continue

        episode_number = int(episode_match.group(1))
        episode_title = episode_match.group(2).strip()

        if not episode_title:
            continue

        key = (current_season, episode_number)
        if key in seen_episodes:
            continue

        seen_episodes.add(key)
        position += 1

        units.append(
            {
                "work_id": work_id.strip(),
                "unit_id": (
                    f"s{current_season:02d}e{episode_number:02d}"
                ),
                "position": position,
                "label": f"S{current_season}E{episode_number}",
                "text": (
                    f"Season {current_season}, "
                    f"Episode {episode_number}: {episode_title}"
                ),
            }
        )

    return units



def parse_episode_table(text: str, work_id: str) -> list[dict]:
    """Parse simple episode tables and season-grouped Wikipedia tables."""
    import re

    episodes = []
    seen = set()
    current_season = None
    header = None

    for line in text.splitlines():
        line = line.strip()

        season_match = re.search(r"\bSeason\s+(\d+)\b", line, re.IGNORECASE)
        if season_match:
            current_season = int(season_match.group(1))
            header = None
            continue

        if not line.startswith("|"):
            continue

        columns = [part.strip() for part in line.strip("|").split("|")]
        normalized = [re.sub(r"\s+", " ", col.lower()).strip() for col in columns]

        # Recognize a table header.
        if (
            any("season" in col for col in normalized)
            and any("episode" in col for col in normalized)
            and any(col == "title" for col in normalized)
        ):
            header = normalized
            continue

        if (
            any("no. overall" in col for col in normalized)
            and any("no. in season" in col for col in normalized)
            and any(col == "title" for col in normalized)
        ):
            header = normalized
            continue

        if not header:
            continue

        def find_column(predicate):
            return next(
                (i for i, name in enumerate(header) if predicate(name)),
                None,
            )

        title_idx = find_column(lambda name: name == "title")
        season_idx = find_column(lambda name: "season" in name)
        episode_idx = find_column(
            lambda name: name == "episode"
            or "no. in season" in name
        )
        overall_idx = find_column(lambda name: "no. overall" in name)

        if title_idx is None or episode_idx is None:
            continue
        if max(title_idx, episode_idx) >= len(columns):
            continue

        # Simple tables carry the season in every row.
        if season_idx is not None and season_idx != episode_idx:
            if season_idx >= len(columns) or not columns[season_idx].isdigit():
                continue
            season = int(columns[season_idx])
        else:
            # Wikipedia tables carry the season in the heading.
            season = current_season

        if season is None or not columns[episode_idx].isdigit():
            continue

        episode_number = int(columns[episode_idx])
        raw_title = columns[title_idx].strip()

        if episode_number < 1 or not raw_title:
            continue

        # Strip duplicate Wikipedia link labels from extracted titles.
        raw_title = raw_title.strip('"').strip()
        linked = re.match(r'^(.+?)\s+"[^"]+\s+\([^)]*\).*', raw_title)
        if linked:
            raw_title = linked.group(1).strip()

        title = re.sub(r"\s+", " ", raw_title).strip(" \"'")
        if not title:
            continue

        if overall_idx is not None and overall_idx < len(columns):
            if not columns[overall_idx].isdigit():
                continue
            position = int(columns[overall_idx])
        else:
            position = len(episodes) + 1

        key = (season, episode_number)
        if key in seen:
            continue
        seen.add(key)

        episodes.append({
            "work_id": work_id,
            "unit_id": f"s{season:02d}e{episode_number:02d}",
            "position": position,
            "label": f"S{season}E{episode_number}",
            "text": f"Season {season}, Episode {episode_number}: {title}",
        })

    episodes.sort(key=lambda item: item["position"])
    return episodes



def normalized_index(cells: list[str], name: str) -> int:
    """Find a column by its exact normalized header."""
    return cells.index(name)
