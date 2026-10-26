
from catchmeup.content_parser import parse_episode_table


def test_parses_pipe_separated_episode_table():
    text = """
    | Season | Episode | Title |
    | 1 | 1 | Pilot |
    | 1 | 2 | Cat's in the Bag... |
    | 2 | 1 | Seven Thirty-Seven |
    """

    units = parse_episode_table(text, "breaking-bad")

    assert len(units) == 3
    assert units[0]["label"] == "S1E1"
    assert units[1]["label"] == "S1E2"
    assert units[2]["label"] == "S2E1"
    assert [unit["position"] for unit in units] == [1, 2, 3]


def test_rejects_text_without_table_header():
    units = parse_episode_table(
        "Season 1\n1. Pilot\n2. Grilled",
        "breaking-bad",
    )

    assert units == []


def test_skips_duplicate_episode_rows():
    text = """
    | Season | Episode | Title |
    | 1 | 1 | Pilot |
    | 1 | 1 | Pilot |
    """

    units = parse_episode_table(text, "breaking-bad")

    assert len(units) == 1
