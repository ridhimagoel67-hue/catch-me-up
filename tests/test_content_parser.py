
from catchmeup.content_parser import parse_episode_list


def test_parses_episode_list_in_order():
    text = """
    Season 1
    1. Pilot
    2. Cat's in the Bag...
    3. ...And the Bag's in the River

    Season 2
    1. Seven Thirty-Seven
    2. Grilled
    """

    units = parse_episode_list(
        text=text,
        work_id="breaking-bad",
    )

    assert len(units) == 5

    assert units[0]["position"] == 1
    assert units[0]["label"] == "S1E1"
    assert units[0]["text"].find("Pilot") >= 0

    assert units[1]["position"] == 2
    assert units[1]["label"] == "S1E2"

    assert units[3]["position"] == 4
    assert units[3]["label"] == "S2E1"


def test_empty_content_returns_no_units():
    units = parse_episode_list(
        text="",
        work_id="breaking-bad",
    )

    assert units == []


def test_unstructured_content_returns_no_units():
    units = parse_episode_list(
        text="This page contains general information about television.",
        work_id="breaking-bad",
    )

    assert units == []
