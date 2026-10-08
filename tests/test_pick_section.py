from catchmeup.adapters.fandom import pick_section


def test_prefers_recap_over_synopsis():
    sections = [{"line": "Synopsis", "index": "1"},
                {"line": "Recap", "index": "2"}]
    assert pick_section(sections) == "2"


def test_falls_back_to_synopsis_when_no_recap():
    assert pick_section([{"line": "Synopsis", "index": "1"}]) == "1"


def test_returns_none_when_nothing_matches():
    assert pick_section([{"line": "Trivia", "index": "1"}]) is None