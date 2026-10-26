
from catchmeup.discovery import classify_source, rank_source


def test_wikipedia_episode_list_is_structured_guide():
    source_type, _ = classify_source(
        title="List of Breaking Bad episodes - Wikipedia",
        url="https://en.wikipedia.org/wiki/List_of_Breaking_Bad_episodes",
        snippet="Episodes Season 1 Season 2 Episode number",
        content_type="show",
    )

    assert source_type == "structured_guide"


def test_episode_summaries_page_is_structured_guide():
    source_type, _ = classify_source(
        title="Summaries from Season 1 - Studying Breaking Bad",
        url=(
            "https://studyingbreakingbadcourse.wordpress.com/"
            "episoderesources/episode-summaries/season-1"
        ),
        snippet=(
            "Episode 101: Pilot. Episode 103: "
            "And the Bag's in the River. Chapter 1."
        ),
        content_type="show",
    )

    assert source_type == "structured_guide"


def test_recap_is_not_classified_as_structured_guide():
    source_type, _ = classify_source(
        title="Breaking Bad Episode Recaps & News",
        url="https://www.vulture.com/tv/breaking-bad",
        snippet="Episode list and season summaries",
        content_type="show",
    )

    assert source_type == "recap_discussion"


def test_book_chapter_list_is_structured_guide():
    source_type, _ = classify_source(
        title="Harry Potter - Table of Contents",
        url="https://example.org/harry-potter",
        snippet="Chapter 1, Chapter 2, Chapter 3",
        content_type="book",
    )

    assert source_type == "structured_guide"


def test_course_curriculum_is_structured_guide():
    source_type, _ = classify_source(
        title="Machine Learning Course Curriculum",
        url="https://example.org/ml-course",
        snippet="Modules and lessons",
        content_type="course",
    )

    assert source_type == "structured_guide"


def test_recap_penalty_is_applied():
    score, reasons = rank_source(
        title="Breaking Bad Episode Recap",
        url="https://example.org/recap",
        snippet="Episode discussion",
        tavily_score=0.8,
        content_type="show",
    )

    assert score < 0.8
    assert any("penalty" in reason for reason in reasons)


def test_ranking_score_uses_original_relevance():
    score, _ = rank_source(
        title="Unrelated page",
        url="https://example.org/page",
        snippet="General information",
        tavily_score=0.6,
        content_type="show",
    )

    assert score == 0.6
