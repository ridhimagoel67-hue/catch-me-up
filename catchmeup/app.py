from catchmeup.content_parser import parse_episode_guide
import json
import os
import re
from pathlib import Path

import httpx
import streamlit as st

from catchmeup.collections import COLLECTIONS, CollectionConfig, DATA_DIR


API_BASE_URL = os.getenv("CATCHMEUP_API_URL", "http://127.0.0.1:8000").rstrip("/")
CONTENT_TYPES = {
    "TV show / series": "show",
    "Book": "book",
    "Course": "course",
    "Podcast": "podcast",
    "Other": "other",
}
UNIT_NAMES = {
    "show": "episode",
    "book": "chapter",
    "course": "lesson",
    "podcast": "episode",
    "other": "section",
}

st.set_page_config(
    page_title="Catch Me Up",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Quiet editorial styling: navy, white, and restrained blue accents.
st.markdown(
    """
    <style>
    :root {
        --cmu-bg: #0b1220;
        --cmu-panel: #111c2e;
        --cmu-panel-soft: #17243a;
        --cmu-border: #293850;
        --cmu-text: #f4f7fc;
        --cmu-muted: #aab7ca;
        --cmu-accent: #dbeafe;
        --cmu-blue: #8db8ff;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 50% -18%, rgba(65, 105, 170, .16), transparent 48%),
            var(--cmu-bg);
        color: var(--cmu-text);
        font-family: Inter, "Segoe UI", Arial, sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #0e1829;
        border-right: 1px solid var(--cmu-border);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.5rem; }
    .block-container {
        max-width: 1160px;
        padding-top: 2rem;
        padding-bottom: 3.5rem;
    }
    h1, h2, h3 {
        color: var(--cmu-text);
        letter-spacing: -0.025em;
    }
    h1 { font-size: 2.1rem !important; font-weight: 750 !important; }
    h2 { font-size: 1.42rem !important; font-weight: 680 !important; }
    h3 { font-size: 1.08rem !important; font-weight: 650 !important; }
    p, label, .stMarkdown { color: var(--cmu-text); }
    [data-testid="stCaptionContainer"], .stCaption { color: var(--cmu-muted) !important; }
    .cmu-brand {
        font-size: 2rem;
        line-height: 1.12;
        font-weight: 800;
        letter-spacing: -.055em;
        margin: .2rem 0 .35rem 0;
    }
    .cmu-brand-mark {
        display: inline-block;
        color: var(--cmu-blue);
        margin-right: .45rem;
        font-size: 1.65rem;
        vertical-align: .08em;
    }
    .cmu-subtitle {
        color: var(--cmu-muted);
        font-size: .98rem;
        line-height: 1.55;
        margin-bottom: 1.8rem;
        max-width: 720px;
    }
    .cmu-section-note {
        color: var(--cmu-muted);
        margin-top: -.45rem;
        margin-bottom: 1rem;
        font-size: .92rem;
        line-height: 1.5;
    }
    div[data-testid="stForm"] {
        background: linear-gradient(145deg, rgba(17, 28, 46, .98), rgba(15, 25, 42, .98));
        border: 1px solid var(--cmu-border);
        border-radius: 14px;
        padding: 1.15rem 1.15rem .85rem 1.15rem;
        box-shadow: 0 12px 28px rgba(0, 0, 0, .10);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--cmu-border) !important;
        border-radius: 12px !important;
        background: rgba(17, 28, 46, .72);
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        background: var(--cmu-panel-soft) !important;
        border-color: var(--cmu-border) !important;
        border-radius: 8px !important;
    }
    input, textarea { color: var(--cmu-text) !important; }
    div[data-testid="stButton"] button,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 8px;
        border: 1px solid var(--cmu-accent) !important;
        background: var(--cmu-accent) !important;
        color: #102039 !important;
        font-weight: 700;
        min-height: 2.65rem;
        box-shadow: none !important;
        transition: transform .12s ease, background .12s ease;
    }
    div[data-testid="stButton"] button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #ffffff !important;
        border-color: #ffffff !important;
        transform: translateY(-1px);
    }
    div[data-testid="stButton"] button:focus,
    div[data-testid="stFormSubmitButton"] button:focus {
        box-shadow: 0 0 0 2px var(--cmu-bg), 0 0 0 4px var(--cmu-blue) !important;
    }
    hr { border-color: var(--cmu-border); margin: 1.8rem 0; }
    [data-testid="stExpander"] {
        border: 1px solid var(--cmu-border);
        border-radius: 10px;
        overflow: hidden;
    }
    [data-testid="stMetric"] { background: transparent; }
    a { color: #a9ccff !important; text-underline-offset: 3px; }
    [data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid var(--cmu-border);
    }
    [data-testid="stExpander"] details summary { font-weight: 650; }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--cmu-border);
        border-radius: 9px;
        overflow: hidden;
    }
    [data-testid="stProgress"] > div > div { background: var(--cmu-blue); }
    div[data-testid="stSpinner"] { color: var(--cmu-muted); }
    .stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--cmu-blue) !important; }
    [data-testid="stTextArea"] textarea { line-height: 1.55; }
    @media (max-width: 720px) {
        .block-container { padding: 1.2rem 1rem 2rem; }
        .cmu-brand { font-size: 1.75rem; }
        h1 { font-size: 1.7rem !important; }
        div[data-testid="stForm"] { padding: .9rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { transition: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="cmu-brand"><span class="cmu-brand-mark">↗</span>Catch Me Up</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cmu-subtitle">Pick up where you left off. Ask about what you have reached, without jumping ahead.</div>',
    unsafe_allow_html=True,
)


def api_post(endpoint: str, payload: dict, timeout: float = 45.0) -> dict:
    """Call the FastAPI backend and turn API errors into readable messages."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/{endpoint.lstrip('/')}",
            json=payload,
            timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError("The request timed out. Please try again.") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            "Couldn't connect to the Catch Me Up API. Start the API server and try again."
        ) from exc

    try:
        body = response.json()
    except ValueError:
        body = {}

    if response.status_code >= 400:
        detail = body.get("detail", f"API request failed ({response.status_code}).")
        raise RuntimeError(str(detail))

    if not isinstance(body, dict):
        raise RuntimeError("The API returned an unexpected response.")
    return body


def refresh_collection_registry() -> dict:
    """Load persisted collection metadata, including collections created this session."""
    registry_path = DATA_DIR / "collections.json"
    refreshed = dict(COLLECTIONS)

    if registry_path.is_file():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            registry = {}

        for collection_id in registry:
            try:
                from catchmeup.collections import get_collection
                refreshed[collection_id] = get_collection(collection_id)
            except (KeyError, FileNotFoundError, ValueError):
                continue

    return {
        collection_id: config
        for collection_id, config in refreshed.items()
        if Path(config.data_path).is_file()
    }


@st.cache_data(show_spinner=False)
def load_units(data_path: str) -> list[dict]:
    with Path(data_path).open("r", encoding="utf-8") as file:
        units = json.load(file)

    if not isinstance(units, list) or not units:
        raise ValueError("This collection contains no ordered content units.")

    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("A content unit is not a valid object.")
        if not isinstance(unit.get("position"), int) or not unit.get("label"):
            raise ValueError("Every content unit needs an integer position and a label.")

    positions = [unit["position"] for unit in units]
    if len(positions) != len(set(positions)):
        raise ValueError("Content unit positions must be unique.")

    return sorted(units, key=lambda unit: unit["position"])




def discover_episode_units(title: str, max_results: int = 10) -> tuple[list[dict], str | None]:
    """Find and parse a show's episode guide, returning season/episode titles."""
    candidates = []
    response = api_post(
        "/discover",
        {
            "title": title,
            "content_type": "show",
            "max_results": max_results,
            "search_mode": "episode_guide",
        },
        timeout=180,
    )
    candidates.extend(response.get("results", []))
    candidates.sort(
        key=lambda item: (
            item.get("source_type") == "structured_guide",
            "episode guide" in item.get("title", "").lower(),
            "episode list" in item.get("title", "").lower(),
            item.get("ranking_score", 0),
        ),
        reverse=True,
    )

    for candidate in candidates[:3]:
        try:
            extracted = api_post("/extract", {"url": candidate["url"]}, timeout=180)
            content = extracted.get("content", "").strip()
            # Keep extracted source text out of the main UI; show only the
            # reviewed episode list and source URL to the user.
            if not content:
                continue
            units = parse_episode_guide(content, work_id="episode-guide")
            # Only accept a source if parsing yields plausible episode metadata.
            parsed = []
            for unit in units:
                season = unit.get("season")
                episode = unit.get("episode")
                label = str(unit.get("label", ""))
                if not season or not episode:
                    match = re.search(r"S(\d{1,2})\s*E(\d{1,3})", label, re.I)
                    if match:
                        season, episode = int(match.group(1)), int(match.group(2))
                if not season or not episode:
                    continue
                raw_text = str(unit.get("text", ""))
                episode_title = re.sub(
                    r"^\s*Season\s+\d+\s*,?\s*Episode\s+\d+\s*[:—-]?\s*",
                    "", raw_text, flags=re.I,
                ).splitlines()[0].strip()
                if not episode_title or re.fullmatch(r"S\d{1,2}E\d{1,3}", episode_title, re.I):
                    explicit_title = unit.get("title")
                    episode_title = str(explicit_title).strip() if explicit_title else ""
                if not episode_title or re.fullmatch(r"S\d{1,2}E\d{1,3}", episode_title, re.I):
                    continue
                parsed.append({
                    "season": int(season),
                    "episode": int(episode),
                    "title": episode_title,
                    "label": f"S{int(season):02d}E{int(episode):02d}",
                })
            # Deduplicate season/episode pairs and keep guide order.
            unique = {}
            for unit in parsed:
                unique.setdefault((unit["season"], unit["episode"]), unit)
            parsed = sorted(unique.values(), key=lambda x: (x["season"], x["episode"]))
            if len(parsed) >= 2:
                return parsed, candidate.get("url")
        except (RuntimeError, KeyError, ValueError):
            continue
    return [], None


def find_episode_transcripts(title: str, episode_units: list[dict]):
    """Search by actual episode title, extract candidates, and track missing episodes."""
    episode_results = []
    missing = []
    seen_urls = set()
    seen_episodes = set()
    progress = st.progress(0)
    status = st.empty()
    total = len(episode_units)

    for index, episode in enumerate(episode_units, start=1):
        season = int(episode["season"])
        number = int(episode["episode"])
        episode_title = str(episode["title"]).strip()
        label = episode["label"]
        status.write(f"Finding transcript {index}/{total}: {label} — {episode_title}")

        queries = [
            f'{title} "{episode_title}" transcript',
            f'{title} season {season} episode {number} "{episode_title}" full script transcript',
        ]
        candidates = []
        for query in queries:
            try:
                result = api_post(
                    "/discover",
                    {"title": query, "content_type": "show", "max_results": 10, "search_mode": "transcripts"},
                    timeout=180,
                )
                candidates.extend(result.get("results", []))
            except RuntimeError:
                continue

        unique_candidates = []
        local_urls = set()
        for item in candidates:
            url = str(item.get("url", "")).strip()
            if not url or url in local_urls or url in seen_urls:
                continue
            local_urls.add(url)
            unique_candidates.append(item)

        def score(item):
            text = f'{item.get("title", "")} {item.get("url", "")} {item.get("snippet", "")}'.lower()
            points = 0
            if episode_title and episode_title.lower() in text:
                points += 8
            if label.lower() in re.sub(r"[^a-z0-9]", "", text):
                points += 8
            if f"season {season} episode {number}" in text:
                points += 5
            if "transcript" in text:
                points += 5
            if "script" in text or "dialogue" in text:
                points += 2
            if "recap" in text or "review" in text or "episode guide" in text:
                points -= 5
            return points

        unique_candidates.sort(key=score, reverse=True)
        chosen = None
        for candidate in unique_candidates[:5]:
            try:
                extracted = api_post("/extract", {"url": candidate["url"]}, timeout=180)
                content = str(extracted.get("content", "")).strip()
                if len(content) < 1000:
                    continue
                # If the source clearly names a different episode, don't import it.
                text_to_check = f'{candidate.get("title", "")} {candidate.get("url", "")} {content[:2500]}'.lower()
                other_codes = re.findall(r"\bS(\d{1,2})\s*E(\d{1,3})\b", text_to_check, re.I)
                if other_codes and not any(int(s) == season and int(e) == number for s, e in other_codes):
                    # Transcript hosts sometimes include season/episode navigation;
                    # allow title-matched pages but skip explicit conflicting pages.
                    if episode_title.lower() not in text_to_check:
                        continue
                chosen = {
                    "season": season,
                    "episode": number,
                    "label": label,
                    "title": episode_title,
                    "url": candidate["url"],
                    "content": content,
                }
                seen_urls.add(candidate["url"])
                break
            except (RuntimeError, KeyError):
                continue

        key = (season, number)
        if chosen and key not in seen_episodes:
            episode_results.append(chosen)
            seen_episodes.add(key)
        else:
            missing.append({"label": label, "title": episode_title})
        progress.progress(index / total if total else 1)

    status.empty()
    progress.empty()
    return episode_results, missing



# ---- Discover, extract, and import -------------------------------------------------

st.subheader("Find something to catch up on")
st.markdown('<div class="cmu-section-note">Search for a title, then choose a useful source to import.</div>', unsafe_allow_html=True)
with st.form("discover_form", clear_on_submit=False):
    left, right = st.columns([3, 1])
    with left:
        search_title = st.text_input(
            "Title",
            placeholder="e.g. Breaking Bad, Dune, a Python course...",
            max_chars=200,
        )
    with right:
        content_type_label = st.selectbox("Content type", list(CONTENT_TYPES))
    max_results = st.slider("Sources to find", min_value=3, max_value=10, value=5)
    discover_clicked = st.form_submit_button("Search the web", type="primary", use_container_width=True)

if discover_clicked:
    if not search_title.strip():
        st.warning("Enter a title first.")
    else:
        with st.spinner("Finding structured sources on the web..."):
            try:
                found = api_post(
                    "/discover",
                    {
                        "title": search_title.strip(),
                        "content_type": CONTENT_TYPES[content_type_label],
                        "max_results": max_results,
                    },
                )
                st.session_state["discovery_results"] = found.get("results", [])
                st.session_state["discovery_title"] = search_title.strip()
                st.session_state["discovery_type"] = CONTENT_TYPES[content_type_label]
                st.session_state.pop("extracted_source", None)
                st.session_state.pop("extracted_content", None)
                if not st.session_state["discovery_results"]:
                    st.info("No sources were found. Try another title or content type.")
            except RuntimeError as exc:
                st.error(str(exc))

results = st.session_state.get("discovery_results", [])
if results:
    st.markdown("### Search results")
    st.markdown('<div class="cmu-section-note">Choose the source that best matches the content you want.</div>', unsafe_allow_html=True)
    labels = [
        f'{item.get("title", "Untitled")}  ·  {item.get("source_type", "unknown")}  ·  score {item.get("ranking_score", item.get("score", 0))}'
        for item in results
    ]
    chosen_index = st.selectbox(
        "Discovered sources",
        options=list(range(len(results))),
        format_func=lambda index: labels[index],
        key="chosen_source_index",
    )
    chosen = results[chosen_index]

    with st.container(border=True):
        st.markdown(f"#### {chosen.get('title', 'Untitled')}")
        st.caption(chosen.get("url", ""))
        st.write(chosen.get("snippet", "No preview available."))
        reasons = chosen.get("classification_reasons", [])
        if reasons:
            st.caption("Source assessment: " + " · ".join(reasons))
        else:
            st.caption("Source classification is an estimate; review the page before importing.")

    if st.button("Extract this source", type="primary", key="extract_source"):
        with st.spinner("Extracting source content..."):
            try:
                extracted = api_post("/extract", {"url": chosen["url"]}, timeout=120)
                st.session_state["extracted_source"] = chosen
                st.session_state["extracted_content"] = extracted.get("content", "")
                st.session_state["extraction_meta"] = extracted
            except (RuntimeError, KeyError) as exc:
                st.error(str(exc))

extracted_content = st.session_state.get("extracted_content", "")
extracted_source = st.session_state.get("extracted_source")
if extracted_content and extracted_source:
    st.markdown("### Review extracted content")
    meta = st.session_state.get("extraction_meta", {})
    st.caption(
        f'{meta.get("character_count", len(extracted_content)):,} characters extracted'
        + (" · content was shortened by the provider" if meta.get("truncated") else "")
    )
    with st.expander("Preview extracted text", expanded=False):
        st.text_area("Extracted text", value=extracted_content[:12000], height=220, disabled=True, label_visibility="collapsed")

    st.markdown("### Save to your collections")
    default_name = st.session_state.get("discovery_title", "")
    current_type = st.session_state.get("discovery_type", "show")
    safe_default_id = re.sub(r"[^a-z0-9]+", "-", default_name.lower()).strip("-")[:80]
    with st.form("import_form"):
        import_name = st.text_input("Collection name", value=default_name)
        import_id = st.text_input("Collection ID (lowercase, hyphens)", value=safe_default_id)
        st.caption(f"This source will be imported as ordered {UNIT_NAMES.get(current_type, 'section')} units.")
        import_clicked = st.form_submit_button("Import collection", type="primary")

    if import_clicked:
        if not import_name.strip() or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", import_id.strip()):
            st.error("Enter a collection name and a valid lowercase, hyphen-separated ID.")
        else:
            with st.spinner("Validating and importing ordered content..."):
                try:
                    imported = api_post(
                        "/ingest",
                        {
                            "collection_id": import_id.strip(),
                            "name": import_name.strip(),
                            "source_url": extracted_source["url"],
                            "content": extracted_content,
                            "content_type": current_type,
                        },
                        timeout=100,
                    )
                    st.success(
                        f'Imported **{imported.get("name", import_name)}** '
                        f'with {imported.get("unit_count", 0)} ordered {imported.get("unit_name", "content")} units.'
                    )
                    st.cache_data.clear()
                    st.session_state["selected_collection_id"] = imported["collection_id"]
                    st.session_state.pop("discovery_results", None)
                    st.session_state.pop("extracted_content", None)
                    st.session_state.pop("extracted_source", None)
                    st.rerun()
                except (RuntimeError, KeyError) as exc:
                    st.error(str(exc))




with st.expander("Automatically discover and import episode transcripts", expanded=False):
    st.caption("Discover an episode guide, identify season and episode titles, then search for each transcript. Episodes without a usable transcript are listed after the import attempt.")
    batch_title = st.text_input(
        "Show title",
        value=st.session_state.get("discovery_title", ""),
        key="batch_show_title",
    )
    episode_cap = st.number_input(
        "Maximum episodes to process (safety limit)",
        min_value=1,
        max_value=200,
        value=10,
        help="The episode list is discovered automatically. This cap prevents unexpectedly large imports.",
    )
    batch_id = st.text_input("New collection ID", value="batch-transcripts", key="batch_collection_id")

    if st.button("Discover episodes and import transcripts", type="primary", key="discover_import_batch"):
        if not batch_title.strip():
            st.error("Enter a show title.")
        elif not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", batch_id.strip()):
            st.error("Use lowercase letters, numbers, and hyphens for the collection ID.")
        elif batch_id.strip() in refresh_collection_registry():
            st.error("That collection ID already exists. Choose a new ID to avoid overwriting an existing collection.")
        else:
            try:
                with st.spinner("Discovering the episode guide and episode titles..."):
                    episode_units, guide_url = discover_episode_units(batch_title.strip())
                if not episode_units:
                    st.error("Couldn't parse an episode list from the discovered guides. Try the normal search and select a structured episode guide.")
                else:
                    if len(episode_units) > int(episode_cap):
                        episode_units = episode_units[:int(episode_cap)]
                        st.info(f"Processing the first {int(episode_cap)} episodes from the discovered guide due to the safety limit.")
                    st.success(f"Discovered {len(episode_units)} episodes across {len(set(x['season'] for x in episode_units))} season(s).")
                    if guide_url:
                        st.caption(f"Episode guide source: {guide_url}")
                    with st.expander("Review discovered episode list", expanded=False):
                        st.dataframe(
                            [{"Season": x["season"], "Episode": x["label"], "Title": x["title"]} for x in episode_units],
                            use_container_width=True,
                            hide_index=True,
                        )
                    with st.spinner("Searching for transcript pages by episode title..."):
                        found_episodes, missing_episodes = find_episode_transcripts(batch_title.strip(), episode_units)

                    if not found_episodes:
                        st.error(f"No usable episode transcripts were extracted. {len(missing_episodes)} episodes are missing.")
                    else:
                        # Normalize every block to a unique explicit header; the parser
                        # uses these boundaries to keep each episode separate.
                        combined = "\n\n".join(
                            f"{ep['label']} — {ep['title']}\n\n{ep['content']}"
                            for ep in found_episodes
                        )
                        if len(combined) > 2_000_000:
                            st.error("Combined transcripts exceed the 2 MB import limit. Lower the episode cap and retry.")
                        else:
                            imported = api_post(
                                "/ingest",
                                {
                                    "collection_id": batch_id.strip(),
                                    "name": batch_title.strip(),
                                    "source_url": found_episodes[0]["url"],
                                    "content": combined,
                                    "content_type": "show",
                                },
                                timeout=200,
                            )
                            st.success(f"Imported {imported.get('unit_count', 0)} episode units from {len(found_episodes)} transcript(s).")
                            st.write("**Imported episodes**")
                            st.dataframe(
                                [{"Episode": x["label"], "Title": x["title"], "Source": x["url"]} for x in found_episodes],
                                use_container_width=True,
                                hide_index=True,
                            )
                            if missing_episodes:
                                st.warning(f"No usable transcript found for {len(missing_episodes)} episode(s).")
                                st.write(", ".join(f"{x['label']} — {x['title']}" for x in missing_episodes))
                            st.cache_data.clear()
                            st.session_state["selected_collection_id"] = imported["collection_id"]
                            st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))



# ---- Existing spoiler-aware Q&A ----------------------------------------------------

st.divider()
st.subheader("Your collections")
available_collections = refresh_collection_registry()

if not available_collections:
    st.info("Import a collection above, or check that the backend has a registered dataset.")
    st.stop()

collection_ids = list(available_collections)
preferred_id = st.session_state.get("selected_collection_id")
if preferred_id not in collection_ids:
    preferred_id = collection_ids[0]

with st.sidebar:
    st.markdown("## Your progress")
    collection_id = st.selectbox(
        "Collection",
        options=collection_ids,
        index=collection_ids.index(preferred_id),
        format_func=lambda cid: available_collections[cid].name,
        key="active_collection_id",
    )
    config = available_collections[collection_id]
    try:
        units = load_units(str(config.data_path))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        st.error(f"Could not load collection: {exc}")
        st.stop()

    unit_by_label = {unit["label"]: unit for unit in units}
    labels = list(unit_by_label)
    selected_label = st.selectbox(
        f"I've reached this {config.unit_name}",
        labels,
        index=max(0, len(labels) - 1),
    )
    current_position = unit_by_label[selected_label]["position"]
    st.caption(f"Answers are limited to content through **{selected_label}**.")

st.subheader("Ask a question")
st.markdown('<div class="cmu-section-note">Answers are limited to the progress you selected in the sidebar.</div>', unsafe_allow_html=True)
question = st.text_area(
    "Your question",
    placeholder="What happened, or what do I need a reminder about?",
    max_chars=2000,
    height=100,
    key="question_text",
)
ask_clicked = st.button("Ask question", type="primary", use_container_width=False)

if ask_clicked:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Searching the content you have reached..."):
            try:
                result = api_post(
                    "/ask",
                    {
                        "collection_id": collection_id,
                        "question": question.strip(),
                        "current_position": current_position,
                    },
                    timeout=180,
                )
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

        st.markdown("### Answer")
        st.markdown(result.get("answer", "I couldn't find enough evidence in the available content."))

        status = result.get("status", "UNKNOWN")
        if status in {"OK", "SUCCESS", "ANSWERED"}:
            st.success(f"Workflow status: {status}")
        elif status in {"REFUSED", "INSUFFICIENT_EVIDENCE"}:
            st.info(f"Workflow status: {status}")
        else:
            st.caption(f"Workflow status: {status}")

        with st.expander("Evidence validation and safety details"):
            citation_valid = result.get("citation_valid")
            if citation_valid is True:
                st.success("Citation ID validation passed.")
            elif citation_valid is False:
                st.warning("Citation validation reported an issue.")
            else:
                st.info("Citation validation status was not provided.")
            if result.get("citation_issues"):
                st.write("**Citation issues**")
                for issue in result["citation_issues"]:
                    st.write(f"- {issue}")
            if result.get("refusal_reason"):
                st.write("**Refusal reason:**", result["refusal_reason"])
            st.write("**Spoiler protection reported by API:**", result.get("spoiler_protected", False))
            if result.get("safety_flags"):
                st.write("**Safety flags:**", ", ".join(result["safety_flags"]))
            st.write("**Generation retries:**", result.get("retry_count", 0))

with st.expander("About source quality and spoiler protection"):
    st.caption(
        "Search rankings are estimates. Citation-ID validation does not prove every sentence is supported, "
        "and progress filtering cannot guarantee perfect spoiler detection. Review source quality before relying on an answer."
    )