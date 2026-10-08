import re
import time

import requests

HEADERS = {"User-Agent": "catch-me-up-student-project/0.1"}
# Best name first. A page often has BOTH a short "Synopsis" blurb and a long "Recap".
RECAP_PRIORITY = ["recap", "plot", "summary", "synopsis"]


def api_get(wiki, params):
    """Ask a Fandom wiki's API a question and return the answer as a dict."""
    url = f"https://{wiki}.fandom.com/api.php"
    params = {**params, "format": "json"}
    response = requests.get(url, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(1)  # be polite: wait 1 second between requests
    return response.json()


def clean_wikitext(text):
    """Remove wiki markup so only readable sentences remain."""
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[(?:File|Image):.*?\]\]", "", text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"^=+.*?=+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pick_section(sections):
    """From a page's section list, return the index of the best recap-like one."""
    found = {}
    for section in sections:
        heading = re.sub(r"<[^>]+>", "", section["line"]).strip().lower()
        found.setdefault(heading, section["index"])
    for name in RECAP_PRIORITY:
        if name in found:
            return found[name]
    return None


def fetch_recap(wiki, title):
    """Return the cleaned recap of one episode page, or None if not found."""
    data = api_get(wiki, {"action": "parse", "page": title,
                          "prop": "sections", "redirects": 1})
    if "error" in data:
        return None

    index = pick_section(data["parse"]["sections"])
    if index is None:
        return None

    data = api_get(wiki, {"action": "parse", "page": title,
                          "prop": "wikitext", "section": index,
                          "redirects": 1})
    if "error" in data:
        return None
    return clean_wikitext(data["parse"]["wikitext"]["*"])