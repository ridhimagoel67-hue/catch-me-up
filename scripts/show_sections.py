import re
import sys

from catchmeup.adapters.fandom import api_get

WIKI = "gossipgirl"
title = " ".join(sys.argv[1:])

data = api_get(WIKI, {"action": "parse", "page": title,
                      "prop": "sections", "redirects": 1})
if "error" in data:
    print("ERROR:", data["error"].get("info"))
else:
    for s in data["parse"]["sections"]:
        print(s["index"], re.sub(r"<[^>]+>", "", s["line"]))