import sys

from catchmeup.adapters.fandom import api_get

WIKI = "gossipgirl"
query = " ".join(sys.argv[1:])

data = api_get(WIKI, {"action": "query", "list": "search",
                      "srsearch": query, "srlimit": 5})
for hit in data["query"]["search"]:
    print(hit["title"])