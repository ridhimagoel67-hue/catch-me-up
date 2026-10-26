
from pathlib import Path
import httpx

url = "https://en.wikipedia.org/wiki/List_of_Breaking_Bad_episodes"

print("Requesting extracted content...")

response = httpx.post(
    "http://127.0.0.1:8000/extract",
    json={"url": url},
    timeout=60,
)

response.raise_for_status()
data = response.json()

content = data["content"]

output_path = Path("data/processed/breaking_bad_extracted.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(content, encoding="utf-8")

print("\nExtraction successful!")
print("Status:", data.get("status"))
print("Characters:", len(content))
print("Saved to:", output_path)

print("\n========== FIRST 80 LINES ==========\n")

for number, line in enumerate(content.splitlines()[:80], start=1):
    print(f"{number:03}: {line[:250]}")
