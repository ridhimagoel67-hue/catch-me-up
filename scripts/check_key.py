import os

from dotenv import load_dotenv

load_dotenv(encoding="utf-8-sig")
key = os.environ.get("GEMINI_API_KEY", "")

if not key:
    print("No key found. Check that .env is in the project folder and has "
          "a line GEMINI_API_KEY=...")
else:
    print(f"Key found, {len(key)} characters long.")
    if " " in key or '"' in key or "'" in key:
        print("Warning: the key contains spaces or quotes. Remove them in .env.")