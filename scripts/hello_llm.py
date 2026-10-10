import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(encoding="utf-8-sig")
client = OpenAI(api_key=os.environ["LLM_API_KEY"],
                base_url=os.environ["LLM_BASE_URL"])
MODEL = os.environ["LLM_MODEL"]

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user",
                   "content": "In one short sentence, what is a spoiler?"}],
        max_tokens=60,
    )
    print(response.choices[0].message.content)
except Exception as error:
    print(type(error).__name__, ":", str(error)[:300])