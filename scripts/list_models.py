import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(encoding="utf-8-sig")
client = OpenAI(api_key=os.environ["LLM_API_KEY"],
                base_url=os.environ["LLM_BASE_URL"])

try:
    for model in client.models.list():
        print(model.id)
except Exception as error:
    print(type(error).__name__, ":", str(error)[:300])