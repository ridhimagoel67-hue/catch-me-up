
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load configuration from .env
load_dotenv(encoding="utf-8-sig")

# Connect LangChain to the same LLM provider
llm = ChatOpenAI(
    model=os.environ["LLM_MODEL"],
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
    max_tokens=60,
)

# Create a reusable prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for a spoiler-safe TV recap app."),
    ("human", "{question}"),
])

# Combine the prompt and model into a chain
chain = prompt | llm

# Run the chain
response = chain.invoke({
    "question": "Explain what a spoiler is in one short sentence."
})

print(response.content)
