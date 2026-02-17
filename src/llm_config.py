import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # Loads from .env if available

llm = ChatOpenAI(
    model="gpt-4.1-mini",  # or "gpt-4"
    temperature=0.2,
    max_tokens=500,
    api_key=os.getenv("OPENAI_API_KEY")  # Optional: explicit or rely on environment
)

# VLM for vision tasks (like reflector node)
vlm = ChatOpenAI(
    model="gpt-4o",  # GPT-4o supports vision
    temperature=0.2,
    max_tokens=1000,  # More tokens for detailed visual analysis
    api_key=os.getenv("OPENAI_API_KEY")
)
