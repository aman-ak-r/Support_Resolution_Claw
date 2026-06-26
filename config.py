import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
FAISS_INDEX_PATH = BASE_DIR / "faiss_index"
DB_PATH = BASE_DIR / "logs" / "support_claw.db"

KB_DIR.mkdir(exist_ok=True)
FAISS_INDEX_PATH.mkdir(exist_ok=True)
DB_PATH.parent.mkdir(exist_ok=True)

# Embedding model used for FAISS vector search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# Confidence threshold for routing: score >= 3.0 → answer, else → escalate
# score = (llm_confidence * 0.7) + (faiss_similarity * 5.0 * 0.3)
CONFIDENCE_THRESHOLD = 3.0
LLM_CONFIDENCE_WEIGHT = 0.7
SIMILARITY_WEIGHT = 0.3

# Supported providers: "mock", "openai", "gemini", "anthropic"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
