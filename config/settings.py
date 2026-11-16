import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables
    or .env file with resilient defaults.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM Settings
    llm_provider: str = Field(default="local_mock", description="local_mock, ollama, or openai_compatible")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:latest")
    openai_api_base: str = Field(default="https://api.openai.com/v1")
    openai_api_key: str = Field(default="")

    # Level 1 Feedback (Retrieval & Query) Settings
    relevance_threshold: float = Field(default=0.65, description="Min relevance score for doc grading")
    max_retrieval_retries: int = Field(default=3, description="Max query rewrite and retrieval attempts")
    top_k_documents: int = Field(default=4, description="Number of documents to retrieve initially")

    # Level 2 Feedback (Generation & Grounding) Settings
    groundedness_threshold: float = Field(default=0.75, description="Min faithfulness score to avoid hallucination")
    completeness_threshold: float = Field(default=0.70, description="Min completeness score to ensure query was answered")
    max_generation_retries: int = Field(default=2, description="Max generation critique revision attempts")

    # Level 3 Feedback (System & Memory) Settings
    database_path: str = Field(default="feedback_store.db", description="SQLite database path for feedback audit log")
    enable_golden_cache: bool = Field(default=True, description="Enable instant retrieval for human-verified Q&A")
    similarity_cache_threshold: float = Field(default=0.92, description="Cosine threshold for golden cache match")
    dynamic_weight_increment: float = Field(default=0.10, description="Delta applied to doc weight on feedback")

    # Server Settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

settings = Settings()

