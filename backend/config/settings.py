from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    chroma_persist_dir: str = "data/chroma_db"
    knowledge_dir: str = "data/plant_knowledge"
    log_dir: str = "data/logs"

    rag_top_k: int = 5
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 50

    tavily_api_key: str = ""

    vision_max_tokens: int = 1024
    diagnosis_max_tokens: int = 2048

    # API / deployment
    cors_origins: str = "*"
    max_upload_mb: float = 10.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
