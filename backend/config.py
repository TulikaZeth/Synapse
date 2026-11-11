import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project metadata
    PROJECT_NAME: str = "GraphRAG"
    API_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    # Storage paths
    DATA_DIR: str = "./data"
    UPLOAD_DIR: str = "./data/uploads"
    DATABASE_URL: str = "sqlite:///./data/graphrag.db"

    # ChromaDB Vector Store
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHROMA_COLLECTION_NAME: str = "graphrag_chunks"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Graph Persistence (Neo4j with Embedded Fallback)
    GRAPH_PERSIST_FILE: str = "./data/embedded_graph.json"
    NEO4J_ENABLED: bool = False
    NEO4J_URI: Optional[str] = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # LLM Settings
    GEMINI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gemini-1.5-flash"

    # Ingestion constraints
    MAX_UPLOAD_SIZE_MB: int = 50
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    def ensure_directories(self):
        """Ensure runtime directories exist."""
        for path_str in [self.DATA_DIR, self.UPLOAD_DIR, self.CHROMA_PERSIST_DIR]:
            p = Path(path_str)
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()

