import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.config import settings
from backend.database import init_db, get_db
from backend.services.vector_store import vector_store
from backend.services.graph_store import graph_store
from backend.api.documents import router as documents_router
from backend.api.graph import router as graph_router
from backend.api.query import router as query_router
from backend.api.evaluation import router as evaluation_router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("graphrag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for backend startup and shutdown."""
    logger.info("Initializing GraphRAG storage and database foundations...")
    # 1. Initialize relational DB tables
    init_db()
    logger.info("Relational database schema verified.")

    # 2. Initialize ChromaDB vector store
    try:
        vector_store.initialize()
    except Exception as e:
        logger.warning("ChromaDB initialization notice: %s", e)

    # 3. Check graph store status
    stats = graph_store.get_stats()
    logger.info("Graph Store active backend: %s (status: %s)", stats.get("backend"), stats.get("status"))

    yield
    logger.info("GraphRAG backend shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Knowledge Graph-Enhanced Retrieval-Augmented Generation Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
@app.get(f"{settings.API_PREFIX}/health", tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """
    Comprehensive system health check verifying:
    - FastAPI server
    - Relational database connectivity (SQLite / PostgreSQL)
    - ChromaDB dense vector store
    - Neo4j / Embedded Knowledge Graph store
    """
    # 1. Check DB
    db_status = "healthy"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    # 2. Check Vector Store
    vector_stats = vector_store.get_stats()

    # 3. Check Graph Store
    graph_stats = graph_store.get_stats()

    is_healthy = (
        db_status == "healthy"
        and vector_stats.get("status") == "healthy"
        and graph_stats.get("status") == "healthy"
    )

    return {
        "status": "healthy" if is_healthy else "degraded",
        "service": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "components": {
            "database": {
                "status": db_status,
                "url": settings.DATABASE_URL.split("@")[-1]  # sanitize passwords
            },
            "vector_store": vector_stats,
            "graph_store": graph_stats
        }
    }


# Include API routers
app.include_router(documents_router, prefix=settings.API_PREFIX)
app.include_router(graph_router, prefix=settings.API_PREFIX)
app.include_router(query_router, prefix=settings.API_PREFIX)
app.include_router(evaluation_router, prefix=settings.API_PREFIX)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)

