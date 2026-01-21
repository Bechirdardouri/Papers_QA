"""REST API module for Papers QA system.

This module provides a FastAPI-based REST API for the Papers QA system,
enabling HTTP-based access to QA generation, retrieval, and evaluation.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from papers_qa.config import get_settings
from papers_qa.data import DataLoader, DataProcessor
from papers_qa.evaluation import QAEvaluator
from papers_qa.logging_config import configure_logging, get_logger
from papers_qa.retrieval import RetrieverPipeline

logger = get_logger(__name__)


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    question: str = Field(..., min_length=3, description="Question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class QueryResult(BaseModel):
    """Single query result."""

    document: str = Field(..., description="Retrieved document text")
    score: float = Field(..., description="Similarity score")
    rank: int = Field(..., description="Result rank")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    question: str
    results: list[QueryResult]
    total_results: int


class IndexRequest(BaseModel):
    """Request model for indexing documents."""

    documents: list[str] = Field(..., min_length=1, description="Documents to index")


class IndexResponse(BaseModel):
    """Response model for index endpoint."""

    indexed_count: int
    message: str


class EvaluateRequest(BaseModel):
    """Request model for evaluation endpoint."""

    reference: str = Field(..., description="Reference answer")
    hypothesis: str = Field(..., description="Generated answer")


class EvaluateResponse(BaseModel):
    """Response model for evaluation endpoint."""

    bleu: float
    rouge1_f1: float
    rougeL_f1: float
    semantic_similarity: float
    overall_score: float


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    environment: str
    index_loaded: bool
    document_count: int


class AppState:
    """Application state container."""

    def __init__(self) -> None:
        self.retriever: RetrieverPipeline | None = None
        self.evaluator: QAEvaluator | None = None
        self.data_processor: DataProcessor | None = None
        self.settings = get_settings()


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    configure_logging()
    logger.info("starting_api_server")

    app_state.data_processor = DataProcessor()
    app_state.settings = get_settings()

    index_path = app_state.settings.data.cache_dir / "retriever_index"
    if index_path.exists():
        try:
            app_state.retriever = RetrieverPipeline()
            app_state.retriever.load()
            logger.info("index_loaded_at_startup", path=str(index_path))
        except Exception as e:
            logger.warning("failed_to_load_index_at_startup", error=str(e))
            app_state.retriever = RetrieverPipeline()
    else:
        app_state.retriever = RetrieverPipeline()
        logger.info("no_existing_index_found")

    yield

    logger.info("shutting_down_api_server")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    from papers_qa import __version__

    app = FastAPI(
        title="Papers QA API",
        description="REST API for Medical Paper Question Answering System",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check() -> HealthResponse:
        """Check API health status."""
        from papers_qa import __version__

        doc_count = 0
        index_loaded = False

        if app_state.retriever and app_state.retriever.indexer:
            doc_count = len(app_state.retriever.indexer.documents)
            index_loaded = doc_count > 0

        return HealthResponse(
            status="healthy",
            version=__version__,
            environment=app_state.settings.environment,
            index_loaded=index_loaded,
            document_count=doc_count,
        )

    @app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
    async def query_documents(request: QueryRequest) -> QueryResponse:
        """Query the document index for relevant passages."""
        if not app_state.retriever:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Retriever not initialized",
            )

        if len(app_state.retriever.indexer.documents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents indexed. Please index documents first.",
            )

        try:
            results = app_state.retriever.retrieve(request.question, k=request.top_k)

            query_results = [
                QueryResult(document=doc, score=score, rank=i + 1)
                for i, (doc, score) in enumerate(results)
            ]

            return QueryResponse(
                question=request.question,
                results=query_results,
                total_results=len(query_results),
            )

        except Exception as e:
            logger.error("query_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query failed: {str(e)}",
            )

    @app.post("/index", response_model=IndexResponse, tags=["Indexing"])
    async def index_documents(request: IndexRequest) -> IndexResponse:
        """Index documents for retrieval."""
        if not app_state.retriever:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Retriever not initialized",
            )

        try:
            cleaned_docs = [
                app_state.data_processor.clean_text(doc) for doc in request.documents
            ]
            cleaned_docs = [doc for doc in cleaned_docs if doc]

            if not cleaned_docs:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid documents to index after cleaning",
                )

            app_state.retriever.index_documents(cleaned_docs)
            app_state.retriever.save()

            return IndexResponse(
                indexed_count=len(cleaned_docs),
                message=f"Successfully indexed {len(cleaned_docs)} documents",
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("indexing_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Indexing failed: {str(e)}",
            )

    @app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
    async def evaluate_answer(request: EvaluateRequest) -> EvaluateResponse:
        """Evaluate a generated answer against a reference."""
        if app_state.evaluator is None:
            try:
                app_state.evaluator = QAEvaluator()
            except Exception as e:
                logger.error("evaluator_initialization_failed", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Evaluator initialization failed",
                )

        try:
            metrics = app_state.evaluator.evaluate_answer(
                reference=request.reference,
                hypothesis=request.hypothesis,
            )

            return EvaluateResponse(
                bleu=metrics.get("bleu", 0.0),
                rouge1_f1=metrics.get("rouge1_f1", 0.0),
                rougeL_f1=metrics.get("rougeL_f1", 0.0),
                semantic_similarity=metrics.get("semantic_similarity", 0.0),
                overall_score=metrics.get("overall_score", 0.0),
            )

        except Exception as e:
            logger.error("evaluation_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Evaluation failed: {str(e)}",
            )

    @app.delete("/index", tags=["Indexing"])
    async def clear_index() -> dict[str, str]:
        """Clear the document index."""
        if not app_state.retriever:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Retriever not initialized",
            )

        try:
            app_state.retriever = RetrieverPipeline()
            return {"message": "Index cleared successfully"}

        except Exception as e:
            logger.error("clear_index_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to clear index: {str(e)}",
            )

    return app


app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the API server.

    Args:
        host: Host address to bind to.
        port: Port number to listen on.
        reload: Enable auto-reload for development.
    """
    import uvicorn

    uvicorn.run(
        "papers_qa.api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_server()
