"""Embedding and retrieval module."""

from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from papers_qa.config import get_settings
from papers_qa.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """Wrapper for embedding models."""

    def __init__(self, model_name: str = "", device: str = "cpu") -> None:
        """Initialize embedding model.

        Args:
            model_name: Model identifier from Hugging Face.
            device: Device to use (cpu, cuda, mps).
        """
        settings = get_settings()
        self.model_name = model_name or settings.model.embedding_model
        self.device = device or settings.model.embedding_device

        logger.info(
            "loading_embedding_model",
            model=self.model_name,
            device=self.device,
        )

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info("embedding_model_loaded", dimension=self.embedding_dim)

    def encode(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: Single text or list of texts.
            batch_size: Batch size for encoding.

        Returns:
            np.ndarray: Embeddings array.
        """
        if isinstance(texts, str):
            texts = [texts]

        logger.debug("encoding_texts", count=len(texts), batch_size=batch_size)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        if isinstance(embeddings, np.ndarray):
            return embeddings

        return np.array(embeddings)


class FAISSIndexer:
    """FAISS-based vector index for retrieval."""

    def __init__(self, embedding_dim: int, index_type: str = "flat") -> None:
        """Initialize FAISS indexer.

        Args:
            embedding_dim: Dimension of embeddings.
            index_type: Type of FAISS index (flat, ivf).
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.documents: list[str] = []
        self.index = self._create_index()

        logger.info(
            "faiss_index_created",
            dimension=embedding_dim,
            type=index_type,
        )

    def _create_index(self) -> Any:
        """Create FAISS index.

        Returns:
            faiss.Index: The created index.
        """
        if self.index_type in ("flat", "faiss_flat"):
            return faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type in ("ivf", "faiss_ivf"):
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            return faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def add(self, embeddings: np.ndarray, documents: list[str]) -> None:
        """Add embeddings and documents to index.

        Args:
            embeddings: Document embeddings.
            documents: Document texts.
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.documents.extend(documents)

        logger.info(
            "documents_added_to_index",
            count=len(documents),
            total=len(self.documents),
        )

    def search(self, embedding: np.ndarray, k: int = 5) -> tuple[list[str], list[float]]:
        """Search for similar documents.

        Args:
            embedding: Query embedding.
            k: Number of results to return.

        Returns:
            tuple: (list of documents, list of distances)
        """
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        embedding = embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(embedding, k)

        results = []
        scores = []

        for idx, distance in zip(indices[0], distances[0], strict=False):
            if 0 <= idx < len(self.documents):
                results.append(self.documents[int(idx)])
                # Convert L2 distance to similarity (0-1 range)
                scores.append(float(1.0 / (1.0 + distance)))

        return results, scores

    def save(self, path: Path) -> None:
        """Save index to disk.

        Args:
            path: Path to save index.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))

        # Save documents
        import json

        with open(path / "documents.json", "w") as f:
            json.dump(self.documents, f)

        logger.info("index_saved", path=str(path))

    def load(self, path: Path) -> None:
        """Load index from disk.

        Args:
            path: Path to load index from.
        """
        path = Path(path)

        self.index = faiss.read_index(str(path / "index.faiss"))

        import json

        with open(path / "documents.json", "r") as f:
            self.documents = json.load(f)

        logger.info("index_loaded", path=str(path), documents=len(self.documents))


class RetrieverPipeline:
    """Complete retrieval pipeline."""

    def __init__(self, model_name: str = "", cache_dir: Path | None = None) -> None:
        """Initialize retriever.

        Args:
            model_name: Embedding model name.
            cache_dir: Directory for caching.
        """
        settings = get_settings()
        self.model_name = model_name or settings.model.embedding_model
        self.cache_dir = cache_dir or settings.data.cache_dir

        self.embedding_model = EmbeddingModel(self.model_name)
        self.indexer = FAISSIndexer(
            self.embedding_model.embedding_dim,
            index_type=settings.retrieval.index_type,
        )

    def index_documents(self, documents: list[str]) -> None:
        """Index documents.

        Args:
            documents: List of documents to index.
        """
        logger.info("indexing_documents", count=len(documents))

        embeddings = self.embedding_model.encode(
            documents,
            batch_size=32,
        )

        self.indexer.add(embeddings, documents)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Retrieve similar documents for a query.

        Args:
            query: Query text.
            k: Number of results.

        Returns:
            list: List of (document, score) tuples.
        """
        query_embedding = self.embedding_model.encode(query)

        documents, scores = self.indexer.search(query_embedding[0], k=k)

        return list(zip(documents, scores, strict=False))

    def save(self) -> None:
        """Save retriever state."""
        self.indexer.save(self.cache_dir / "retriever_index")

    def load(self) -> None:
        """Load retriever state."""
        self.indexer.load(self.cache_dir / "retriever_index")
