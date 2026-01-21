"""Unit tests for retrieval module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from papers_qa.retrieval import EmbeddingModel, FAISSIndexer, RetrieverPipeline


class TestEmbeddingModel:
    """Tests for EmbeddingModel class."""

    @pytest.fixture
    def embedding_model(self):
        """Create embedding model for tests."""
        return EmbeddingModel(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

    def test_model_initialization(self, embedding_model):
        """Test embedding model initializes correctly."""
        assert embedding_model is not None
        assert embedding_model.embedding_dim > 0

    def test_encode_single_text(self, embedding_model):
        """Test encoding a single text."""
        text = "This is a test sentence."
        embedding = embedding_model.encode(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, embedding_model.embedding_dim)

    def test_encode_multiple_texts(self, embedding_model):
        """Test encoding multiple texts."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedding_model.encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, embedding_model.embedding_dim)

    def test_encode_empty_list(self, embedding_model):
        """Test encoding empty list."""
        embeddings = embedding_model.encode([])
        assert len(embeddings) == 0

    def test_embeddings_are_normalized(self, embedding_model):
        """Test that embeddings have reasonable magnitude."""
        text = "Test sentence for normalization check."
        embedding = embedding_model.encode(text)

        norm = np.linalg.norm(embedding[0])
        assert 0.5 < norm < 2.0


class TestFAISSIndexer:
    """Tests for FAISSIndexer class."""

    @pytest.fixture
    def indexer(self):
        """Create FAISS indexer for tests."""
        return FAISSIndexer(embedding_dim=384, index_type="flat")

    def test_indexer_creation_flat(self):
        """Test flat index creation."""
        indexer = FAISSIndexer(embedding_dim=128, index_type="flat")
        assert indexer is not None
        assert indexer.embedding_dim == 128

    def test_indexer_creation_ivf(self):
        """Test IVF index creation."""
        indexer = FAISSIndexer(embedding_dim=128, index_type="ivf")
        assert indexer is not None

    def test_invalid_index_type(self):
        """Test invalid index type raises error."""
        with pytest.raises(ValueError):
            FAISSIndexer(embedding_dim=128, index_type="invalid")

    def test_add_documents(self, indexer):
        """Test adding documents to index."""
        embeddings = np.random.randn(5, 384).astype(np.float32)
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        indexer.add(embeddings, documents)

        assert len(indexer.documents) == 5

    def test_search_basic(self, indexer):
        """Test basic search functionality."""
        embeddings = np.random.randn(10, 384).astype(np.float32)
        documents = [f"Document {i}" for i in range(10)]

        indexer.add(embeddings, documents)

        query = embeddings[0]
        results, scores = indexer.search(query, k=3)

        assert len(results) == 3
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_search_returns_correct_k(self, indexer):
        """Test search returns correct number of results."""
        embeddings = np.random.randn(5, 384).astype(np.float32)
        documents = [f"Doc {i}" for i in range(5)]

        indexer.add(embeddings, documents)

        query = np.random.randn(384).astype(np.float32)

        for k in [1, 3, 5]:
            results, _ = indexer.search(query, k=k)
            assert len(results) == k

    def test_save_and_load(self, indexer):
        """Test saving and loading index."""
        embeddings = np.random.randn(5, 384).astype(np.float32)
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        indexer.add(embeddings, documents)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_index"

            indexer.save(save_path)

            new_indexer = FAISSIndexer(embedding_dim=384, index_type="flat")
            new_indexer.load(save_path)

            assert len(new_indexer.documents) == 5
            assert new_indexer.documents == documents


class TestRetrieverPipeline:
    """Tests for RetrieverPipeline class."""

    @pytest.fixture
    def retriever(self):
        """Create retriever pipeline for tests."""
        return RetrieverPipeline(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

    def test_pipeline_initialization(self, retriever):
        """Test pipeline initializes correctly."""
        assert retriever is not None
        assert retriever.embedding_model is not None
        assert retriever.indexer is not None

    def test_index_and_retrieve(self, retriever):
        """Test indexing and retrieval workflow."""
        documents = [
            "The cat sat on the mat.",
            "Dogs are loyal companions.",
            "Machine learning is a branch of AI.",
            "Python is a programming language.",
            "Medical research advances healthcare.",
        ]

        retriever.index_documents(documents)

        results = retriever.retrieve("What animals make good pets?", k=2)

        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

    def test_retrieve_returns_scores(self, retriever):
        """Test that retrieval returns valid scores."""
        documents = ["Document about topic A.", "Document about topic B."]

        retriever.index_documents(documents)

        results = retriever.retrieve("topic A", k=2)

        for doc, score in results:
            assert isinstance(doc, str)
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_save_and_load_pipeline(self, retriever):
        """Test saving and loading the complete pipeline."""
        documents = ["Test document 1.", "Test document 2."]

        retriever.index_documents(documents)

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever.cache_dir = Path(tmpdir)
            retriever.save()

            new_retriever = RetrieverPipeline(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=Path(tmpdir),
            )
            new_retriever.load()

            assert len(new_retriever.indexer.documents) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
