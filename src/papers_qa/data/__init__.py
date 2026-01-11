"""Data loading and processing utilities."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from papers_qa.logging_config import get_logger

logger = get_logger(__name__)


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, path: Path) -> dict[str, Any]:
        """Load a document from path.

        Args:
            path: Path to the document.

        Returns:
            dict: Loaded document data.
        """
        pass


class JSONDocumentLoader(DocumentLoader):
    """Load JSON documents."""

    def load(self, path: Path) -> dict[str, Any]:
        """Load a JSON document.

        Args:
            path: Path to the JSON file.

        Returns:
            dict: Parsed JSON content.

        Raises:
            json.JSONDecodeError: If JSON parsing fails.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class CSVDocumentLoader(DocumentLoader):
    """Load CSV documents."""

    def load(self, path: Path) -> dict[str, Any]:
        """Load a CSV document.

        Args:
            path: Path to the CSV file.

        Returns:
            dict: First row of CSV as dictionary.
        """
        df = pd.read_csv(path)
        return df.to_dict(orient="records")


class DataLoader:
    """Main data loader for the system."""

    def __init__(self) -> None:
        """Initialize the data loader."""
        self.loaders: dict[str, DocumentLoader] = {
            ".json": JSONDocumentLoader(),
            ".csv": CSVDocumentLoader(),
        }

    def load_documents(self, directory: Path) -> list[dict[str, Any]]:
        """Load all documents from a directory.

        Args:
            directory: Directory containing documents.

        Returns:
            list: List of loaded documents.
        """
        directory = Path(directory)
        documents = []

        if not directory.exists():
            logger.warning("directory_not_found", directory=str(directory))
            return documents

        files = sorted(directory.glob("*"))
        logger.info("loading_documents", directory=str(directory), file_count=len(files))

        for file_path in tqdm(files, desc="Loading documents", disable=False):
            if file_path.suffix not in self.loaders:
                logger.debug("skipping_unsupported_format", file=file_path.name)
                continue

            try:
                loader = self.loaders[file_path.suffix]
                doc = loader.load(file_path)

                if isinstance(doc, list):
                    documents.extend(doc)
                else:
                    documents.append(doc)

                logger.debug("document_loaded", file=file_path.name)
            except Exception as e:
                logger.error("failed_to_load_document", file=file_path.name, error=str(e))

        logger.info("documents_loaded", total_count=len(documents))
        return documents

    def load_qa_dataset(self, path: Path) -> pd.DataFrame:
        """Load QA dataset from CSV.

        Args:
            path: Path to CSV file.

        Returns:
            pd.DataFrame: Loaded QA dataset.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("qa_file_not_found", path=str(path))
            return pd.DataFrame()

        df = pd.read_csv(path)
        logger.info("qa_dataset_loaded", rows=len(df), columns=list(df.columns))
        return df


class DataProcessor:
    """Process and validate data."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Raw text to clean.

        Returns:
            str: Cleaned text.
        """
        if not isinstance(text, str):
            return ""

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove control characters
        text = "".join(c for c in text if c.isprintable() or c.isspace())

        return text.strip()

    @staticmethod
    def extract_text_from_doc(doc: dict[str, Any]) -> str:
        """Extract text from document structure.

        Args:
            doc: Document dictionary.

        Returns:
            str: Extracted text.
        """
        text_parts = []

        # Common field names for text content
        text_fields = ["text", "content", "body", "abstract", "title"]
        for field in text_fields:
            if field in doc and doc[field]:
                text_parts.append(str(doc[field]))

        # Handle nested structures
        if "body_text" in doc and isinstance(doc["body_text"], list):
            for section in doc["body_text"]:
                if isinstance(section, dict) and "text" in section:
                    text_parts.append(section["text"])

        full_text = " ".join(text_parts)
        return DataProcessor.clean_text(full_text)

    @staticmethod
    def split_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to split.
            chunk_size: Size of each chunk.
            overlap: Overlap between chunks.

        Returns:
            list: List of text chunks.
        """
        if not text or chunk_size <= 0:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            if end >= len(text):
                break

            start = end - overlap

        return chunks

    @staticmethod
    def validate_qa_pair(question: str, answer: str) -> bool:
        """Validate a QA pair.

        Args:
            question: Question text.
            answer: Answer text.

        Returns:
            bool: True if valid, False otherwise.
        """
        # Check minimum lengths
        if len(question.split()) < 3 or len(answer.split()) < 5:
            return False

        # Check for empty or placeholder content
        if question.lower() in ("?", "question") or answer.lower() in (".", "answer"):
            return False

        return True
