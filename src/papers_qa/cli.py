"""CLI for Papers QA system."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from papers_qa import (
    BatchQAGenerator,
    DataLoader,
    DataProcessor,
    QAEvaluator,
    RetrieverPipeline,
    configure_logging,
    get_logger,
    get_settings,
)

logger = get_logger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Papers QA: Medical Paper Question Answering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  papers-qa generate --input data/raw --output data/generated
  papers-qa index --documents data/generated/documents.json
  papers-qa query --index data/cache --question "What is adenomyosis?"
  papers-qa evaluate --references data/answers.json --predictions data/predictions.json
        """,
    )

    # Global options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate QA pairs from documents")
    gen_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory with documents",
    )
    gen_parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file for QA pairs",
    )
    gen_parser.add_argument(
        "--model",
        type=str,
        help="LLM model to use",
    )
    gen_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation",
    )

    # Index command
    index_parser = subparsers.add_parser("index", help="Create vector index")
    index_parser.add_argument(
        "--documents",
        type=Path,
        required=True,
        help="Path to documents JSON file",
    )
    index_parser.add_argument(
        "--output",
        type=Path,
        help="Output index directory",
    )
    index_parser.add_argument(
        "--model",
        type=str,
        help="Embedding model to use",
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument(
        "--index",
        type=Path,
        required=True,
        help="Path to vector index",
    )
    query_parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to answer",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to retrieve",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate QA system")
    eval_parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="Path to reference answers",
    )
    eval_parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predicted answers",
    )
    eval_parser.add_argument(
        "--output",
        type=Path,
        help="Output evaluation results",
    )

    return parser


def generate_qa(args: argparse.Namespace) -> int:
    """Handle generate command.

    Args:
        args: Parsed arguments.

    Returns:
        int: Exit code.
    """
    try:
        logger.info("starting_qa_generation", input_dir=str(args.input))

        # Load documents
        loader = DataLoader()
        documents = loader.load_documents(args.input)

        if not documents:
            logger.error("no_documents_found", input_dir=str(args.input))
            return 1

        logger.info("documents_loaded", count=len(documents))

        # Extract text from documents
        processor = DataProcessor()
        texts = [
            processor.extract_text_from_doc(doc)
            if isinstance(doc, dict)
            else str(doc)
            for doc in documents
        ]

        # Generate QA pairs
        batch_size = args.batch_size
        generator = BatchQAGenerator(batch_size=batch_size, model_name=args.model)
        qa_dataset = generator.generator.generate_dataset(texts)

        # Save output
        output_path = args.output or get_settings().data.output_dir / "qa_pairs.csv"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        qa_dataset.to_csv(output_path, index=False)
        logger.info("qa_dataset_saved", path=str(output_path), records=len(qa_dataset))

        return 0

    except Exception as e:
        logger.error("generation_failed", error=str(e), exc_info=True)
        return 1


def index_documents(args: argparse.Namespace) -> int:
    """Handle index command.

    Args:
        args: Parsed arguments.

    Returns:
        int: Exit code.
    """
    try:
        logger.info("starting_indexing", documents_path=str(args.documents))

        # Load documents
        loader = DataLoader()
        if args.documents.suffix == ".json":
            import json

            with open(args.documents) as f:
                documents = json.load(f)
        else:
            documents = loader.load_documents(args.documents)

        if not documents:
            logger.error("no_documents_found")
            return 1

        # Create retriever
        retriever = RetrieverPipeline(model_name=args.model)

        # Index documents
        doc_texts = [
            d if isinstance(d, str) else str(d)
            for d in documents
        ]
        retriever.index_documents(doc_texts)

        # Save index
        output_path = args.output or get_settings().data.cache_dir / "index"
        retriever.indexer.save(Path(output_path))

        logger.info("indexing_complete", output_path=str(output_path))
        return 0

    except Exception as e:
        logger.error("indexing_failed", error=str(e), exc_info=True)
        return 1


def query_system(args: argparse.Namespace) -> int:
    """Handle query command.

    Args:
        args: Parsed arguments.

    Returns:
        int: Exit code.
    """
    try:
        logger.info("starting_query", question=args.question)

        # Load retriever
        retriever = RetrieverPipeline()
        retriever.load()

        # Retrieve documents
        results = retriever.retrieve(args.question, k=args.top_k)

        # Display results
        print("\n" + "=" * 80)
        print(f"Question: {args.question}\n")

        for idx, (doc, score) in enumerate(results, 1):
            print(f"[{idx}] Similarity: {score:.4f}")
            print(f"Document: {doc[:500]}...\n")

        print("=" * 80 + "\n")
        return 0

    except Exception as e:
        logger.error("query_failed", error=str(e), exc_info=True)
        return 1


def evaluate_system(args: argparse.Namespace) -> int:
    """Handle evaluate command.

    Args:
        args: Parsed arguments.

    Returns:
        int: Exit code.
    """
    try:
        logger.info("starting_evaluation")

        # Load predictions
        refs_df = pd.read_csv(args.references)
        preds_df = pd.read_csv(args.predictions)

        if len(refs_df) != len(preds_df):
            logger.warning("mismatched_lengths", refs=len(refs_df), preds=len(preds_df))

        # Evaluate
        evaluator = QAEvaluator()
        metrics_list = []

        for ref_ans, pred_ans in zip(
            refs_df.iloc[:, 0], preds_df.iloc[:, 0], strict=False
        ):
            metrics = evaluator.evaluate_answer(str(ref_ans), str(pred_ans))
            metrics_list.append(metrics)

        # Aggregate and save
        avg_metrics = {
            key: sum(m.get(key, 0) for m in metrics_list) / len(metrics_list)
            for key in metrics_list[0].keys()
            if isinstance(metrics_list[0].get(key), (int, float))
        }

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.Series(avg_metrics).to_csv(output_path, header=False)

        # Display results
        print("\n" + "=" * 80)
        print("Evaluation Results:")
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.4f}")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        logger.error("evaluation_failed", error=str(e), exc_info=True)
        return 1


def main() -> int:
    """Main CLI entry point.

    Returns:
        int: Exit code.
    """
    parser = setup_parser()
    args = parser.parse_args()

    # Configure logging
    configure_logging()

    # Handle commands
    if args.command == "generate":
        return generate_qa(args)
    elif args.command == "index":
        return index_documents(args)
    elif args.command == "query":
        return query_system(args)
    elif args.command == "evaluate":
        return evaluate_system(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
