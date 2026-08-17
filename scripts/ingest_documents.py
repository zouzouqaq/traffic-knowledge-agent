"""Ingest one document into the local metadata store."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from traffic_knowledge.ingestion.chunking import DocumentChunk
from traffic_knowledge.ingestion.repository import DocumentRepository
from traffic_knowledge.ingestion.service import IngestionService


class MetadataIndex:
    """Task-4 adapter; Chroma persistence replaces this in the retrieval batch."""

    def upsert(self, chunks: tuple[DocumentChunk, ...]) -> None:
        if not chunks:
            raise ValueError("a document must produce at least one chunk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument(
        "--database-path", type=Path, default=Path("data/metadata.sqlite3")
    )
    parser.add_argument("--max-file-bytes", type=int, default=50 * 1024 * 1024)
    parser.add_argument("--max-characters", type=int, default=1000)
    parser.add_argument("--overlap-characters", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = DocumentRepository(args.database_path)
    repository.initialize()
    service = IngestionService(
        repository=repository,
        vector_index=MetadataIndex(),
        max_file_bytes=args.max_file_bytes,
        max_characters=args.max_characters,
        overlap_characters=args.overlap_characters,
    )
    result = service.ingest(args.document)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
