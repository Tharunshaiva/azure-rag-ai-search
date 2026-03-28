# shared/indexer_service.py
import logging
import os
import json
from typing import List, Dict, Any

from shared.blob_client import BlobStorageClient
from shared.extractor import DocumentExtractor
from shared.chunker import Chunker
from shared.embedder import Embedder
from shared.search_client import SearchIndexClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class IndexerService:
    """
    Simplified indexer:
      - lists blobs (or accepts local files)
      - extracts -> chunk -> embed -> upsert
    NOTE: This version assumes the index already exists and exactly matches:
      id, file_id, chunk_id, topic, content, embedding, metadata
    """

    def __init__(
        self,
        blob_client: BlobStorageClient,
        extractor: DocumentExtractor,
        chunker: Chunker,
        embedder: Embedder,
        search_index_client: SearchIndexClient,
        chunk_batch_size: int = 16,
    ):
        self.blob_client = blob_client
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.search = search_index_client
        self.chunk_batch_size = chunk_batch_size

    def _make_chunk_doc(self, doc: Dict[str, Any], chunk_text: str, chunk_idx: int) -> Dict[str, Any]:
        chunk_id = f"{doc['id']}__chunk__{chunk_idx}"
        # build metadata and include source_link (if available)
        metadata_obj = dict(doc.get("metadata", {}) or {})
        metadata_obj["source_path"] = doc.get("source_path")
        source_link = None
        try:
            if hasattr(self, "blob_client") and doc.get("source_path"):
                source_link = self.blob_client.get_blob_url(doc.get("source_path"))
        except Exception:
            source_link = None
        metadata_obj["source_link"] = source_link
        metadata_json = json.dumps(metadata_obj, ensure_ascii=False)

        return {
            "id": chunk_id,
            "file_id": doc["id"],
            "chunk_id": str(chunk_idx),  # Convert to string to match schema
            "topic": doc.get("topic", "unknown"),
            "content": chunk_text,
            "embedding": None,  # filled later
            "metadata": metadata_json,
        }

    def index_files(self, blob_paths: List[str]) -> Dict[str, Any]:
        """
        Index the given list of blob paths or local file paths.
        Returns a summary.
        """
        overall = {
            "files_processed": 0,
            "total_chunks": 0,
            "total_upserted": 0,
            "errors": [],
        }

        for path in blob_paths:
            try:
                # Download blob or read local file
                try:
                    file_bytes, file_type = self.blob_client.download_blob(path)
                except Exception:
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            file_bytes = f.read()
                        _, ext = os.path.splitext(path)
                        file_type = ext.replace(".", "").lower()
                    else:
                        raise RuntimeError(f"Cannot read path: {path}")

                # Extract normalized doc
                doc = self.extractor.extract(file_bytes=file_bytes, file_type=file_type, blob_path=path)

                # Ensure we don't duplicate chunks for the same file_id: delete existing ones first
                try:
                    self.search.delete_chunks_by_file_id(doc["id"])
                except Exception:
                    # best-effort; continue even if delete fails
                    logger.warning("Could not delete existing chunks for %s; continuing", doc.get("id"))

                content = doc.get("content", "") or ""
                chunks = self.chunker.chunk_text(content)
                overall["files_processed"] += 1
                overall["total_chunks"] += len(chunks)

                # Prepare chunk docs
                chunk_docs = [self._make_chunk_doc(doc, c, i) for i, c in enumerate(chunks)]

                # embed in batches and upsert
                upserted = 0
                for i in range(0, len(chunk_docs), self.chunk_batch_size):
                    batch = chunk_docs[i : i + self.chunk_batch_size]
                    # generate embeddings and attach
                    for cd in batch:
                        try:
                            emb = self.embedder.embed_text(cd["content"])
                            cd["embedding"] = emb
                        except Exception as ex:
                            logger.exception("Embedding failed for %s: %s", cd["id"], ex)
                            cd["embedding"] = []

                    # upsert batch
                    res = self.search.upsert_chunks(batch)
                    upserted += res.get("uploaded", 0) if isinstance(res, dict) else 0

                overall["total_upserted"] += upserted

            except Exception as e:
                logger.exception("Failed to index path %s: %s", path, e)
                overall["errors"].append({path: str(e)})

        return overall
