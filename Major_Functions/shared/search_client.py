# shared/search_client.py
import logging
import json
from typing import List, Dict, Any, Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

logger = logging.getLogger(__name__)


class SearchIndexClient:
    """
    Simple SearchIndexClient that assumes the index already exists.
    Schema expected:
      id, file_id, chunk_id, topic, content, embedding, metadata
    """

    def __init__(self, endpoint: str, admin_key: str, index_name: str = "rag-chunks"):
        self.endpoint = endpoint.rstrip("/")
        self.admin_key = admin_key
        self.index_name = index_name

        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.admin_key),
        )

    def upsert_chunks(self, chunk_documents: List[Dict[str, Any]]):
        """
        Upsert documents into the existing index.
        Each document must have fields that match the index schema.
        """
        try:
            if not chunk_documents:
                return {"uploaded": 0}

            result = self.search_client.upload_documents(documents=chunk_documents)
            succeeded = sum(1 for r in result if getattr(r, "succeeded", False))
            logger.info("Upserted %d documents to index %s", succeeded, self.index_name)
            return {"uploaded": succeeded}
        except Exception as ex:
            logger.exception("Failed to upsert chunks: %s", ex)
            raise

    def delete_chunks_by_file_id(self, file_id: str):
        """
        Optional helper if you later want to delete old chunks for a file.
        Uses the search API to find docs with file_id and delete them by id.
        """
        try:
            filter_expr = f"file_id eq '{file_id}'"
            docs_to_delete = []
            results = self.search_client.search(search_text="*", filter=filter_expr, top=1000)
            for r in results:
                try:
                    doc_id = r["id"] if "id" in r else getattr(r, "id", None)
                except Exception:
                    doc_id = None
                if doc_id:
                    docs_to_delete.append({"id": doc_id})

            if not docs_to_delete:
                return {"deleted": 0}

            self.search_client.delete_documents(documents=docs_to_delete)
            logger.info("Deleted %d docs for file_id %s", len(docs_to_delete), file_id)
            return {"deleted": len(docs_to_delete)}
        except Exception as ex:
            logger.exception("Failed to delete by file_id: %s", ex)
            raise

    def vector_search(self, query_embedding: List[float], top_k: int = 5, topic_filter: Optional[str] = None):
        """
        Run vector search. This assumes the index's 'embedding' vector field is configured.
        Returns list of dicts with at least: id, file_id, chunk_id, topic, content, score
        """
        try:
            if not query_embedding:
                return []

            # Use VectorizedQuery for modern SDK
            vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=top_k, fields="embedding")
            
            filter_expr = None
            if topic_filter:
                safe_topic = topic_filter.replace("'", "''")
                filter_expr = f"topic eq '{safe_topic}'"

            resp = self.search_client.search(search_text=None, vector_queries=[vector_query], filter=filter_expr, top=top_k)
            results = []
            for r in resp:
                # Ensure we read '@search.score' correctly whether r is a dict (raw doc) or an SDK object
                if isinstance(r, dict):
                    score_val = r.get("@search.score") or r.get("score")
                else:
                    score_val = getattr(r, "@search.score", None) or getattr(r, "score", None)

                # parse metadata if present (may be stored as JSON string)
                if isinstance(r, dict):
                    metadata_raw = r.get("metadata")
                else:
                    metadata_raw = getattr(r, "metadata", None)
                metadata = None
                if isinstance(metadata_raw, str):
                    try:
                        metadata = json.loads(metadata_raw)
                    except Exception:
                        metadata = {"raw": metadata_raw}
                elif isinstance(metadata_raw, dict):
                    metadata = metadata_raw

                source_link = None
                if metadata and isinstance(metadata, dict):
                    source_link = metadata.get("source_link")

                entry = {
                    "id": r.get("id") if isinstance(r, dict) else getattr(r, "id", None),
                    "file_id": r.get("file_id") if isinstance(r, dict) else getattr(r, "file_id", None),
                    "chunk_id": r.get("chunk_id") if isinstance(r, dict) else getattr(r, "chunk_id", None),
                    "topic": r.get("topic") if isinstance(r, dict) else getattr(r, "topic", None),
                    "content": r.get("content") if isinstance(r, dict) else getattr(r, "content", None),
                    "score": score_val,
                    "metadata": metadata,
                    "source_link": source_link,
                }
                results.append(entry)
            return results
        except Exception as ex:
            logger.exception("Vector search failed: %s", ex)
            raise
