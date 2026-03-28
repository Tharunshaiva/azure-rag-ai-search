from typing import Optional
import logging

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient as AdminSearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Responsible for creating or ensuring the Cognitive Search index exists.
    This creates a vector-capable index suitable for RAG chunk storage.
    """

    def __init__(self, endpoint: str, admin_key: str):
        self.endpoint = endpoint.rstrip("/")
        self.admin_key = admin_key
        self.client = AdminSearchIndexClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.admin_key))

    def ensure_index(self, index_name: str, vector_dim: int = 1536) -> None:
        """
        Create the index if it does not exist. If it exists, do nothing.
        """
        try:
            existing = None
            try:
                existing = self.client.get_index(index_name)
            except Exception:
                existing = None

            if existing:
                logger.info("Index '%s' already exists", index_name)
                return

            # Define fields with proper vector configuration
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=False),
                SimpleField(name="file_id", type=SearchFieldDataType.String, filterable=True, facetable=False),
                SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=False, facetable=False),
                SearchField(
                    name="content",
                    type=SearchFieldDataType.String,
                    searchable=True,
                    analyzer_name="en.microsoft"
                ),
                SearchField(
                    name="topic",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    searchable=True,
                    analyzer_name="en.microsoft"
                ),
                SimpleField(name="metadata", type=SearchFieldDataType.String, filterable=False),
                # embedding vector field with proper vector search configuration
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=vector_dim,
                    vector_search_profile_name="myHnsw"
                ),
            ]

            # Define vector search configuration
            vector_search = VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
                profiles=[VectorSearchProfile(name="myHnsw", algorithm_configuration_name="myHnsw")]
            )

            index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)

            self.client.create_index(index)
            logger.info("Created index '%s' with vector dimension=%s", index_name, vector_dim)

        except Exception as ex:
            logger.exception("Failed to ensure index %s: %s", index_name, ex)
            raise
