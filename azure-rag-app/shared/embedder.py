# shared/embedder.py

from openai import AzureOpenAI
import logging

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generates embeddings using Azure OpenAI REST API.
    """

    def __init__(self, aoai_endpoint: str, aoai_key: str, model_name: str):
        self.endpoint = aoai_endpoint
        self.key = aoai_key
        self.model = model_name

        self.client = AzureOpenAI(
            api_key=self.key,
            api_version="2024-02-15-preview",
            azure_endpoint=self.endpoint
        )

    def embed_text(self, text: str):
        """
        Returns a list (vector) representing the embedding for the given text.
        """
        if text is None or text.strip() == "":
            return []

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            # response.data[0].embedding is the embedding vector
            embedding = response.data[0].embedding
            # logger.info(f"embeddings generated successfully: {embedding[:10]}... (truncated)")
            return embedding
        except Exception as ex:
            logger.exception("Embedding generation failed: %s", ex)
            raise
