import azure.functions as func
import logging
import json
import os

from shared.blob_client import BlobStorageClient
from shared.extractor import DocumentExtractor
from shared.chunker import Chunker
from shared.embedder import Embedder
from shared.search_client import SearchIndexClient
from shared.indexer_service import IndexerService
from shared.index_manager import IndexManager
from shared.rag_service import RagService

# Azure OpenAI for GPT
from openai import AzureOpenAI


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
logging.basicConfig(level=logging.INFO)



AZURE_STORAGE_CONNSTR = os.getenv("AZURE_STORAGE_CONNSTR")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER")

AZ_SEARCH_ENDPOINT = os.getenv("AZ_SEARCH_ENDPOINT")
AZ_SEARCH_ADMIN_KEY = os.getenv("AZ_SEARCH_ADMIN_KEY")
AZ_SEARCH_INDEX = os.getenv("AZ_SEARCH_INDEX", "rahuraam-index")  # Default fallback

AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_KEY = os.getenv("AOAI_KEY")
AOAI_EMBED_MODEL = os.getenv("AOAI_EMBEDDING_MODEL", "text-embedding-ada-002")
AOAI_CHAT_MODEL = os.getenv("AOAI_CHAT_MODEL", "gpt-4o")

FORM_ENDPOINT = os.getenv("FORM_RECOGNIZER_ENDPOINT")
FORM_KEY = os.getenv("FORM_RECOGNIZER_KEY")

logger = logging.getLogger(__name__)

# Validate critical env vars
if not all([AOAI_ENDPOINT, AOAI_KEY, AZ_SEARCH_ENDPOINT, AZ_SEARCH_ADMIN_KEY]):
    logger.warning("Missing critical Azure credentials. Some features may not work.")

# ---------------------------------------------------------
# Create shared service objects

blob_client = BlobStorageClient(AZURE_STORAGE_CONNSTR, AZURE_CONTAINER)
extractor = DocumentExtractor(FORM_ENDPOINT, FORM_KEY)
chunker = Chunker(chunk_size=1024, overlap=256)
embedder = Embedder(AOAI_ENDPOINT, AOAI_KEY, AOAI_EMBED_MODEL)
search_client = SearchIndexClient(AZ_SEARCH_ENDPOINT, AZ_SEARCH_ADMIN_KEY, AZ_SEARCH_INDEX)

# Ensure index exists (best-effort). Vector dim default 1536 for common embedding models.
index_manager = IndexManager(AZ_SEARCH_ENDPOINT, AZ_SEARCH_ADMIN_KEY)
try:
    index_manager.ensure_index(AZ_SEARCH_INDEX, vector_dim=1536)
except Exception as ex:
    # continue even if index creation fails here; index operations will error later if missing
    logger.warning("Index creation failed: %s", ex)

indexer = IndexerService(blob_client, extractor, chunker, embedder, search_client)

# GPT client for /ask
gpt_client = AzureOpenAI(
    api_key=AOAI_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AOAI_ENDPOINT
)

# RAG orchestrator
rag_service = RagService(embedder=embedder, search_index_client=search_client, llm_client=gpt_client, chat_model=AOAI_CHAT_MODEL)


# ---------------------------------------------------------
# /index endpoint
# ---------------------------------------------------------
@app.function_name(name="index")
@app.route(route="index", methods=["POST"])
def index_documents(req: func.HttpRequest) -> func.HttpResponse:
    """
    Body examples:

    1) Index ALL blobs:
       {}

    2) Index a single file from blob:
       { "blob_path": "topic1/myfile.pdf" }

    """

    try:
        body = req.get_json()
        logging.info("this is a info message")
        logging.warning("this is a warning message")

        if "blob_path" in body:
            paths = [body["blob_path"]]

        elif "local_path" in body:
            paths = [body["local_path"]]

        else:
            # index all blobs in container
            paths = blob_client.list_blobs()

        result = indexer.index_files(paths)
        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False, indent=2),
            mimetype="application/json",
            status_code=200
        )

    except Exception as ex:
        logging.exception("Index API failed")
        return func.HttpResponse(str(ex), status_code=500)


# ---------------------------------------------------------
# /ask endpoint
# ---------------------------------------------------------
@app.function_name(name="ask")
@app.route(route="ask", methods=["POST"])
def ask_question(req: func.HttpRequest) -> func.HttpResponse:
    """
    Body:
    {
        "question": "What is supervised learning?",
        "top_k": 4,
        "topic": "ML"    
    }
    """

    try:
        body = req.get_json()
        question = body.get("question")
        if not question:
            return func.HttpResponse("Missing 'question' field", status_code=400)

        top_k = body.get("top_k", 4)
        topic = body.get("topic")  # optional filter

        # Use RagService to perform embed -> retrieve -> generate answer
        resp = rag_service.answer_question(question=question, top_k=top_k, topic=topic)
        response = resp

        return func.HttpResponse(
            json.dumps(response, ensure_ascii=False, indent=2),
            mimetype="application/json",
            status_code=200
        )

    except Exception as ex:
        logging.exception("Ask API failed")
        return func.HttpResponse(str(ex), status_code=500)
