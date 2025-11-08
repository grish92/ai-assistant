import logging
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client.http.models import VectorParams, Distance

from app.core.config import settings

LOGGER = logging.getLogger(__name__)


class VectorStoreReader:
    """
    Lazy Qdrant-based vector store reader.

    - Does NOT connect to Qdrant in __init__
    - Connects only on first real use
    - Raises a clear error if Qdrant is down
    """

    def __init__(self) -> None:
        self._initialized = False
        self.embeddings: Optional[OpenAIEmbeddings] = None
        self.client: Optional[QdrantClient] = None
        self.vectorstore: Optional[QdrantVectorStore] = None
        self.retriever = None

    def _init(self) -> None:
        """Connect to Qdrant and prepare collection (runs once)."""
        if self._initialized:
            return

        LOGGER.info("Initializing Qdrant vector store connection to %s", settings.QDRANT_URL)
        self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

        self.client = QdrantClient(
            url=settings.QDRANT_URL,
        )

        collection_name = settings.QDRANT_COLLECTION

        try:
            if not self.client.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                LOGGER.info("Created Qdrant collection '%s'", collection_name)
        except Exception as e:
            LOGGER.error("Cannot connect to Qdrant at %s: %s", settings.QDRANT_URL, e)
            raise RuntimeError(f"Cannot connect to Qdrant or create collection: {e}") from e

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        self._initialized = True
        LOGGER.info("Vector store initialized (collection=%s)", collection_name)


    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        LangChain-style retriever.
        """
        self._init()
        LOGGER.debug("Retrieving %d documents using query='%s...'", k, query[:50])
        return self.vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Simple vector similarity search. No metadata filters.
        """
        self._init()
        LOGGER.debug("Similarity search (k=%d) query='%s...'", k, query[:50])
        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_filter(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        k: int = 4,
    ) -> List[Document]:
        """
        Filtered semantic search using stored metadata.
        """
        self._init()
        LOGGER.debug(
            "Similarity search with filter (k=%d, filter=%s) query='%s...'",
            k,
            metadata_filter,
            query[:50],
        )
        return self.vectorstore.similarity_search(
            query,
            k=k,
            filter=metadata_filter,
        )