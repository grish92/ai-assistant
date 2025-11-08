from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from bs4 import BeautifulSoup, Comment
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from newspaper import Article

from app.core.config import settings

JUNK_TAGS = ("header", "footer", "nav", "aside", "form", "script", "style")
JUNK_HINTS = [
    "header", "footer", "navbar", "breadcrumb",
    "sidebar", "ads", "ad-", "advert", "cookie",
    "subscribe", "newsletter", "social", "share",
]
MOBILE_HINTS = [
    "mobile", "mobi", "m-only", "mobile-only", "hidden-md-up",
]

def ensure_vectorstore() -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

    collection_name = settings.QDRANT_COLLECTION

    client = QdrantClient(
        url=settings.QDRANT_URL,
    )

    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE,
            ),
        )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vectorstore


def store_docs_serial(vectorstore: QdrantVectorStore, docs: List[Document]) -> None:
    vectorstore.add_documents(docs)


def store_docs_parallel(vectorstore: QdrantVectorStore, docs: List[Document], chunk_size: int = 10) -> None:
    """
    Store documents in parallel chunks to speed up embedding+upload for large lists.
    """
    chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(vectorstore.add_documents, chunk) for chunk in chunks]
        for fut in as_completed(futures):
            fut.result()





def clean_downloaded_html(raw_html: str) -> str:
    """
    Super-defensive HTML cleaner.
    Removes nav/header/footer/mobile/junk blocks and returns a body-only HTML.
    Never raises on weird tags.
    """
    soup = BeautifulSoup(raw_html or "", "html.parser")

    # remove comments
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    # iterate over a snapshot so decompose() is safe
    for el in list(soup.find_all(True)):
        # be ultra defensive
        try:
            if el is None:
                continue

            # some nodes may not have attrs at all
            if not hasattr(el, "name"):
                continue

            # 1) junk by tag
            if el.name in JUNK_TAGS:
                el.decompose()
                continue

            # safely get class/id
            classes = el.get("class") or []
            if isinstance(classes, str):
                classes = [classes]
            cls = " ".join(classes).lower()
            el_id = (el.get("id") or "").lower()

            # 2) junk by id/class
            if any(h in cls or h in el_id for h in JUNK_HINTS):
                el.decompose()
                continue

            # 3) mobile-only
            if any(h in cls or h in el_id for h in MOBILE_HINTS):
                el.decompose()
                continue

        except Exception:
            continue

    body = soup.find("body")
    return str(body or soup)


def extract_html(url: str):
    """Download, clean, and parse only the main desktop content."""
    art = Article(url)
    art.download()
    cleaned_html = clean_downloaded_html(art.html)
    art.set_html(cleaned_html)
    art.parse()
    print(art)

    return {"html":art.html}