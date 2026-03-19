"""Knowledge Retrieval Module — RAG pipeline using Chroma for plant health knowledge."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.agent.models import RetrievedDocument
from backend.config.settings import settings

logger = logging.getLogger(__name__)

_vectorstore: Chroma | None = None


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name="plant_knowledge",
            embedding_function=_get_embeddings(),
            persist_directory=settings.chroma_persist_dir,
        )
    return _vectorstore


def load_knowledge_base(force_reload: bool = False) -> int:
    """Load text files from the knowledge directory into Chroma. Returns document count."""
    vs = _get_vectorstore()

    existing = vs._collection.count()
    if existing > 0 and not force_reload:
        logger.info("Knowledge base already loaded (%d chunks), skipping.", existing)
        return existing

    knowledge_dir = Path(settings.knowledge_dir)
    if not knowledge_dir.exists():
        logger.warning("Knowledge directory not found: %s", knowledge_dir)
        return 0

    raw_docs: list[Document] = []
    for filepath in sorted(knowledge_dir.glob("*.txt")):
        text = filepath.read_text(encoding="utf-8")
        raw_docs.append(
            Document(page_content=text, metadata={"source": filepath.name})
        )

    if not raw_docs:
        logger.warning("No .txt files found in %s", knowledge_dir)
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info("Split %d documents into %d chunks.", len(raw_docs), len(chunks))

    if force_reload:
        vs._collection.delete(where={"source": {"$ne": ""}})

    vs.add_documents(chunks)
    logger.info("Loaded %d chunks into Chroma.", len(chunks))
    return len(chunks)


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedDocument]:
    """Retrieve the most relevant plant-health documents for a given query."""
    vs = _get_vectorstore()
    k = top_k or settings.rag_top_k

    if vs._collection.count() == 0:
        logger.warning("Vector store is empty — loading knowledge base.")
        load_knowledge_base()

    results = vs.similarity_search_with_relevance_scores(query, k=k)

    docs: list[RetrievedDocument] = []
    for doc, score in results:
        docs.append(
            RetrievedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                relevance_score=round(score, 4),
            )
        )

    logger.info("Retrieved %d documents for query: '%s'", len(docs), query[:80])
    return docs
