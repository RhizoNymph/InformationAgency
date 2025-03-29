from indexing.models import DocumentType
import logging
from typing import Tuple
from pydantic_ai import Agent
from indexing.settings import model

logger = logging.getLogger(__name__)

document_classifier = Agent(
    model,
    result_type=DocumentType,
    system_prompt="""
    You are a document classification specialist. Analyze the provided text sample (first ~3000 characters) from a document and determine its type.
    Choose ONE from the following categories:
    BOOK, PAPER, BLOG_ARTICLE, TECHNICAL_REPORT, THESIS, PRESENTATION, DOCUMENTATION, PATENT.
    If unsure or it doesn't fit well, classify as UNKNOWN.

    Pay attention to:
    - Keywords like 'abstract', 'doi', 'isbn', 'patent', 'claims', 'report number', 'thesis', 'dissertation', 'slides'.
    - Formatting: Presence of citations, references, table of contents, legal clauses.
    - Tone: Formal academic, technical, legal, informal blog style.
    - Structure: Chapters, sections, claims, slides layout hints.
    """
)

async def classify(text_sample: str) -> DocumentType:
    """
    Classifies the document based on a text sample using LLM and heuristics.
    Returns DocumentType
    """
    logger.info("Attempting document classification using LLM...")

    if not text_sample.strip():
            logger.warning("Text sample is empty, cannot classify.")
            return DocumentType.UNKNOWN

    result = await document_classifier.run(text_sample)
    if result:
        doc_type = result
        logger.info(f"LLM classified document as: {doc_type.value}")
        return doc_type
    else:
            logger.warning("LLM classification returned no result.")
            raise Exception("LLM returned no result")