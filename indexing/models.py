from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class FileType(Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    EPUB = "epub"
    ODT = "odt"

class DocumentType(Enum):
    BOOK = "book"
    PAPER = "paper"
    BLOG_ARTICLE = "blog_article"    
    THESIS = "thesis"
    PRESENTATION = "presentation"
    DOCUMENTATION = "documentation"
    PATENT = "patent"
    UNKNOWN = "unknown"

# Base model for common metadata fields useful for filtering
class BaseDocumentMetadata(BaseModel):
    title: str = Field(description="The title of the document")
    authors: List[str] = Field(description="List of author names")
    publication_date: Optional[datetime] = Field(description="Publication date of the document")
    keywords: List[str] = Field(description="Keywords or subject terms")
    language: Optional[str] = Field(description="Language of the document")

class BookMetadata(BaseDocumentMetadata):
    """Simplified metadata schema for books"""
    publisher: Optional[str] = Field(description="Name of the publishing company")    
    isbn: Optional[str] = Field(description="ISBN number if available")

class PaperMetadata(BaseDocumentMetadata):
    """Simplified metadata schema for academic papers"""    
    doi: Optional[str] = Field(description="Digital Object Identifier")
    abstract: Optional[str] = Field(description="Abstract of the paper")
    journal: Optional[str] = Field(description="Journal name if published")
    conference: Optional[str] = Field(description="Conference name if presented")
    institution: Optional[str] = Field(description="Research institution(s)")
    citation_count: Optional[int] = Field(description="Number of citations")

class BlogArticleMetadata(BaseDocumentMetadata):
    """Simplified metadata schema for blog articles"""
    blog_name: str = Field(description="Name of the blog or platform")    

class ThesisMetadata(BaseDocumentMetadata):
    """Simplified metadata schema for theses and dissertations"""
    degree: str = Field(description="Degree type (e.g., PhD, Masters)")
    institution: str = Field(description="Academic institution")
    citation_count: Optional[int] = Field(description="Number of citations")

class PatentMetadata(BaseDocumentMetadata):
    """Simplified metadata schema for patents"""
    assignee: Optional[str] = Field(description="Patent assignee/owner")
    patent_number: Optional[str] = Field(description="Patent number")
    filing_date: Optional[datetime] = Field(description="Filing date")
    classification: Optional[str] = Field(description="Patent classification")

class PresentationMetadata(BaseDocumentMetadata):
    """Simplified metadata schema for presentations"""
    event_name: Optional[str] = Field(description="Name of the conference, meeting, or event")

class DocumentationMetadata(BaseDocumentMetadata):
    """Simplified metadata schema for technical documentation"""
    product_name: str = Field(description="Name of the software or product documented")
    version: Optional[str] = Field(description="Version of the software or product documented")
    project_url: Optional[str] = Field(description="URL for the project or product website")

# Mapping from DocumentType to the corresponding metadata model
models = {
    DocumentType.BOOK: BookMetadata,
    DocumentType.PAPER: PaperMetadata,
    DocumentType.BLOG_ARTICLE: BlogArticleMetadata,
    DocumentType.THESIS: ThesisMetadata,
    DocumentType.PATENT: PatentMetadata,
    DocumentType.PRESENTATION: PresentationMetadata,
    DocumentType.DOCUMENTATION: DocumentationMetadata,    
    DocumentType.UNKNOWN: BaseDocumentMetadata
}