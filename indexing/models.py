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
    TECHNICAL_REPORT = "technical_report"
    THESIS = "thesis"
    PRESENTATION = "presentation"
    DOCUMENTATION = "documentation"
    PATENT = "patent"
    UNKNOWN = "unknown"

class BookMetadata(BaseModel):
    """Metadata schema for books"""
    title: str = Field(description="The full title of the book")
    authors: List[str] = Field(description="List of author names")
    publisher: Optional[str] = Field(description="Name of the publishing company")
    publication_year: Optional[int] = Field(description="Year the book was published")
    isbn: Optional[str] = Field(description="ISBN number if available")
    edition: Optional[str] = Field(description="Edition information if available")
    language: Optional[str] = Field(description="Primary language of the book")
    subject_areas: List[str] = Field(description="Main subject areas or categories")
    table_of_contents: Optional[List[str]] = Field(description="Main chapter titles")

class PaperMetadata(BaseModel):
    """Metadata schema for academic papers"""
    title: str = Field(description="The full title of the paper")
    authors: List[str] = Field(description="List of author names")
    abstract: str = Field(description="Paper abstract")
    keywords: List[str] = Field(description="Keywords or subject terms")
    doi: Optional[str] = Field(description="Digital Object Identifier")
    journal: Optional[str] = Field(description="Journal name if published")
    conference: Optional[str] = Field(description="Conference name if presented")
    publication_year: Optional[int] = Field(description="Year published/presented")
    institution: Optional[str] = Field(description="Research institution(s)")
    citations: Optional[List[str]] = Field(description="Key citations from first page")

class BlogArticleMetadata(BaseModel):
    """Metadata schema for blog articles"""
    title: str = Field(description="The full title of the article")
    authors: List[str] = Field(description="List of author names")
    publication_date: Optional[datetime] = Field(description="Publication date")
    blog_name: Optional[str] = Field(description="Name of the blog or platform")
    url: Optional[str] = Field(description="Original URL if available")
    tags: List[str] = Field(description="Article tags or categories")
    reading_time: Optional[int] = Field(description="Estimated reading time in minutes")
    summary: str = Field(description="Article summary or introduction")
    series: Optional[str] = Field(description="Blog post series name if part of one")

class TechnicalReportMetadata(BaseModel):
    """Metadata schema for technical reports"""
    title: str = Field(description="Report title")
    authors: List[str] = Field(description="List of authors")
    organization: str = Field(description="Organization that produced the report")
    report_number: Optional[str] = Field(description="Report identifier/number")
    date: Optional[datetime] = Field(description="Publication date")
    executive_summary: Optional[str] = Field(description="Executive summary")
    keywords: List[str] = Field(description="Key terms")
    classification: Optional[str] = Field(description="Report classification (e.g., Internal, Public)")

class ThesisMetadata(BaseModel):
    """Metadata schema for theses and dissertations"""
    title: str = Field(description="Thesis title")
    author: str = Field(description="Author name")
    degree: str = Field(description="Degree type (e.g., PhD, Masters)")
    institution: str = Field(description="Academic institution")
    department: Optional[str] = Field(description="Department or faculty")
    year: int = Field(description="Year of submission")
    advisors: List[str] = Field(description="Thesis advisors/supervisors")
    abstract: str = Field(description="Thesis abstract")
    keywords: List[str] = Field(description="Key terms")

class PatentMetadata(BaseModel):
    """Metadata schema for patents"""
    title: str = Field(description="Patent title")
    inventors: List[str] = Field(description="List of inventors")
    assignee: Optional[str] = Field(description="Patent assignee/owner")
    patent_number: Optional[str] = Field(description="Patent number")
    filing_date: Optional[datetime] = Field(description="Filing date")
    publication_date: Optional[datetime] = Field(description="Publication date")
    abstract: str = Field(description="Patent abstract")
    classification: Optional[str] = Field(description="Patent classification")
    claims: Optional[List[str]] = Field(description="Main patent claims")

models = {
    DocumentType.BOOK: BookMetadata,
    DocumentType.PAPER: PaperMetadata,
    DocumentType.BLOG_ARTICLE: BlogArticleMetadata,
    DocumentType.TECHNICAL_REPORT: TechnicalReportMetadata,
    DocumentType.THESIS: ThesisMetadata,
    DocumentType.PATENT: PatentMetadata,
}