# main.py
import uvicorn
import shutil
import hashlib
import asyncio
import tempfile
import numpy as np 
from datetime import datetime, timezone
import fitz # <--- Add PyMuPDF import


# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from typing import Annotated, Dict, Any, List, Tuple, Optional # Updated typing

# Standard library imports
import os
import logging # Added logging
from collections import defaultdict # For RRF

# --- Import your project modules ---
from storage.qdrant_client import QdrantColbertClient
from storage.opensearch_client import OpenSearchClient
from storage.minio_client import MinioStorageClient
from indexing.document_classifier import classify
from indexing.metadata_extractor import extract_metadata
from indexing.models import DocumentType, FileType, models # <--- Import FileType
from indexing.utils import calculate_file_hash
from indexing.settings import *
# Import OpenSearch mappings
from storage.opensearch_mappings import mappings as opensearch_doc_mappings
from qdrant_client import models as qdrant_models # Import qdrant models separately

# --- Import and Setup Embedding Model ---
try:
    from fastembed.embedding import FlagEmbedding # Generic embedding if needed later
    from fastembed import LateInteractionTextEmbedding # For ColBERT
    # Initialize ColBERT model globally for efficiency
    # Consider making the model name configurable via settings
    colbert_model_name = "colbert-ir/colbertv2.0"
    print(f"Loading ColBERT embedding model: {colbert_model_name}")
    embedding_model = LateInteractionTextEmbedding(model_name=colbert_model_name, cache_dir=os.getenv("FASTEMBED_CACHE_DIR", "local_cache/fastembed"))
    print("ColBERT embedding model loaded.")
except ImportError:
    print("Error: 'fastembed' library not found or ColBERT model unavailable. Install it: pip install 'fastembed>=0.2.0'")
    # Define a dummy model or raise an error to prevent startup without embeddings
    raise ImportError("fastembed required")
except Exception as e:
     print(f"Error loading embedding model '{colbert_model_name}': {e}")
     embedding_model = None # Handle loading errors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Instantiate Clients ---
qdrant_client = QdrantColbertClient()
opensearch_client = OpenSearchClient()
minio_client = MinioStorageClient() # Sync client, connection handled internally per call


# --- FastAPI App ---
app = FastAPI(
    title="Document Indexing API",
    description="An API to upload, classify, extract metadata, and index documents.",
    version="1.0.2", # Incremented version
    # Add lifespan management for async clients in production
    # lifespan=lifespan, # Requires defining lifespan async context manager
)

# --- Helper Functions ---

def get_opensearch_mappings(doc_type: DocumentType) -> Dict[str, Any]:
    """Returns the appropriate OpenSearch mappings for the document type."""
    if doc_type == DocumentType.UNKNOWN:
         logger.warning("Attempted to get mappings for UNKNOWN document type.")
         # Return base properties only, or raise error? Let's return base.
         return {"properties": opensearch_client.settings.BASE_MAPPING_PROPERTIES}

    # Get mappings specific to the doc_type from the imported dict
    specific_mappings = opensearch_doc_mappings.get(doc_type.value.lower()) # Use enum value, ensure lowercase key match
    if specific_mappings:
        # Base properties should be handled by create_index_if_not_exists in the client now
        # logger.debug(f"Found specific mappings for {doc_type.value}")
        return specific_mappings # Return only the specific properties part
    else:
        logger.warning(f"No specific OpenSearch mappings found for document type '{doc_type.value}'. Using base properties.")
        # Return base properties only if specific ones are missing
        return {"properties": opensearch_client.settings.BASE_MAPPING_PROPERTIES}


def get_file_type_from_upload(upload_file: UploadFile) -> FileType:
    """Determines the FileType based on filename extension or content type."""
    extension_map = {
        ".pdf": FileType.PDF,
        ".txt": FileType.TXT,
        ".docx": FileType.DOCX,
        ".html": FileType.HTML,
        ".htm": FileType.HTML,
        ".epub": FileType.EPUB,
        ".odt": FileType.ODT,
    }
    filename = upload_file.filename or ""
    content_type = upload_file.content_type or ""
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension in extension_map:
        return extension_map[file_extension]

    # Fallback based on content type
    if "pdf" in content_type:
        return FileType.PDF
    elif "text/plain" in content_type:
        return FileType.TXT
    elif "vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type:
        return FileType.DOCX
    elif "text/html" in content_type:
        return FileType.HTML
    elif "epub+zip" in content_type:
        return FileType.EPUB
    elif "vnd.oasis.opendocument.text" in content_type:
        return FileType.ODT

    logger.warning(f"Could not determine FileType for '{filename}' (type: {content_type}). Defaulting to TXT.")
    # Consider raising an error or returning a specific 'UNKNOWN_FILETYPE' if needed
    return FileType.TXT # Or handle unknown type more robustly


async def chunk_document(file_path: str, file_type: FileType) -> list[str]:
    """
    Chunks the document based on its FileType.
    Uses PyMuPDF for PDFs, basic text splitting for TXT.
    Other types are placeholders.
    """
    logger.info(f"Starting chunking for file: {file_path}, type: {file_type.value}")
    chunks = []
    try:
        if file_type == FileType.PDF:
            logger.debug("Processing PDF using PyMuPDF...")
            try:
                doc = fitz.open(file_path)
                for page_num, page in enumerate(doc.pages(), start=1):
                    page_text = page.get_text("text", sort=True) # Extract text, sorting helps reading order
                    if page_text:
                        # Simple chunking: Split page text by double newline, then maybe by single if needed
                        page_chunks = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                        if not page_chunks and page_text.strip(): # Fallback for pages without double newlines
                            page_chunks = [p.strip() for p in page_text.split('\n') if p.strip()]
                        # Optional: Add page number to chunk metadata later if needed
                        chunks.extend(page_chunks)
                doc.close()
                logger.info(f"PyMuPDF processed {page_num} pages, generated {len(chunks)} initial chunks.")
            except fitz.EmptyFileError:
                 logger.error(f"PyMuPDF error: File is empty or invalid PDF: {file_path}")
                 return [] # Return empty list for empty/invalid PDFs
            except Exception as pdf_exc: # Catch specific PyMuPDF errors if known, else general Exception
                 logger.error(f"PyMuPDF failed to process {file_path}: {pdf_exc}", exc_info=True)
                 raise # Re-raise to be caught by the endpoint handler

        elif file_type == FileType.TXT:
            logger.debug("Processing TXT file...")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # Simple split by paragraph (adjust as needed)
            chunks = [p.strip() for p in content.split('\n\n') if p.strip()]
            if not chunks and content.strip(): # If no double newlines, maybe split by single?
                chunks = [p.strip() for p in content.split('\n') if p.strip()]
                if not chunks and content.strip(): # Fallback to single large chunk
                    chunks = [content.strip()]
            logger.info(f"TXT processing generated {len(chunks)} chunks.")

        else:
            # Placeholder/Warning for other types
            logger.warning(f"Chunking not implemented for file type '{file_type.value}'. No chunks generated.")
            # Consider adding support using libraries like 'python-docx', 'beautifulsoup4', 'unstructured.io'
            # For now, return empty list for unsupported types.
            return []

        # Basic post-processing: remove very short chunks (optional)
        min_chunk_length = 10 # Example threshold
        final_chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_length]
        if len(chunks) != len(final_chunks):
             logger.debug(f"Removed {len(chunks) - len(final_chunks)} short chunks.")

        logger.info(f"Chunking generated {len(final_chunks)} final chunks for {file_type.value}.")
        return final_chunks

    except FileNotFoundError:
         logger.error(f"File not found during chunking: {file_path}")
         raise # Re-raise to be caught by the endpoint handler
    except Exception as e:
        logger.error(f"Chunking failed for {file_path} ({file_type.value}): {e}", exc_info=True)
        raise # Re-raise

async def embed_chunks(chunks: List[str]) -> List[Tuple[str, np.ndarray]]:
    """
    Generates ColBERT embeddings for text chunks using FastEmbed.
    Returns a list of tuples: (chunk_text, embedding_matrix).
    """     
    if not embedding_model:
         logger.error("Embedding model is not available. Cannot embed chunks.")
         raise RuntimeError("Embedding model failed to load or is not configured.")
    if not chunks:
        logger.warning("embed_chunks called with empty list. No embeddings to generate.")
        return []

    logger.info(f"Generating ColBERT embeddings for {len(chunks)} chunks using {embedding_model.model_name}...")
    try:
        # FastEmbed's embed method for late interaction models returns a generator of numpy arrays (matrices)
        embedding_generator = embedding_model.embed(chunks, batch_size=16) # Adjust batch_size as needed

        # Combine original chunks with their embeddings
        chunks_with_embeddings = list(zip(chunks, embedding_generator))

        # Validate shapes (optional but recommended)
        if chunks_with_embeddings:
             first_matrix = chunks_with_embeddings[0][1]
             if not isinstance(first_matrix, np.ndarray) or first_matrix.ndim != 2:
                  logger.error(f"Embedding model did not return expected 2D numpy arrays. Got type: {type(first_matrix)}")
                  raise ValueError("Embedding model output format is incorrect.")
             # Qdrant client expects specific dimension, check if it matches
             qdrant_dim = qdrant_client.vector_dim
             if first_matrix.shape[1] != qdrant_dim:
                   logger.error(f"Embedding dimension mismatch. Model produced {first_matrix.shape[1]}, Qdrant expects {qdrant_dim}.")
                   raise ValueError(f"Embedding dimension mismatch: Model={first_matrix.shape[1]}, Qdrant={qdrant_dim}")

        logger.info(f"Successfully generated embeddings for {len(chunks_with_embeddings)} chunks.")
        return chunks_with_embeddings # Return list of (text, matrix)
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
        raise # Re-raise the exception to be caught by the main endpoint handler

def reciprocal_rank_fusion(
    results_lists: List[List[Tuple[str, float]]],
    rrf_k: int = 60
) -> Dict[str, float]:
    """
    Performs Reciprocal Rank Fusion (RRF) on multiple ranked result lists.

    Args:
        results_lists: A list containing ranked lists. Each inner list
                       contains tuples of (doc_id, score). Scores are ignored,
                       only rank matters.
        rrf_k: The constant k used in the RRF formula (default: 60).

    Returns:
        A dictionary mapping doc_id to its RRF score, sorted by score descending.
    """
    if not results_lists:
        return {}

    rrf_scores = defaultdict(float)

    for results in results_lists:
        if not isinstance(results, list):
            logger.warning(f"RRF skipping invalid result type: {type(results)}")
            continue
        for rank, item in enumerate(results):
            if not isinstance(item, (tuple, list)) or len(item) < 1:
                 logger.warning(f"RRF skipping invalid item format: {item}")
                 continue
            doc_id = item[0]
            # RRF formula: 1 / (k + rank). Rank starts at 0, so add 1.
            rrf_scores[doc_id] += 1.0 / (rrf_k + (rank + 1))

    # Sort by score descending
    sorted_rrf_scores = dict(sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True))
    return sorted_rrf_scores

# --- Root Endpoint ---
@app.get("/")
async def read_root():
    """Provides a welcome message."""
    return {"message": "Welcome to the Document Indexing API. Use POST /index_document to upload a file."}

# --- File Upload and Indexing Endpoint ---
# Use 200 OK for potentially updating/indexing existing files, 201 for fully new
@app.post("/index_document", status_code=status.HTTP_200_OK)
async def index_document_flow(file: Annotated[UploadFile, File(description="The document file to be indexed.")]):
    """
    Accepts file upload. Checks duplicates across storage.
    If new or only in MinIO: classifies, extracts metadata, stores original,
    and indexes content/embeddings in OpenSearch/Qdrant.
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file or filename provided."
        )

    temp_file_path = None
    temp_dir = None
    doc_type = DocumentType.UNKNOWN # Default
    file_hash = ""
    final_status_code = status.HTTP_200_OK # Default to OK (update/already exists)

    try:
        # --- Initial Setup: Save temp file & calculate hash ---
        temp_dir = tempfile.mkdtemp(prefix="indexing_")
        temp_file_path = os.path.join(temp_dir, file.filename)
        logger.info(f"Saving uploaded file temporarily to: {temp_file_path}")
        file_size = 0
        try:
            with open(temp_file_path, "wb") as buffer:
                await file.seek(0)
                while content := await file.read(8192):
                    buffer.write(content)
                    file_size += len(content)
            logger.info(f"File '{file.filename}' ({file_size} bytes) saved temporarily.")
        except Exception as e:
             logger.error(f"Failed to save temporary file: {e}")
             raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to save uploaded file.")

        file_hash = calculate_file_hash(temp_file_path)
        logger.info(f"Calculated SHA256 hash: {file_hash}")

        # --- Duplicate Check Stage 1: MinIO ---
        logger.info(f"Checking if hash '{file_hash}' exists in MinIO bucket '{minio_client.bucket_name}'...")
        if not minio_client.connect(): # Ensure connection before check
             # Log error but don't fail immediately, maybe allow indexing if OS/Qdrant is up?
             # Or raise 503? Let's raise 503 for now.
             logger.error("MinIO connection failed before hash check.")
             raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Could not connect to MinIO storage.")

        hash_exists_in_minio = minio_client.hash_exists(file_hash)

        if hash_exists_in_minio:
            logger.info(f"Hash '{file_hash}' found in MinIO.")
            # --- Duplicate Check Stage 2: OpenSearch & Qdrant (All Types) ---
            logger.info("Checking for hash across all OpenSearch document type indices and Qdrant collections...")
            os_hash_found = False
            qdrant_hash_found = False

            # Run OpenSearch and Qdrant checks concurrently
            check_results = await asyncio.gather(
                # OpenSearch check (existing logic)
                *[opensearch_client.check_hash_exists(file_hash, type_enum.value) 
                  for type_enum in DocumentType if type_enum != DocumentType.UNKNOWN],
                # Add Qdrant check
                *[qdrant_client.check_hash_exists(file_hash, type_enum.value)
                  for type_enum in DocumentType if type_enum != DocumentType.UNKNOWN],
                return_exceptions=True
            )

            # Process OpenSearch results (first half of results)
            num_doc_types = sum(1 for type_enum in DocumentType if type_enum != DocumentType.UNKNOWN)
            for i, result in enumerate(check_results[:num_doc_types]):
                if isinstance(result, Exception):
                    logger.error(f"OpenSearch check failed for type {i}: {result}")
                    continue
                if result:
                    os_hash_found = True
                    doc_type = next(type_enum for j, type_enum in enumerate(DocumentType) 
                                   if type_enum != DocumentType.UNKNOWN and j == i)
                    break

            # Process Qdrant results (second half of results)
            for i, result in enumerate(check_results[num_doc_types:]):
                if isinstance(result, Exception):
                    logger.error(f"Qdrant check failed for type {i}: {result}")
                    continue
                if result:
                    qdrant_hash_found = True
                    break

            if os_hash_found and qdrant_hash_found:
                # Document exists in all systems (MinIO was checked earlier)
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Document with hash '{file_hash}' already exists and is indexed (found in OpenSearch type '{doc_type.value}' and Qdrant).",
                )
            elif os_hash_found or qdrant_hash_found:
                # Document exists in some but not all systems - proceed with indexing
                logger.warning(f"Document found in {'OpenSearch' if os_hash_found else 'Qdrant'} but not in {'Qdrant' if not qdrant_hash_found else 'OpenSearch'}. Proceeding with indexing to ensure consistency.")
                # Continue with classification and indexing steps

        else:
            # Hash NOT found in MinIO -> Completely new file
            logger.info(f"Hash '{file_hash}' not found in MinIO. This is a new document.")
            final_status_code = status.HTTP_201_CREATED # Mark as new resource creation

            # 5. Store Original File in MinIO (Do this early for new files)
            logger.info(f"Uploading original file to MinIO with hash '{file_hash}'...")
            # Prepare minimal metadata for MinIO (no classification/extraction yet)
            minio_base_metadata = {
                "original_filename": file.filename,
                "upload_timestamp": datetime.now(timezone.utc).isoformat()
            }
            # Convert to strings for MinIO
            minio_upload_metadata = {k: str(v) for k, v in minio_base_metadata.items() if v is not None}

            upload_success = minio_client.upload_document(
                file_path=temp_file_path,
                file_hash=file_hash,
                metadata=minio_upload_metadata # Use minimal metadata
            )
            if not upload_success:
                # If initial MinIO upload fails, we cannot proceed.
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload original file to MinIO for new document.")
            logger.info("New original file uploaded to MinIO successfully.")
            # Now proceed with classification, metadata extraction, and indexing (OS & Qdrant)


        # --- Indexing Steps (Executed for new files OR files found only in MinIO) ---

        # 1. Determine FileType (Do this first)
        file_type = get_file_type_from_upload(file)
        logger.info(f"Determined file type: {file_type.value}")

        # 2. Classify Document (Only if indexing is needed)
        logger.info("Preparing sample text for classification...")
        sample_text = "" # Initialize sample_text
        try:
            if file_type == FileType.PDF:
                logger.debug("Extracting text from PDF for classification using PyMuPDF...")
                extracted_text = ""
                try:
                    doc = fitz.open(temp_file_path)
                    # Extract text from first few pages until sample size is reached
                    for page_num, page in enumerate(doc.pages(), start=1):
                         page_text = page.get_text("text", sort=True).strip()
                         if page_text:
                              extracted_text += page_text + "\n\n" # Add separator
                         if len(extracted_text) >= MAX_CLASSIFICATION_SAMPLE_SIZE:
                              logger.debug(f"Reached classification sample size limit after page {page_num}.")
                              break
                    doc.close()
                    sample_text = extracted_text[:MAX_CLASSIFICATION_SAMPLE_SIZE] # Trim to exact limit
                    if not sample_text:
                         logger.warning(f"No text could be extracted from the first pages of PDF: {file.filename}")
                except fitz.EmptyFileError:
                     logger.error(f"PyMuPDF error: File is empty or invalid PDF: {file.filename}")
                     # Decide how to handle: maybe raise, or classify as UNKNOWN? Let's raise for now.
                     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid or empty PDF file: {file.filename}")
                except Exception as pdf_exc:
                     logger.error(f"PyMuPDF failed during text extraction for classification: {pdf_exc}", exc_info=True)
                     # Re-raise or handle as internal error
                     raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to extract text from PDF for classification.")

            elif file_type == FileType.TXT:
                 logger.debug("Reading text from TXT file for classification...")
                 try:
                      with open(temp_file_path, 'r', encoding='utf-8') as f:
                           sample_text = f.read(MAX_CLASSIFICATION_SAMPLE_SIZE)
                 except UnicodeDecodeError:
                      logger.warning(f"Could not decode TXT file {file.filename} as UTF-8. Trying with errors ignored.")
                      # Fallback: attempt decoding with error handling
                      with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                           sample_text = f.read(MAX_CLASSIFICATION_SAMPLE_SIZE)
                 except Exception as txt_exc:
                      logger.error(f"Failed to read TXT file for classification: {txt_exc}", exc_info=True)
                      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to read text file for classification.")

            # Add elif blocks here for other supported file types (DOCX, HTML, etc.) if needed
            # Use libraries like python-docx, beautifulsoup4, unstructured.io

            else:
                 # Fallback for unsupported types (or keep the old byte reading method if preferred as last resort)
                 logger.warning(f"Classification for file type '{file_type.value}' relies on basic byte decoding. Results may be inaccurate.")
                 try:
                      with open(temp_file_path, 'rb') as f:
                           sample_bytes = f.read(MAX_CLASSIFICATION_SAMPLE_SIZE)
                      sample_text = sample_bytes.decode('utf-8', errors='ignore')
                 except Exception as fallback_exc:
                      logger.error(f"Failed fallback read for classification: {fallback_exc}", exc_info=True)
                      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to read file content for classification.")

            # --- Now use the extracted/read sample_text ---
            if not sample_text.strip():
                 logger.error(f"File '{file.filename}' resulted in empty sample text for classification.")
                 # Handle empty content: classify as UNKNOWN or raise error?
                 raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot classify empty file content.")

            logger.debug(f"Passing sample text to classifier (first 100 chars): '{sample_text[:100]}...'") # Log beginning of sample
            doc_type = await classify(sample_text) # Use the correctly prepared sample_text
            logger.info(f"Classified document as: {doc_type.value}")

            if doc_type == DocumentType.UNKNOWN:
                 raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document type could not be determined or is unsupported. Cannot index.")

        except HTTPException as http_exc:
             raise http_exc # Re-raise if classification itself raised HTTP error
        except Exception as e:
             logger.error(f"Error during classification phase: {e}", exc_info=True)
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed during document classification: {e}")


        # 4. Extract Metadata
        logger.info("Reading sample for metadata extraction...")
        extracted_metadata = {}
        metadata_sample = "" # Initialize sample text
        try:
            # Extract text based on file_type for metadata sample
            if file_type == FileType.PDF:
                logger.debug("Extracting text from PDF for metadata using PyMuPDF...")
                extracted_text = ""
                try:
                    doc = fitz.open(temp_file_path)
                    for page_num, page in enumerate(doc.pages(), start=1):
                        page_text = page.get_text("text", sort=True).strip()
                        if page_text:
                            extracted_text += page_text + "\n\n" # Add separator
                        if len(extracted_text) >= MAX_METADATA_SAMPLE_SIZE:
                            logger.debug(f"Reached metadata sample size limit after page {page_num}.")
                            break
                    doc.close()
                    metadata_sample = extracted_text[:MAX_METADATA_SAMPLE_SIZE] # Trim to exact limit
                    if not metadata_sample:
                        logger.warning(f"No text could be extracted from PDF for metadata: {file.filename}")
                except fitz.EmptyFileError:
                     logger.error(f"PyMuPDF error (metadata): File is empty or invalid PDF: {file.filename}")
                     # Decide if this should be fatal, or just skip metadata. Skipping for now.
                except Exception as pdf_exc:
                     logger.warning(f"PyMuPDF failed during text extraction for metadata: {pdf_exc}. Proceeding without extracted metadata.", exc_info=True)

            elif file_type == FileType.TXT:
                 logger.debug("Reading text from TXT file for metadata...")
                 try:
                      with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                           metadata_sample = f.read(MAX_METADATA_SAMPLE_SIZE)
                 except Exception as txt_exc:
                      logger.warning(f"Failed to read TXT file for metadata: {txt_exc}. Proceeding without extracted metadata.", exc_info=True)

            # Add elif blocks here for other supported file types if needed

            else:
                 # Fallback for unsupported types (maybe try byte decoding as last resort?)
                 logger.warning(f"Metadata extraction for file type '{file_type.value}' relies on basic byte decoding. Results may be inaccurate.")
                 try:
                      with open(temp_file_path, 'rb') as f:
                           metadata_bytes = f.read(MAX_METADATA_SAMPLE_SIZE)
                      metadata_sample = metadata_bytes.decode('utf-8', errors='ignore')
                 except Exception as fallback_exc:
                      logger.warning(f"Failed fallback read for metadata: {fallback_exc}. Proceeding without extracted metadata.", exc_info=True)

            # --- Use the extracted/read metadata_sample ---
            if not metadata_sample.strip():
                 logger.warning("Could not read sample for metadata extraction. Proceeding without.")
            else:
                 logger.debug(f"Passing metadata sample to extractor (first 100 chars): '{metadata_sample[:100]}...'")
                 metadata_model = await extract_metadata(metadata_sample, doc_type)
                 if metadata_model:
                      extracted_metadata = metadata_model.model_dump(exclude_unset=True)
                      logger.info(f"Extracted metadata: {extracted_metadata}")
                 else:
                      logger.warning("Metadata extraction returned None.")

        except Exception as e:
             # Catch any unexpected error during the extraction process itself
             logger.warning(f"Metadata extraction phase failed unexpectedly: {e}. Proceeding without extracted metadata.", exc_info=True)
             extracted_metadata = {} # Ensure it's empty

        # Combine metadata for indexing (OS & Qdrant)
        # Base info + classified type + extracted info
        indexing_metadata = {
            "original_filename": file.filename,
            "file_size_bytes": str(file_size), # Stringify
            "content_type": file.content_type or 'application/octet-stream',
            "doc_type": doc_type.value, # Use the classified type
            "file_type": file_type.value, # Add file_type to metadata if useful
            **extracted_metadata # Merge extracted data
        }

        # 6. Chunk Document (Using the determined file_type)
        logger.info(f"Chunking document ({file_type.value})...")
        text_chunks = await chunk_document(temp_file_path, file_type) # <--- Pass file_type here
        if not text_chunks:
            logger.warning("No text chunks were generated. Skipping OS/Qdrant indexing.")
            # File is in MinIO, but cannot be chunked/indexed further. Return success.
            # Message should reflect this.
            return {
                "message": f"File '{file.filename}' saved to storage, but no content chunks generated for indexing.",
                "filename": file.filename,
                "file_hash": file_hash,
                "detected_doc_type": doc_type.value, # Still useful info
                "status": "Stored but not indexed (no chunks).",
            }
        else:
            logger.info(f"Generated {len(text_chunks)} text chunks.")

             # 7. Generate Embeddings (Pass doc_type as before, embedding model might need it)
            logger.info("Generating embeddings...")
            chunks_with_embeddings = await embed_chunks(text_chunks)
            # embed_chunks raises RuntimeError or ValueError on failure, caught by main handler

            logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks.")

            # 8. Index in OpenSearch
            logger.info(f"Indexing text chunks in OpenSearch (Index: {opensearch_client.get_index_name(doc_type.value)})...")
            os_mappings = get_opensearch_mappings(doc_type)
            # Ensure OpenSearch client is connected before indexing
            if not await opensearch_client.connect():
                logger.error("OpenSearch connection failed before indexing.")
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Could not connect to OpenSearch.")

            os_index_success = await opensearch_client.index_document(
                chunks=text_chunks,
                metadata=indexing_metadata, # Use combined metadata
                file_hash=file_hash,
                doc_type=doc_type.value,
                mappings=os_mappings # Pass mappings for index creation check
            )
            if not os_index_success:
                # Consider cleanup? Maybe not automatically. Log error.
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to index document chunks in OpenSearch.")
            logger.info("Indexing in OpenSearch successful.")

            # 9. Index in Qdrant
            logger.info(f"Indexing embeddings in Qdrant (Collection: {qdrant_client.get_collection_name(doc_type.value)})...")
             # Ensure Qdrant client is connected before indexing
            if not await qdrant_client.connect():
                logger.error("Qdrant connection failed before indexing.")
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Could not connect to Qdrant.")

            # Prepare points with flattened vectors
            points_to_upload = []
            for i, (chunk_text, embedding_matrix) in enumerate(chunks_with_embeddings):
                 if not isinstance(embedding_matrix, np.ndarray) or embedding_matrix.ndim != 2:
                     logger.warning(f"Skipping invalid embedding matrix for chunk {i}, hash {file_hash}")
                     continue

                 # Combine chunk-specific info with overall metadata
                 payload = indexing_metadata.copy()
                 payload.update({
                     qdrant_client.settings.FIELD_FILE_HASH: file_hash,
                     qdrant_client.settings.FIELD_CHUNK_INDEX: i,
                     qdrant_client.settings.FIELD_TOTAL_CHUNKS: len(chunks_with_embeddings),
                     qdrant_client.settings.FIELD_TIMESTAMP: datetime.now(timezone.utc).isoformat(),
                     qdrant_client.settings.FIELD_CONTENT: chunk_text # Include chunk text if needed
                 })
                 # point_id = f"{file_hash}_{i}" # <-- INVALID ID format
                 # Use integer chunk index as ID (valid format for Qdrant)
                 # File hash and chunk index are in payload for linking.

                 points_to_upload.append(qdrant_models.PointStruct(
                    id=i, # <-- USE INTEGER INDEX i AS ID
                    vector=embedding_matrix.tolist(),
                    payload=payload
                 ))

            if not points_to_upload:
                 logger.warning(f"No valid points with embeddings generated for Qdrant indexing for hash {file_hash}.")
                 # Don't raise error, just skip Qdrant indexing if no valid points
            else:
                 qdrant_index_success = await qdrant_client.index_document_points(
                     points=points_to_upload,
                     doc_type=doc_type.value
                 )

                 if not qdrant_index_success:
                     # Consider cleanup? Log error.
                     raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to index document embeddings in Qdrant.")
                 logger.info("Indexing in Qdrant successful.")


        # --- Final Response ---
        response_message = f"File '{file.filename}' processed and indexed successfully as type '{doc_type.value}'."
        if final_status_code == status.HTTP_200_OK and hash_exists_in_minio:
             response_message = f"Existing file '{file.filename}' (found in storage) was successfully indexed as type '{doc_type.value}'."

        # Manually set response status code *before* returning dictionary
        # This requires using the Response object directly or a more complex setup.
        # For simplicity here, we return a standard dict and note the intended status code.
        logger.info(f"Intended status code: {final_status_code}")

        return {
            "message": response_message,
            "filename": file.filename,
            "file_hash": file_hash,
            "detected_doc_type": doc_type.value,
            "metadata_indexed": indexing_metadata, # Return the metadata used for indexing
            "chunks_indexed": len(text_chunks) if text_chunks else 0,
            "final_status": "Created" if final_status_code == 201 else "Indexed" # Indicate action
        }

    except HTTPException as http_exc:
         # Log HTTP exceptions before re-raising
         logger.warning(f"HTTP Exception {http_exc.status_code}: {http_exc.detail}")
         raise http_exc
    except ConnectionError as conn_err:
         logger.error(f"Connection Error: {conn_err}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"A required storage or indexing service is unavailable: {conn_err}")
    except FileNotFoundError:
         logger.error(f"Temporary file not found during processing: {temp_file_path}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error: Processing file not found.")
    except RuntimeError as rt_err: # Catch specific errors like embedding model not loaded
         logger.error(f"Runtime error during processing: {rt_err}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal processing error: {rt_err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred during processing: {e}",
        )
    finally:
        # --- Cleanup ---
        if file:
            await file.close()
            logger.debug(f"Closed upload file stream: {file.filename}")
        # Clean up the temporary directory and its contents
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Removed temporary directory: {temp_dir}")
            except Exception as cleanup_e:
                 logger.error(f"Error cleaning up temporary directory {temp_dir}: {cleanup_e}")


# --- Search Endpoint ---
@app.get("/search")
async def search_documents(
    query: str,
    k: int = 10, # Number of results to return
    doc_type: Optional[str] = None, # Optional: Filter by specific document type
    rrf_k: int = 60, # Constant for RRF calculation
):
    """
    Searches for documents using a hybrid approach (text + vector),
    combines results using Reciprocal Rank Fusion (RRF), and returns
    ranked results including chunk content and metadata.
    """
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query parameter cannot be empty.")
    if not embedding_model:
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Embedding model not available.")

    initial_fetch_k = max(k * 3, 50)
    logger.info(f"Received search query: '{query}', k={k}, doc_type={doc_type}, initial_fetch_k={initial_fetch_k}, rrf_k={rrf_k}")

    # --- 1. Embed the Query for Qdrant ---
    try:
        logger.debug("Embedding query for vector search...")
        query_embedding_generator = embedding_model.embed([query], is_query=True)
        query_embedding_matrix = next(query_embedding_generator, None)
        if query_embedding_matrix is None or not isinstance(query_embedding_matrix, np.ndarray):
             raise ValueError("Failed to generate valid query embedding matrix.")
        query_vector = query_embedding_matrix.flatten().tolist()
        logger.debug(f"Query vector generated (shape: {query_embedding_matrix.shape}).")
    except Exception as e:
        logger.error(f"Failed to embed query '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to embed query: {e}")

    # --- 2. Perform Searches Concurrently and Collect Payloads ---
    all_payloads = {} # Store payload by chunk_id {chunk_id: payload}
    opensearch_results_for_rrf = []
    qdrant_results_for_rrf = []

    try:
        logger.info(f"Performing concurrent search in OpenSearch and Qdrant (fetching top {initial_fetch_k})...")
        await asyncio.gather(opensearch_client.connect(), qdrant_client.connect())

        # IMPORTANT ASSUMPTION: Clients now return List[Tuple[chunk_id, score, payload]]
        # where payload contains 'content' and metadata fields.
        opensearch_results_task = opensearch_client.search_documents(
            query_text=query,
            k=initial_fetch_k,
            doc_type=doc_type
        )
        qdrant_results_task = qdrant_client.search_documents(
            query_vector=query_vector,
            k=initial_fetch_k,
            doc_type=doc_type
        )

        os_results_raw, qdrant_results_raw = await asyncio.gather(
            opensearch_results_task,
            qdrant_results_task,
            return_exceptions=True
        )

        # Process OpenSearch results
        if isinstance(os_results_raw, Exception):
            logger.error(f"OpenSearch search failed: {os_results_raw}", exc_info=os_results_raw)
        elif os_results_raw:
            logger.info(f"OpenSearch returned {len(os_results_raw)} raw results.")
            for result in os_results_raw:
                if len(result) == 3: # Expecting (chunk_id, score, payload)
                    chunk_id, score, payload = result
                    opensearch_results_for_rrf.append((chunk_id, score))
                    if chunk_id not in all_payloads: # Keep the first seen payload
                         all_payloads[chunk_id] = payload
                else:
                    logger.warning(f"Unexpected result format from OpenSearch: {result}")

        # Process Qdrant results
        if isinstance(qdrant_results_raw, Exception):
            logger.error(f"Qdrant search failed: {qdrant_results_raw}", exc_info=qdrant_results_raw)
        elif qdrant_results_raw:
            logger.info(f"Qdrant returned {len(qdrant_results_raw)} raw results.")
            for result in qdrant_results_raw:
                if len(result) == 3: # Expecting (chunk_id, score, payload)
                    chunk_id, score, payload = result
                    qdrant_results_for_rrf.append((chunk_id, score))
                    if chunk_id not in all_payloads: # Keep the first seen payload
                        all_payloads[chunk_id] = payload
                else:
                     logger.warning(f"Unexpected result format from Qdrant: {result}")

        if not opensearch_results_for_rrf and not qdrant_results_for_rrf:
             logger.error("Both OpenSearch and Qdrant searches failed or returned no results.")
             # Decide if 404 or 503 is appropriate. If searches failed, 503. If no results, return empty list.
             # Let's return empty list for now, assuming no results found.
             return {"query": query, "retrieved_count": 0, "results": []}
             # raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Both search backends failed.")

    except Exception as e:
        logger.error(f"Error during search execution: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error performing search: {e}")

    # --- 3. Apply Reciprocal Rank Fusion ---
    logger.info("Applying Reciprocal Rank Fusion...")
    # Pass lists of (chunk_id, score) to RRF
    fused_scores = reciprocal_rank_fusion([opensearch_results_for_rrf, qdrant_results_for_rrf], rrf_k=rrf_k)

    if not fused_scores:
        logger.info("No results found after fusion.")
        return {"query": query, "retrieved_count": 0, "results": []}

    # Get the top K chunk IDs from the fused results
    top_k_fused_ids = list(fused_scores.keys())[:k]
    logger.info(f"Top {len(top_k_fused_ids)} chunk IDs after RRF: {top_k_fused_ids}")

    # --- 4. Construct Final Results with Content and Metadata ---
    final_results = []
    for chunk_id in top_k_fused_ids:
        rrf_score = fused_scores[chunk_id]
        payload = all_payloads.get(chunk_id)

        if payload:
            # Assuming payload contains 'content' and other metadata fields
            # Adapt the keys ('content', 'metadata_field_1', etc.) based on actual payload structure
            final_results.append({
                "id": chunk_id,
                "rrf_score": rrf_score,
                "content": payload.get('content', None), # Get chunk content
                "metadata": {k: v for k, v in payload.items() if k != 'content'} # Get all other fields as metadata
            })
        else:
            # This shouldn't happen if the payload was stored correctly, but handle defensively
            logger.warning(f"Payload not found for chunk_id {chunk_id} after RRF. Skipping.")
            final_results.append({
                 "id": chunk_id,
                 "rrf_score": rrf_score,
                 "content": None,
                 "metadata": None # Indicate data unavailable
             })

    # --- 5. Return Final Ranked Results ---
    return {
        "query": query,
        "retrieved_count": len(final_results),
        "results": final_results
    }


# --- Run the App (for local development) ---
if __name__ == "__main__":
    if not embedding_model:
         logger.error("Embedding model failed to load. API cannot function properly.")
         # Optionally exit sys.exit(1)

    print(f"Starting server at http://127.0.0.1:8000")
    try:
        uvicorn.run(
            "__main__:app",
            host="127.0.0.1",
            port=8000,
            reload=True, # Be mindful of model reloading issues
            # Consider reload_dirs=["."] or similar if structure changes
        )
    finally:
        # Cleanup temp base dir if needed, mkdtemp usually handles its own path well
        pass