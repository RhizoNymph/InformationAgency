# main.py
import uvicorn
import shutil
import hashlib
import asyncio
import tempfile
import numpy as np 
from datetime import datetime, timezone


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
from indexing.models import DocumentType, models # Assuming DocumentType enum is in models.py
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
    embedding_model = None # Or raise ImportError("fastembed required")
except Exception as e:
     print(f"Error loading embedding model '{colbert_model_name}': {e}")
     embedding_model = None # Handle loading errors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Configuration (Optional - can be moved to settings) ---
TEMP_UPLOAD_DIR = tempfile.mkdtemp(prefix="fastapi_uploads_")
MAX_CLASSIFICATION_SAMPLE_SIZE = 5000
MAX_METADATA_SAMPLE_SIZE = 20000


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

def calculate_file_hash(file_path: str) -> str:
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as file:
            while chunk := file.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
         logger.error(f"File not found when calculating hash: {file_path}")
         raise
    except Exception as e:
         logger.error(f"Error calculating hash for {file_path}: {e}")
         raise


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


async def chunk_document(file_path: str, doc_type: DocumentType) -> list[str]:
    """Chunks the document text based on its type."""
    # Placeholder: Implement document reading (e.g., using PyPDF2, python-docx, unstructured.io)
    # and chunking strategy (e.g., fixed size, semantic chunking).
    logger.warning(f"Using placeholder document chunking for {doc_type.value}")
    try:
        # Example: Basic text file reading and simple splitting
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # Simple split by paragraph (adjust as needed)
        chunks = [p.strip() for p in content.split('\n\n') if p.strip()]
        if not chunks and content: # If no double newlines, maybe split by single? Or just one big chunk?
             chunks = [content.strip()]
        logger.info(f"Placeholder chunking generated {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Placeholder chunking failed for {file_path}: {e}")
        return []

async def embed_chunks(chunks: List[str], doc_type: DocumentType) -> List[Tuple[str, np.ndarray]]:
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
            # --- Duplicate Check Stage 2: OpenSearch (All Types) ---
            logger.info("Checking for hash across all OpenSearch document type indices...")
            os_hash_found = False
            # Iterate through all known document types (excluding UNKNOWN)
            for type_enum in DocumentType:
                 if type_enum == DocumentType.UNKNOWN:
                      continue
                 logger.debug(f"Checking OpenSearch index for type: {type_enum.value}...")
                 if await opensearch_client.check_hash_exists(file_hash, type_enum.value):
                      logger.info(f"Hash '{file_hash}' found in OpenSearch index for type '{type_enum.value}'. Document already indexed.")
                      os_hash_found = True
                      # Store the found type if needed for the response
                      doc_type = type_enum
                      break # Found it, no need to check other types

            if os_hash_found:
                # Already fully indexed (exists in MinIO and OpenSearch)
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Document with hash '{file_hash}' already exists and is indexed (found in OpenSearch type '{doc_type.value}').",
                )
            else:
                # Exists in MinIO, but NOT in OpenSearch -> Needs indexing
                logger.info(f"Hash '{file_hash}' exists in MinIO but not in any OpenSearch index. Proceeding with classification and indexing.")
                # Proceed to classification, metadata extraction, and indexing (OS & Qdrant)
                # MinIO upload is skipped as it already exists.
                # Status code remains 200 OK as we are 'updating' by indexing.
                pass # Continue to the indexing steps below

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

        # 2. Classify Document (Only if indexing is needed)
        logger.info("Reading sample for classification...")
        try:
            # Ensure file is readable, use 'rb' if format is unknown, decode carefully
            with open(temp_file_path, 'rb') as f:
                 sample_bytes = f.read(MAX_CLASSIFICATION_SAMPLE_SIZE)
            sample_text = sample_bytes.decode('utf-8', errors='ignore') # Decode for LLM

            if not sample_text.strip():
                 logger.error("File is empty or could not be read for classification.")
                 raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Cannot classify empty or unreadable file content.")
            doc_type = await classify(sample_text) # Overwrites UNKNOWN default
            logger.info(f"Classified document as: {doc_type.value}")
            if doc_type == DocumentType.UNKNOWN:
                 # Decide how to handle: raise error or index into a default 'unknown' index?
                 # Let's raise an error for now, requiring classification.
                 raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document type could not be determined or is unsupported. Cannot index.")
        except HTTPException as http_exc:
             raise http_exc # Re-raise if classification itself raised HTTP error
        except Exception as e:
             logger.error(f"Error during classification: {e}", exc_info=True)
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to classify document: {e}")


        # 4. Extract Metadata (Only if indexing is needed)
        logger.info("Reading sample for metadata extraction...")
        extracted_metadata = {}
        try:
            # Adjust reading based on how metadata_extractor expects input
            with open(temp_file_path, 'rb') as f:
                 metadata_bytes = f.read(MAX_METADATA_SAMPLE_SIZE)
            metadata_sample = metadata_bytes.decode('utf-8', errors='ignore')

            if not metadata_sample.strip():
                 logger.warning("Could not read sample for metadata extraction. Proceeding without.")
            else:
                 metadata_model = await extract_metadata(metadata_sample, doc_type)
                 if metadata_model:
                      extracted_metadata = metadata_model.model_dump(exclude_unset=True)
                 else:
                      logger.warning("Metadata extraction returned None.")
            logger.info(f"Extracted metadata: {extracted_metadata}")
        except Exception as e:
             logger.warning(f"Metadata extraction failed: {e}. Proceeding without extracted metadata.", exc_info=True)
             # Keep extracted_metadata as {}

        # Combine metadata for indexing (OS & Qdrant)
        # Base info + classified type + extracted info
        indexing_metadata = {
            "original_filename": file.filename,
            "file_size_bytes": str(file_size), # Stringify
            "content_type": file.content_type or 'application/octet-stream',
            "doc_type": doc_type.value, # Use the classified type
            **extracted_metadata # Merge extracted data
        }

        # 6. Chunk Document
        logger.info("Chunking document...")
        text_chunks = await chunk_document(temp_file_path, doc_type)
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

             # 7. Generate Embeddings
            logger.info("Generating embeddings...")
            chunks_with_embeddings = await embed_chunks(text_chunks, doc_type)
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
                 point_id = f"{file_hash}_{i}"
                 # Explicit flatten needed based on PointStruct definition
                 flat_vector = embedding_matrix.flatten().tolist()

                 points_to_upload.append(qdrant_models.PointStruct(
                    id=point_id,
                    vector=flat_vector,
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
         raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error: Processing file not found.")
    except RuntimeError as rt_err: # Catch specific errors like embedding model not loaded
         logger.error(f"Runtime error during processing: {rt_err}", exc_info=True)
         raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal processing error: {rt_err}")
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
    # fetch_k: int = 50, # Number of results to fetch initially from each source
    rrf_k: int = 60, # Constant for RRF calculation
    # include_metadata: bool = True # Flag to fetch full metadata
):
    """
    Searches for documents using a hybrid approach (text + vector)
    and combines results using Reciprocal Rank Fusion (RRF).
    """
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query parameter cannot be empty.")
    if not embedding_model:
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Embedding model not available.")

    # Define how many results to fetch initially from each source
    # Fetching more than 'k' initially gives RRF more data to work with
    initial_fetch_k = max(k * 3, 50) # Fetch more results initially, e.g., 3*k or 50

    logger.info(f"Received search query: '{query}', k={k}, doc_type={doc_type}, initial_fetch_k={initial_fetch_k}, rrf_k={rrf_k}")

    # --- 1. Embed the Query for Qdrant ---
    try:
        logger.debug("Embedding query for vector search...")
        # FastEmbed ColBERT might just use embed, check its specific API if needed
        # It returns a generator, get the first (and only) item
        query_embedding_generator = embedding_model.embed([query], is_query=True) # Indicate it's a query if API supports it
        query_embedding_matrix = next(query_embedding_generator, None)

        if query_embedding_matrix is None or not isinstance(query_embedding_matrix, np.ndarray):
             raise ValueError("Failed to generate valid query embedding matrix.")

        # Flatten for Qdrant search (assuming client expects flat list)
        query_vector = query_embedding_matrix.flatten().tolist()
        logger.debug(f"Query vector generated (shape: {query_embedding_matrix.shape}).")
    except Exception as e:
        logger.error(f"Failed to embed query '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to embed query: {e}")

    # --- 2. Perform Searches Concurrently ---
    try:
        logger.info(f"Performing concurrent search in OpenSearch and Qdrant (fetching top {initial_fetch_k})...")
        # Ensure clients are connected (add connect calls if client doesn't handle it internally on search)
        await asyncio.gather(opensearch_client.connect(), qdrant_client.connect())

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
            return_exceptions=True # Allow one search to fail without stopping the other
        )

        # Handle potential errors from searches
        opensearch_results = []
        if isinstance(os_results_raw, Exception):
            logger.error(f"OpenSearch search failed: {os_results_raw}", exc_info=os_results_raw)
            # Depending on requirements, could raise 500 or proceed with Qdrant results only
        elif os_results_raw:
             opensearch_results = os_results_raw
             logger.info(f"OpenSearch returned {len(opensearch_results)} results.")

        qdrant_results = []
        if isinstance(qdrant_results_raw, Exception):
            logger.error(f"Qdrant search failed: {qdrant_results_raw}", exc_info=qdrant_results_raw)
            # Depending on requirements, could raise 500 or proceed with OS results only
        elif qdrant_results_raw:
            qdrant_results = qdrant_results_raw
            logger.info(f"Qdrant returned {len(qdrant_results)} results.")

        if not opensearch_results and not qdrant_results:
             # If both failed, raise error
             logger.error("Both OpenSearch and Qdrant searches failed.")
             raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Both search backends failed.")

    except Exception as e:
        logger.error(f"Error during search execution: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error performing search: {e}")


    # --- 3. Apply Reciprocal Rank Fusion ---
    logger.info("Applying Reciprocal Rank Fusion...")
    # Assume client search methods return List[Tuple[doc_id, score]]
    fused_scores = reciprocal_rank_fusion([opensearch_results, qdrant_results], rrf_k=rrf_k)

    if not fused_scores:
        logger.info("No results found after fusion.")
        return {"query": query, "results": []}

    # Get the top K document IDs from the fused results
    top_k_fused_ids = list(fused_scores.keys())[:k]
    logger.info(f"Top {len(top_k_fused_ids)} results after RRF: {top_k_fused_ids}")


    # --- 4. Fetch Metadata/Content for Top K Results (Optional but Recommended) ---
    final_results = []
    if top_k_fused_ids: # and include_metadata:
        logger.info("Fetching details for top results from OpenSearch...")
        try:
            # Assume OpenSearch client has a method to get docs by ID
            # This might require searching specific indices or all relevant ones
            fetched_details = await opensearch_client.get_documents_by_ids(
                doc_ids=top_k_fused_ids,
                doc_type=doc_type # Pass doc_type to potentially narrow down indices
            )
            # fetched_details should ideally be a dict: {doc_id: payload}

            # Combine details with RRF scores, maintaining the RRF rank order
            for doc_id in top_k_fused_ids:
                detail = fetched_details.get(doc_id)
                if detail:
                    final_results.append({
                        "id": doc_id,
                        "rrf_score": fused_scores[doc_id],
                        "metadata": detail # The payload returned by get_documents_by_ids
                    })
                else:
                     # Document ID from RRF was not found in OpenSearch fetch? Log warning.
                     logger.warning(f"Could not fetch details for document ID from RRF: {doc_id}")
                     # Optionally include anyway with just score?
                     final_results.append({
                         "id": doc_id,
                         "rrf_score": fused_scores[doc_id],
                         "metadata": None # Indicate details unavailable
                     })

        except Exception as e:
            logger.error(f"Failed to fetch details for top results: {e}", exc_info=True)
            # Fallback: return only IDs and scores if fetching fails
            final_results = [{"id": doc_id, "rrf_score": fused_scores[doc_id], "metadata": None} for doc_id in top_k_fused_ids]

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