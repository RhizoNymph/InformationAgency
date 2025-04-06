# storage/qdrant_client.py
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import numpy as np # Added for type hinting embedding matrices

# Use Qdrant client, prefer async for consistency
from qdrant_client import QdrantClient as SyncQdrantClient # Sync client for specific tasks if needed
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# Import settings from the sibling settings.py file
from . import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QdrantColbertClient:
    """
    An asynchronous client for interacting with Qdrant, specifically designed for
    indexing pre-computed ColBERT multi-vector embeddings for document chunks.
    """
    def __init__(self):
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_prefix = settings.QDRANT_COLLECTION_PREFIX
        self.vector_dim = settings.COLBERT_VECTOR_DIM
        self.vector_distance = self._parse_distance(settings.COLBERT_VECTOR_DISTANCE)
        self.multivector_comparator = self._parse_comparator(settings.COLBERT_COMPARATOR)
        self.settings = settings # Store settings for easy access

        # Connection parameters
        self._connection_args = {}
        if settings.QDRANT_LOCATION == ":memory:":
             logger.info("Initializing Qdrant client in :memory: mode.")
             # Use location=None or explicit host/port for async in-memory setup
             self._connection_args['location'] = ":memory:" # Try passing directly if supported by async version
             # Or handle via host/port if the above doesn't work
             # self._connection_args['host'] = 'localhost'
             # self._connection_args['port'] = settings.QDRANT_GRPC_PORT
        elif settings.QDRANT_LOCATION: # If URL is provided
             self._connection_args['url'] = settings.QDRANT_LOCATION
             self._connection_args['prefer_grpc'] = True
             if settings.QDRANT_API_KEY:
                 self._connection_args['api_key'] = settings.QDRANT_API_KEY
        else: # Default to host/port
            self._connection_args['host'] = settings.QDRANT_HOST
            self._connection_args['port'] = settings.QDRANT_GRPC_PORT # Use gRPC port for async
            if settings.QDRANT_API_KEY:
                 self._connection_args['api_key'] = settings.QDRANT_API_KEY
            # Handle TLS specifically for host/port if URL wasn't given
            if settings.QDRANT_USE_TLS:
                 logger.warning("QDRANT_USE_TLS=true with host/port might require https=True argument or manual port adjustment (e.g., 443). Check qdrant-client docs.")
                 # self._connection_args['https'] = True # Example if client supports it

        logger.info(f"QdrantColbertClient initialized. Target: {self._connection_args}")
        logger.info(f"ColBERT settings: Dim={self.vector_dim}, Dist={self.vector_distance}, Comp={self.multivector_comparator}")


    @staticmethod
    def _parse_distance(distance_str: str) -> models.Distance:
        """Converts distance string from settings to Qdrant models.Distance."""
        distance_str = distance_str.strip().upper()
        if distance_str == "COSINE":
            return models.Distance.COSINE
        elif distance_str == "DOT":
            return models.Distance.DOT
        elif distance_str == "EUCLID":
            return models.Distance.EUCLID
        else:
            logger.warning(f"Unsupported distance metric '{distance_str}'. Defaulting to COSINE.")
            return models.Distance.COSINE

    @staticmethod
    def _parse_comparator(comparator_str: str) -> models.MultiVectorComparator:
        """Converts comparator string from settings to Qdrant models.MultiVectorComparator."""
        comparator_str = comparator_str.strip().upper()
        if comparator_str == "MAXSIM":
            return models.MultiVectorComparator.MAX_SIM
        else:
            logger.warning(f"Unsupported multi-vector comparator '{comparator_str}'. Defaulting to MAX_SIM.")
            return models.MultiVectorComparator.MAX_SIM


    async def connect(self) -> bool:
        """Establishes an asynchronous connection to the Qdrant instance."""
        if self.client:
            try:
                 await self.client.get_collections() # Simple check
                 logger.debug("Qdrant connection appears active.")
                 return True
            except Exception as e:
                 logger.warning(f"Qdrant connection check failed: {e}. Reconnecting...")
                 await self.close()

        try:
            logger.info(f"Attempting to connect to Qdrant with args: {self._connection_args}")
            self.client = AsyncQdrantClient(**self._connection_args)
            await self.client.get_collections()
            logger.info("Successfully connected to Qdrant.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}", exc_info=True)
            self.client = None
            return False

    async def close(self):
        """Closes the asynchronous Qdrant connection."""
        if self.client:
            try:
                await self.client.close()
                logger.info("Qdrant connection closed.")
            except Exception as e:
                logger.error(f"Error closing Qdrant connection: {e}", exc_info=True)
            finally:
                self.client = None

    async def _ensure_connected(self):
        """Internal helper to ensure the client is connected."""
        if not self.client:
            logger.warning("Client not connected. Attempting to connect.")
            if not await self.connect():
                raise ConnectionError("Failed to establish connection with Qdrant.")
        # Optional: Add health check before each operation if needed


    def get_collection_name(self, doc_type: str) -> str:
        """Generates the full collection name."""
        if not doc_type:
             raise ValueError("doc_type cannot be empty")
        return f"{self.collection_prefix}_{doc_type.lower().strip()}"

    async def create_collection_if_not_exists(self, doc_type: str):
        """
        Creates a Qdrant collection configured for ColBERT multi-vectors if it doesn't exist.
        Also sets up payload indexing for the file_hash field.
        """
        await self._ensure_connected()
        collection_name = self.get_collection_name(doc_type)

        try:
            collections_response = await self.client.get_collections()
            existing_collections = {col.name for col in collections_response.collections}

            if collection_name not in existing_collections:
                logger.info(f"Collection '{collection_name}' does not exist. Creating...")
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,             # Dimension of each vector in the matrix [cite: 29]
                        distance=self.vector_distance,    # Distance for comparing individual vectors [cite: 31]
                        multivector_config=models.MultiVectorConfig( # Configure for multi-vector [cite: 28]
                            comparator=self.multivector_comparator   # How to compare matrices [cite: 31]
                        )
                    )
                )
                logger.info(f"Collection '{collection_name}' created successfully with ColBERT config.")

                # Create payload index for faster filtering/checking hash
                try:
                     logger.info(f"Creating payload index on '{settings.FIELD_FILE_HASH}' for collection '{collection_name}'.")
                     await self.client.create_payload_index(
                         collection_name=collection_name,
                         field_name=settings.FIELD_FILE_HASH,
                         field_schema=models.PayloadSchemaType.KEYWORD # Use KEYWORD for exact hash matching
                     )
                     logger.info(f"Payload index created for '{settings.FIELD_FILE_HASH}'.")
                except Exception as idx_e:
                     logger.error(f"Failed to create payload index for '{settings.FIELD_FILE_HASH}' on '{collection_name}': {idx_e}")

            else:
                logger.debug(f"Collection '{collection_name}' already exists.")
                # Optional: Verify existing collection config matches settings here

        except UnexpectedResponse as e:
            logger.error(f"Qdrant error checking/creating collection '{collection_name}': {e.status_code} - {e.content.decode()}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error checking/creating collection '{collection_name}': {e}", exc_info=True)
            raise

    async def check_hash_exists(self, file_hash: str, doc_type: str) -> bool:
        """
        Checks if any point with the given file_hash exists in the specified collection's payload.
        """
        if not file_hash or not doc_type:
            logger.warning("check_hash_exists called with empty file_hash or doc_type.")
            return False

        await self._ensure_connected()
        collection_name = self.get_collection_name(doc_type)

        try:
            collections_response = await self.client.get_collections()
            if collection_name not in {col.name for col in collections_response.collections}:
                 logger.debug(f"Collection '{collection_name}' does not exist. Cannot check for hash '{file_hash}'.")
                 return False

            search_result = await self.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=settings.FIELD_FILE_HASH, # Use constant from settings
                            match=models.MatchValue(value=file_hash)
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False
            )

            exists = bool(search_result[0])
            logger.debug(f"Hash '{file_hash}' {'found' if exists else 'not found'}"
                         f" in collection '{collection_name}'.")
            return exists

        except UnexpectedResponse as e:
            if e.status_code == 404:
                 logger.warning(f"Collection '{collection_name}' not found during hash check for '{file_hash}'.")
                 return False
            logger.error(f"Qdrant error checking hash '{file_hash}' in '{collection_name}': {e.status_code} - {e.content.decode()}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking hash '{file_hash}' in '{collection_name}': {e}", exc_info=True)
            return False


    async def index_document_points(self, points: List[models.PointStruct], doc_type: str) -> bool:
        """
        Indexes a list of pre-formatted PointStruct objects into the specified collection.
        Ensures the collection exists before indexing.
        """
        if not doc_type:
            logger.error("index_document_points called with empty doc_type.")
            return False

        await self._ensure_connected()
        collection_name = self.get_collection_name(doc_type)
        try:
            await self.create_collection_if_not_exists(doc_type) # Ensure collection exists

            if not points:
                logger.warning(f"index_document_points called with empty points list for {doc_type}. Nothing to index.")
                return True # Or False depending on desired behavior for empty input

            logger.info(f"Uploading {len(points)} pre-prepared points to collection '{collection_name}'...")
            response = await self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True # Wait for operation to complete
            )
            if response.status == models.UpdateStatus.COMPLETED:
                logger.info(f"Successfully indexed points into '{collection_name}'. Status: {response.status}")
                return True
            else:
                # Log potential errors if status is not completed
                logger.error(f"Failed to index points into '{collection_name}'. Status: {response.status}")
                return False
        except UnexpectedResponse as e:
             logger.error(f"Qdrant API error during indexing points into {collection_name}: {e.status_code} - {e.content.decode()}", exc_info=True)
             return False
        except ConnectionError as e:
             logger.error(f"Connection error indexing points into {collection_name}: {e}")
             return False
        except Exception as e:
            logger.error(f"Unexpected error indexing points into {collection_name}: {e}", exc_info=True)
            return False

    # --- NEW: Search Method ---
    async def search_documents(
        self,
        query_vector: List[float], # Expect flattened query vector
        k: int = 10,
        doc_type: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Performs a vector search using the provided flattened query vector.

        Args:
            query_vector (List[float]): The flattened embedding vector of the query.
            k (int): The maximum number of results to return.
            doc_type (Optional[str]): If provided, searches only the collection
                                      for this document type. Otherwise, searches
                                      across all collections matching the prefix.

        Returns:
            List[Tuple[str, float]]: A list of (document_id, score) tuples.
                                     Score is the similarity score from Qdrant.
        """
        await self._ensure_connected()
        search_results: List[models.ScoredPoint] = []

        try:
            if doc_type:
                # Search a single specific collection
                collection_name = self.get_collection_name(doc_type)
                logger.info(f"Searching Qdrant collection '{collection_name}' (k={k})...")
                # Check if collection exists before searching to avoid 404
                try:
                     await self.client.get_collection(collection_name)
                except (UnexpectedResponse, ValueError) as e:
                     # Handle collection not found gracefully (e.g., ValueError in some client versions)
                     if isinstance(e, UnexpectedResponse) and e.status_code == 404:
                          logger.warning(f"Collection '{collection_name}' not found for searching.")
                          return []
                     elif "not found" in str(e).lower(): # Handle ValueError case
                          logger.warning(f"Collection '{collection_name}' not found for searching.")
                          return []
                     else:
                          raise # Re-raise other errors

                search_results = await self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=k,
                    with_payload=False, # Don't need payload for RRF stage
                    with_vectors=False
                )
                logger.info(f"Qdrant search in '{collection_name}' yielded {len(search_results)} results.")

            else:
                # Search across multiple collections matching the prefix
                logger.info(f"Searching Qdrant collections matching prefix '{self.collection_prefix}_*' (k={k})...")
                collections_response = await self.client.get_collections()
                target_collections = [
                    col.name for col in collections_response.collections
                    if col.name.startswith(self.collection_prefix + "_")
                ]

                if not target_collections:
                    logger.warning(f"No Qdrant collections found matching prefix '{self.collection_prefix}_*'.")
                    return []

                logger.info(f"Found target collections for batch search: {target_collections}")

                # Prepare batch search requests
                search_queries = [
                    models.SearchRequest(
                        vector=query_vector,
                        limit=k,
                        with_payload=False,
                        with_vector=False
                    ) for _ in target_collections # Same query for all target collections
                ]

                batch_results = await self.client.search_batch(
                    collection_names=target_collections,
                    requests=search_queries
                )

                # Aggregate results from all collections in the batch
                aggregated_results = []
                for collection_result_list in batch_results:
                    aggregated_results.extend(collection_result_list)

                # Re-sort the aggregated results by score and take top K
                aggregated_results.sort(key=lambda hit: hit.score, reverse=True)
                search_results = aggregated_results[:k]
                logger.info(f"Qdrant batch search yielded {len(search_results)} top results after aggregation.")

            # Format results into List[Tuple[str, float]]
            formatted_results = []
            for hit in search_results:
                 # Ensure hit.id is a string, Qdrant IDs can be int or UUID
                 doc_id = str(hit.id)
                 score = hit.score
                 formatted_results.append((doc_id, float(score)))

            return formatted_results

        except UnexpectedResponse as e:
             # Handle potential API errors during search
             logger.error(f"Qdrant API error during search: {e.status_code} - {e.content.decode()}", exc_info=True)
             return [] # Return empty list on error
        except ConnectionError as e:
             logger.error(f"Connection error during Qdrant search: {e}")
             return []
        except Exception as e:
            logger.error(f"Unexpected error during Qdrant search: {e}", exc_info=True)
            return []