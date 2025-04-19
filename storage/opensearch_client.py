# storage/opensearch_client.py
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from opensearchpy import AsyncOpenSearch, OpenSearchException, NotFoundError

from . import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenSearchClient:
    """
    An asynchronous client for interacting with OpenSearch, designed for indexing
    pre-chunked documents, checking for existing content, searching, and retrieving by ID.
    """
    def __init__(self):
        self.client: Optional[AsyncOpenSearch] = None
        self.host = settings.OPENSEARCH_HOST
        self.port = settings.OPENSEARCH_PORT
        self.index_prefix = settings.OPENSEARCH_INDEX_PREFIX
        self.refresh_policy = settings.OPENSEARCH_REFRESH_POLICY

        self._connection_args = {
            'hosts': settings.OPENSEARCH_HOSTS,
            'http_auth': settings.OPENSEARCH_HTTP_AUTH,
            'use_ssl': settings.OPENSEARCH_USE_SSL,
            'verify_certs': settings.OPENSEARCH_VERIFY_CERTS,
            'ssl_show_warn': settings.OPENSEARCH_SSL_SHOW_WARN,
            'ca_certs': settings.OPENSEARCH_CA_CERTS,
            'timeout': settings.OPENSEARCH_TIMEOUT,
        }
        primary_host = settings.OPENSEARCH_HOSTS[0] if settings.OPENSEARCH_HOSTS else {'host': 'N/A', 'port': 'N/A'}
        logger.info(f"OpenSearchClient initialized for {primary_host['host']}:{primary_host['port']}...")

    async def connect(self) -> bool:
        """Establishes an asynchronous connection to the OpenSearch cluster."""
        if self.client:
            try:
                await self.client.info()
                logger.debug("Connection already established and verified.")
                return True
            except Exception as e:
                logger.warning(f"Existing connection check failed: {e}. Reconnecting...")
                await self.close()

        try:
            logger.info(f"Attempting to connect to OpenSearch...")
            self.client = AsyncOpenSearch(**self._connection_args)
            info = await self.client.info()
            logger.info(f"Successfully connected to OpenSearch: {info['version']['distribution']} v{info['version']['number']}")
            return True
        except OpenSearchException as e:
            logger.error(f"Failed to connect to OpenSearch: {e}", exc_info=True)
            self.client = None
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenSearch connection: {e}", exc_info=True)
            self.client = None
            return False

    async def close(self):
        """Closes the asynchronous OpenSearch connection."""
        if self.client:
            try:
                await self.client.close()
                logger.info("OpenSearch connection closed.")
            except Exception as e:
                logger.error(f"Error closing OpenSearch connection: {e}", exc_info=True)
            finally:
                self.client = None

    async def _ensure_connected(self):
        """Internal helper to ensure the client is connected before an operation."""
        if not self.client:
            logger.warning("Client not connected. Attempting to connect.")
            if not await self.connect():
                raise ConnectionError("Failed to establish connection with OpenSearch.")

    def get_index_name(self, doc_type: str) -> str:
        """Generates the full index name based on the prefix and document type."""
        if not doc_type:
             raise ValueError("doc_type cannot be empty")
        return f"{self.index_prefix}_{doc_type.lower().strip()}"

    async def create_index_if_not_exists(self, doc_type: str, mappings: Dict[str, Any]):
        """Creates an index for the given document type if it doesn't already exist."""
        await self._ensure_connected()
        index_name = self.get_index_name(doc_type)

        try:
            exists = await self.client.indices.exists(index=index_name)
            if not exists:
                logger.info(f"Index '{index_name}' does not exist. Creating...")
                if 'properties' not in mappings:
                     mappings['properties'] = {}
                mappings['properties'].update(settings.BASE_MAPPING_PROPERTIES)
                await self.client.indices.create(
                    index=index_name,
                    body={"mappings": mappings}
                )
                logger.info(f"Index '{index_name}' created successfully.")
            else:
                logger.debug(f"Index '{index_name}' already exists.")
        except OpenSearchException as e:
            logger.error(f"Error checking or creating index '{index_name}': {e}", exc_info=True)
            raise

    async def check_hash_exists(self, file_hash: str, doc_type: str) -> bool:
        """
        Checks if any document chunk with the given file_hash exists in the
        specified document type's index.

        Args:
            file_hash (str): The file hash to check for.
            doc_type (str): The document type, used to determine the index name.

        Returns:
            bool: True if a document with the hash exists, False otherwise.
        """
        if not file_hash:
            logger.warning("check_hash_exists called with empty file_hash.")
            return False
        if not doc_type:
             logger.warning("check_hash_exists called with empty doc_type.")
             return False

        await self._ensure_connected()
        index_name = self.get_index_name(doc_type)

        try:
            # Check if index exists first to avoid error on count
            if not await self.client.indices.exists(index=index_name):
                logger.info(f"Index '{index_name}' does not exist. Cannot check for hash '{file_hash}'.")
                return False

            # Use count API for efficiency
            # Query for documents where the file_hash field matches exactly
            query = {
                "query": {
                    "term": {
                        settings.FIELD_FILE_HASH: file_hash
                    }
                }
            }
            response = await self.client.count(index=index_name, body=query)
            count = response.get('count', 0)
            logger.debug(f"Found {count} documents with hash '{file_hash}' in index '{index_name}'.")
            return count > 0

        except NotFoundError:
             # This can happen in race conditions if index deleted between exists and count
             logger.warning(f"Index '{index_name}' not found during count operation for hash '{file_hash}'.")
             return False
        except OpenSearchException as e:
            logger.error(f"Error checking hash '{file_hash}' in index '{index_name}': {e}", exc_info=True)
            # Decide if you want to raise or return False on error
            return False # Return False assuming it doesn't exist if check fails

    async def index_document(self,
                             chunks: List[str],
                             metadata: Dict[str, Any],
                             file_hash: str,
                             doc_type: str,
                             mappings: Dict[str, Any]) -> bool:
        """
        Indexes a list of pre-chunked document sections along with shared metadata.

        Args:
            chunks (List[str]): A list where each item is a text chunk of the document.
            metadata (Dict[str, Any]): A dictionary of metadata fields associated
                                       with the entire document.
            file_hash (str): The unique hash identifier for the original file.
            doc_type (str): The type of the document (e.g., 'book', 'paper').
                            Used for index naming and mapping creation.
            mappings (Dict[str, Any]): The OpenSearch mapping definition specific
                                       to this doc_type.

        Returns:
            bool: True if indexing was successful for all chunks, False otherwise.
        """
        if not chunks:
             logger.warning(f"index_document called with empty or no chunks for hash {file_hash}. Nothing to index.")
             return True
        if not all([metadata, file_hash, doc_type]):
            logger.error("index_document called with missing metadata, file_hash, or doc_type.")
            return False

        await self._ensure_connected()
        index_name = self.get_index_name(doc_type)

        try:
            await self.create_index_if_not_exists(doc_type, mappings)
            
            bulk_operations = []
            total_chunks = len(chunks)
            logger.info(f"Indexing document hash '{file_hash}' into '{index_name}' from {total_chunks} provided chunks...")
            for i, chunk_text in enumerate(chunks):
                if not isinstance(chunk_text, str) or not chunk_text.strip():
                    continue
                    
                chunk_doc = metadata.copy()
                chunk_doc.update({
                    settings.FIELD_CONTENT: chunk_text,
                    settings.FIELD_CHUNK_INDEX: i,
                    settings.FIELD_TOTAL_CHUNKS: total_chunks,
                    settings.FIELD_FILE_HASH: file_hash,
                    settings.FIELD_TIMESTAMP: datetime.now(timezone.utc).isoformat()
                })
                
                # Add index operation to bulk list
                bulk_operations.extend([
                    {'index': {'_index': index_name, '_id': f"{file_hash}_{i}"}},
                    chunk_doc
                ])
            
            if bulk_operations:
                response = await self.client.bulk(body=bulk_operations, refresh=self.refresh_policy)
                if response.get('errors'):
                    logger.error(f"Bulk indexing had errors: {response}")
                    return False
                
            return True
            
        except ConnectionError as e:
             logger.error(f"Connection error during indexing: {e}")
             return False
        except ValueError as e:
             logger.error(f"Configuration error during indexing: {e}")
             return False
        except OpenSearchException as e:
            logger.error(f"OpenSearch error during index setup or indexing for hash {file_hash}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error during indexing for hash {file_hash}: {e}", exc_info=True)
            return False

    async def search_documents(
        self,
        query_text: str,
        k: int = 10,
        doc_type: Optional[str] = None
    ) -> List[Tuple[str, float, dict]]:
        """
        Performs a simple keyword search using a match query.
        If doc_type is provided, searches only that index.
        Otherwise, searches all indices matching the prefix.
        Returns a list of tuples: (doc_id, score, payload).
        """
        await self._ensure_connected()

        if doc_type:
            target_index = self.get_index_name(doc_type)
            logger.info(f"Targeting specific OpenSearch index: {target_index}")
        else:
            target_index = f"{self.index_prefix}_*"
            logger.info(f"Searching across indices matching prefix: {target_index}")

        # Simple match query against the content field
        search_body = {
            "size": k,
            "query": {
                "match": {
                    settings.FIELD_CONTENT: {
                        "query": query_text,
                        "operator": "OR"
                    }
                }
            }
            # _source includes all fields by default, no need to specify unless excluding
        }

        try:
            logger.info(f"Executing OpenSearch search on index '{target_index}' for query '{query_text}' (k={k})...")
            response = await self.client.search(
                index=target_index,
                body=search_body,
                ignore=[400, 404]
            )

            results = []
            hits = response.get('hits', {}).get('hits', [])
            for hit in hits:
                doc_id = hit.get('_id')
                score = hit.get('_score')
                source = hit.get('_source') # <-- Get the payload
                if doc_id is not None and score is not None and source is not None:
                    results.append((doc_id, float(score), source)) # <-- Include payload

            logger.info(f"OpenSearch search yielded {len(results)} results.")
            return results

        except NotFoundError:
            logger.warning(f"No indices found matching '{target_index}' for search.")
            return []
        except OpenSearchException as e:
            logger.error(f"Error during OpenSearch search on '{target_index}': {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error during OpenSearch search: {e}", exc_info=True)
            return []

    async def get_documents_by_ids(
        self,
        doc_ids: List[str],
        doc_type: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves multiple documents based on their IDs using the mget API.

        Args:
            doc_ids (List[str]): A list of document IDs (_id) to retrieve.
            doc_type (Optional[str]): If provided, searches only the index
                                      for this document type. Otherwise, searches
                                      across all indices matching the prefix.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping found document IDs
                                       to their source payload (_source).
        """
        if not doc_ids:
            return {}

        await self._ensure_connected()

        if doc_type:
            target_index = self.get_index_name(doc_type)
            mget_body = {"ids": doc_ids}
        else:
            target_index = f"{self.index_prefix}_*"
            mget_body = {
                "docs": [{"_index": target_index, "_id": doc_id} for doc_id in doc_ids]
            }
            logger.info(f"Performing mget across indices matching prefix: {target_index}")

        try:
            logger.info(f"Executing OpenSearch mget on index '{target_index}' for {len(doc_ids)} IDs...")
            response = await self.client.mget(
                index=target_index if doc_type else None,
                body=mget_body
            )

            results = {}
            docs = response.get('docs', [])
            for doc in docs:
                if doc.get('found'):
                    doc_id = doc.get('_id')
                    payload = doc.get('_source')
                    if doc_id and payload:
                        results[doc_id] = payload

            logger.info(f"OpenSearch mget retrieved details for {len(results)} out of {len(doc_ids)} requested IDs.")
            return results

        except NotFoundError:
            logger.warning(f"No indices found matching '{target_index}' for mget.")
            return {}
        except OpenSearchException as e:
            logger.error(f"Error during OpenSearch mget on '{target_index}': {e}", exc_info=True)
            return {}
        except Exception as e:
            logger.error(f"Unexpected error during OpenSearch mget: {e}", exc_info=True)
            return {}