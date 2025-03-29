# storage/minio_client.py
import logging
import os
import mimetypes
from typing import Optional, Dict # Added Dict

# MinIO client library
from minio import Minio
from minio.error import S3Error

# Import settings from the sibling settings.py file
from . import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MinioStorageClient:
    """
    A client for interacting with MinIO (or S3 compatible storage)
    to upload original document files with associated metadata.
    """
    def __init__(self):
        # ... (init remains the same) ...
        self.client: Optional[Minio] = None
        self.endpoint = settings.MINIO_ENDPOINT
        self.access_key = settings.MINIO_ACCESS_KEY
        self.secret_key = settings.MINIO_SECRET_KEY
        self.use_ssl = settings.MINIO_USE_SSL
        self.bucket_name = settings.MINIO_BUCKET_NAME
        logger.info(f"MinioStorageClient initialized for endpoint: {self.endpoint}, bucket: {self.bucket_name}")


    def connect(self) -> bool:
        if self.client:
            try:
                # Simple check: does the bucket exist? If not, connection might be stale or permissions changed.
                self.client.bucket_exists(self.bucket_name)
                logger.debug("MinIO connection already established and seems active.")
                return True
            except S3Error as e:
                 logger.warning(f"MinIO connection check failed (bucket check): {e}. Reconnecting...")
                 self.client = None
            except Exception as e:
                 logger.warning(f"MinIO connection check failed (unexpected): {e}. Reconnecting...")
                 self.client = None

        try:
            logger.info(f"Attempting to connect to MinIO at {self.endpoint}...")
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.use_ssl
            )
            self._ensure_bucket_exists_sync() # Verify connection and bucket existence
            logger.info(f"Successfully connected to MinIO and verified bucket '{self.bucket_name}'.")
            return True
        except S3Error as e:
            logger.error(f"Failed to connect to MinIO or ensure bucket exists: {e}", exc_info=True)
            self.client = None
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during MinIO connection: {e}", exc_info=True)
            self.client = None
            return False


    def _ensure_connected_sync(self):
        if not self.client:
            logger.warning("MinIO client not connected. Attempting to connect.")
            if not self.connect():
                raise ConnectionError(f"Failed to establish connection with MinIO at {self.endpoint}.")


    def _ensure_bucket_exists_sync(self):
        if not self.client:
             raise ConnectionError("MinIO client not initialized before checking bucket.")
        try:
            found = self.client.bucket_exists(self.bucket_name)
            if not found:
                logger.info(f"Bucket '{self.bucket_name}' does not exist. Creating...")
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Bucket '{self.bucket_name}' created successfully.")
            else:
                logger.debug(f"Bucket '{self.bucket_name}' already exists.")
        except S3Error as e:
            logger.error(f"Error checking or creating bucket '{self.bucket_name}': {e}")
            raise
        except Exception as e:
             logger.error(f"Unexpected error during bucket check/creation for '{self.bucket_name}': {e}")
             raise

    def hash_exists(self, file_hash: str) -> bool:
        """
        Checks if an object with the given name exists in the bucket.

        Args:
            object_name (str): The name (key) of the object to check.

        Returns:
            bool: True if the object exists, False otherwise.
        """
        if not file_hash:
            logger.warning("hash_exists called with empty file_hash.")
            return False

        try:
            self._ensure_connected_sync()
            self.client.stat_object(self.bucket_name, file_hash)
            logger.debug(f"Object '{file_hash}' exists in bucket '{self.bucket_name}'.")
            return True  # Object exists
        except S3Error as e:
            # A common S3Error for non-existent objects has status code 404
            # You could add more specific error checking here if needed
            # For example: if e.code == 'NoSuchKey': return False
            logger.debug(f"Object '{file_hash}' does not exist or error checking existence: {e}")
            return False # Object does not exist or another S3 error occurred
        except ConnectionError as e:
             logger.error(f"MinIO connection error during object existence check: {e}")
             return False # Indicate non-existence on connection failure
        except Exception as e:
            logger.error(f"Unexpected error checking existence for object '{file_hash}': {e}", exc_info=True)
            return False # Treat unexpected errors as non-existent for safety


    def upload_document(self,
                          file_path: str,
                          file_hash: str,
                          metadata: Optional[Dict[str, str]] = None) -> bool: # Added metadata parameter
        """
        Uploads the full document file to MinIO with optional metadata.

        The object name in MinIO will be the file_hash. Metadata keys and values
        must be strings.

        Args:
            file_path (str): The local path to the file to upload.
            file_hash (str): The hash of the file, used as the object name in MinIO.
            metadata (Optional[Dict[str, str]]): A dictionary of custom metadata
                                                 to store with the object. Keys and values
                                                 must be strings.

        Returns:
            bool: True if upload was successful, False otherwise.
        """
        if not file_path or not file_hash:
            logger.error("upload_document called with empty file_path or file_hash.")
            return False

        # --- Prepare Metadata ---
        upload_metadata = {}
        if metadata:
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    # Basic validation: ensure keys/values are strings
                    if isinstance(key, str) and isinstance(value, str):
                         # Further cleaning could be added here if needed (e.g., invalid chars)
                         # MinIO library handles adding 'x-amz-meta-' prefix
                         upload_metadata[key] = value
                    else:
                         # Log a warning and skip non-string key/value pairs
                         logger.warning(f"Skipping non-string metadata item: Key='{key}' (type={type(key)}), Value='{value}' (type={type(value)}) for object '{file_hash}'")

        try:
            # Ensure connection (bucket check is done within connect)
            self._ensure_connected_sync()

            if not os.path.exists(file_path):
                 logger.error(f"File not found at path: {file_path}")
                 return False

            content_type, _ = mimetypes.guess_type(file_path)
            content_type = content_type or 'application/octet-stream'

            logger.info(f"Uploading '{file_path}' to bucket '{self.bucket_name}' as object '{file_hash}' (Type: '{content_type}')"
                        f"{' with metadata.' if upload_metadata else '.'}")

            # Use fput_object with the prepared metadata dictionary
            result = self.client.fput_object(
                bucket_name=self.bucket_name,
                object_name=file_hash,
                file_path=file_path,
                content_type=content_type,
                metadata=upload_metadata if upload_metadata else None 
            )
            logger.info(
                f"Successfully uploaded '{file_hash}' to bucket '{self.bucket_name}'. "
                f"ETag: {result.etag}, VersionID: {result.version_id}"
            )
            return True

        except S3Error as e:
            logger.error(f"MinIO S3 error uploading '{file_path}' as '{file_hash}': {e}", exc_info=True)
            return False
        except FileNotFoundError:
             logger.error(f"File not found during upload: {file_path}")
             return False
        except ConnectionError as e:
             logger.error(f"MinIO connection error during upload: {e}")
             return False
        except Exception as e:
            logger.error(f"Unexpected error uploading '{file_path}' as '{file_hash}': {e}", exc_info=True)
            return False