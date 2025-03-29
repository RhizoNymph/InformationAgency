import os
import json # Needed for parsing potential JSON host lists if we use that approach later
from dotenv import load_dotenv

load_dotenv()

# --- OpenSearch Connection Details ---
# --- Environment Variable Configuration ---
OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT = int(os.getenv('OPENSEARCH_PORT', '9200'))

# Construct the hosts list used by the client
OPENSEARCH_HOSTS = [
    {'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}
    # For multi-node via ENV, consider parsing a JSON string:
    # OPENSEARCH_HOSTS_JSON = os.getenv('OPENSEARCH_HOSTS_JSON', '[{"host": "localhost", "port": 9200}]')
    # try:
    #     OPENSEARCH_HOSTS = json.loads(OPENSEARCH_HOSTS_JSON)
    # except json.JSONDecodeError:
    #     print(f"Warning: Invalid JSON in OPENSEARCH_HOSTS_JSON. Using default: [{'host': 'localhost', 'port': 9200}]")
    #     OPENSEARCH_HOSTS = [{'host': 'localhost', 'port': 9200}]
]


# Authentication (Optional) - Set OPENSEARCH_USER and OPENSEARCH_PASSWORD
OPENSEARCH_USER = os.getenv('OPENSEARCH_USER', None)
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD', None)
OPENSEARCH_HTTP_AUTH = (OPENSEARCH_USER, OPENSEARCH_PASSWORD) if OPENSEARCH_USER and OPENSEARCH_PASSWORD else None

# SSL/TLS Settings
OPENSEARCH_USE_SSL = os.getenv('OPENSEARCH_USE_SSL', 'False').lower() == 'true'
OPENSEARCH_VERIFY_CERTS = os.getenv('OPENSEARCH_VERIFY_CERTS', 'False').lower() == 'true'
OPENSEARCH_CA_CERTS = os.getenv('OPENSEARCH_CA_CERTS', None) # Path to CA bundle if VERIFY_CERTS is true
# These are less common to set via ENV, but possible:
# OPENSEARCH_SSL_ASSERT_HOSTNAME = os.getenv('OPENSEARCH_SSL_ASSERT_HOSTNAME', 'False').lower() == 'true'
OPENSEARCH_SSL_SHOW_WARN = os.getenv('OPENSEARCH_SSL_SHOW_WARN', str(not OPENSEARCH_VERIFY_CERTS)).lower() == 'true' # Show warn if not verifying by default
OPENSEARCH_TIMEOUT = int(os.getenv('OPENSEARCH_TIMEOUT', '30')) # Connection timeout

# --- Indexing Configuration ---
# Prefix for index names (e.g., 'docs_book', 'docs_paper')
OPENSEARCH_INDEX_PREFIX = os.getenv('OPENSEARCH_INDEX_PREFIX', 'docs')
# Base properties common to all indices (defined directly, less likely to change via env)
BASE_MAPPING_PROPERTIES = {
    "content": {"type": "text"},
    "chunk_index": {"type": "integer"},
    "total_chunks": {"type": "integer"},
    "file_hash": {"type": "keyword"},
    "timestamp": {"type": "date"}
}

# Refresh policy after indexing ('true', 'false', 'wait_for')
OPENSEARCH_REFRESH_POLICY = os.getenv('OPENSEARCH_REFRESH_POLICY', 'wait_for') # Good default for this app


# --- Qdrant Connection Settings ---
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
QDRANT_GRPC_PORT = int(os.getenv('QDRANT_GRPC_PORT', '6334')) # gRPC port for async client
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', None) # Optional API key
QDRANT_USE_TLS = os.getenv('QDRANT_USE_TLS', 'False').lower() == 'true' # For cloud/secured deployments
# Use ":memory:" for local in-memory instance (useful for testing)
# Set QDRANT_LOCATION = ":memory:" in .env or environment
QDRANT_LOCATION = os.getenv('QDRANT_LOCATION', None) # e.g., "http://localhost:6333" or ":memory:"

# --- Qdrant Indexing Settings ---
QDRANT_COLLECTION_PREFIX = os.getenv('QDRANT_COLLECTION_PREFIX', 'colbert_docs')

# --- ColBERT Specific Settings (Example using colbert-ir/colbertv2.0) ---
# Dimension of individual vectors within the ColBERT multi-vector
COLBERT_VECTOR_DIM = int(os.getenv('COLBERT_VECTOR_DIM', '128'))
# Distance metric for individual vectors
COLBERT_VECTOR_DISTANCE = os.getenv('COLBERT_VECTOR_DISTANCE', 'Cosine') # Options: Cosine, Dot, Euclid
# Comparator for multi-vector similarity
COLBERT_COMPARATOR = os.getenv('COLBERT_COMPARATOR', 'MaxSim') # Currently only MaxSim supported effectively for ColBERT

# --- MinIO Connection Settings ---
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000') # e.g., 'minio.example.com' or 'localhost:9000'
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin') # Default MinIO access key
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin') # Default MinIO secret key
MINIO_USE_SSL = os.getenv('MINIO_USE_SSL', 'False').lower() == 'true' # Use https if true

# --- MinIO Upload Settings ---
MINIO_BUCKET_NAME = os.getenv('MINIO_BUCKET_NAME', 'documents') # Target bucket for uploads

# --- Field Names (Used across clients/indexing) ---
FIELD_CONTENT = "content"
FIELD_CHUNK_INDEX = "chunk_index"
FIELD_TOTAL_CHUNKS = "total_chunks"
FIELD_FILE_HASH = "file_hash"
FIELD_TIMESTAMP = "timestamp"

# --- Embedding Model Cache (Optional) ---
# Controls where FastEmbed downloads models
FASTEMBED_CACHE_DIR = os.getenv('FASTEMBED_CACHE_DIR', None) # Let FastEmbed use its default if not set
