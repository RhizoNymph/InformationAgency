import os
import tempfile

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

TEMP_UPLOAD_DIR = tempfile.mkdtemp(prefix="fastapi_uploads_")
MAX_CLASSIFICATION_SAMPLE_SIZE = 5000
MAX_METADATA_SAMPLE_SIZE = 20000