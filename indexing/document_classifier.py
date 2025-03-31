import os
import json
import logging
from typing import Any  # Keep Any if needed elsewhere, or remove if not
import asyncio # For potential retry delays

# Use the official OpenAI library
from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError
from pydantic import BaseModel, ValidationError

from indexing.models import DocumentType
# We don't need the pydantic_ai Agent specific model import anymore
# from indexing.settings import model # No longer needed directly here

logger = logging.getLogger(__name__)

# --- Configuration (pulled from environment variables like in settings.py) ---
# It's generally better practice to configure the client once, maybe in settings.py
# and import the configured client, but for a direct replacement here, we'll re-read the env vars.
# Ensure these env vars are set where this code runs.
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo") # Provide a default if needed
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") # Can be None if using default OpenAI

# --- Instantiate OpenAI Client ---
# Handle potential missing API key
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set.")
    # Decide how to handle this - raise error, exit, or disable LLM feature?
    # For now, let's allow it to potentially fail later if used.
    # raise ValueError("OPENAI_API_KEY environment variable not set.")

client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL, # Pass None if OPENAI_BASE_URL is not set or empty
)

# --- Pydantic Model for Validation ---
class DocTypeResponse(BaseModel):
    document_type: DocumentType

# --- System Prompt ---
# Keep the prompt clear about the task and desired JSON structure
SYSTEM_PROMPT = """
You are a document classification specialist. Analyze the provided text sample (first ~3000 characters) from a document and determine its type.
Choose ONE from the following categories:
BOOK, PAPER, BLOG_ARTICLE, TECHNICAL_REPORT, THESIS, PRESENTATION, DOCUMENTATION, PATENT.
If unsure or it doesn't fit well, classify as UNKNOWN.

Return ONLY a valid JSON object containing the classification. The JSON object must have exactly one key, "document_type", whose value is the chosen category string (e.g., "PAPER", "BOOK", "UNKNOWN").

Example valid JSON output:
{
    "document_type": "PAPER"
}
"""

# --- Classification Function (Rewritten) ---
async def classify(text_sample: str) -> DocumentType:
    """
    Classifies the document based on a text sample using a direct LLM call
    with the OpenAI library, manual parsing, validation, and retries.
    Returns DocumentType.
    """
    logger.info("Attempting document classification using direct OpenAI call...")

    if not text_sample.strip():
        logger.warning("Text sample is empty, cannot classify.")
        return DocumentType.UNKNOWN

    # Limit the sample size if necessary (OpenAI takes care of token limits, but cost/latency)
    max_chars = 3000 # Match prompt description
    if len(text_sample) > max_chars:
        logger.debug(f"Trimming text sample from {len(text_sample)} to {max_chars} chars.")
        text_sample = text_sample[:max_chars]

    max_retries = 3
    for attempt in range(max_retries):
        logger.info(f"LLM classification attempt {attempt + 1}/{max_retries}")
        try:
            # --- Make the OpenAI API Call ---
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classify the following text sample:\n\n{text_sample}"}
                ],
                # Explicitly request JSON output (requires newer models like gpt-3.5-turbo-1106+)
                response_format={"type": "json_object"},
                temperature=0.2, # Lower temperature for more deterministic classification
            )

            # --- Extract and Parse Content ---
            llm_output_str = response.choices[0].message.content
            if not llm_output_str:
                 logger.warning(f"Attempt {attempt + 1} resulted in empty content.")
                 continue # Go to next retry

            logger.debug(f"LLM raw JSON string output: {llm_output_str}")
            parsed_data = json.loads(llm_output_str)

            # --- Convert enum value to lowercase BEFORE validation ---
            if isinstance(parsed_data.get('document_type'), str):
                 logger.debug(f"Converting document_type value to lowercase: {parsed_data['document_type']}")
                 parsed_data['document_type'] = parsed_data['document_type'].lower()
            else:
                 # Handle case where key is missing or value is not a string, if necessary
                 logger.warning(f"Parsed JSON missing 'document_type' key or value is not a string: {parsed_data}")
                 # Let Pydantic validation handle the error, or raise/continue here
                 pass

            # --- Validate with Pydantic ---
            validated_response = DocTypeResponse.model_validate(parsed_data)
            doc_type = validated_response.document_type

            logger.info(f"Direct call classification successful: {doc_type.value}")
            return doc_type

        # --- Error Handling ---
        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt + 1} failed: JSON Decode Error - {e}. Output: {llm_output_str}")
        except ValidationError as e:
            # Log the specific validation failure details
            logger.warning(f"Attempt {attempt + 1} failed: Pydantic Validation Error - {e}. Data before validation: {parsed_data}") # Log the potentially modified parsed_data
        except APIConnectionError as e:
             logger.error(f"Attempt {attempt + 1} failed: OpenAI Connection Error - {e}")
             # Connection errors are less likely to be fixed by immediate retry, maybe break early?
        except RateLimitError as e:
             logger.warning(f"Attempt {attempt + 1} failed: OpenAI Rate Limit Error - {e}. Retrying after delay...")
             await asyncio.sleep(5) # Wait longer for rate limit errors
        except APIError as e: # Catch other OpenAI API errors
             logger.error(f"Attempt {attempt + 1} failed: OpenAI API Error - Status={e.status_code}, Message={e.message}")
        except Exception as e: # Catch any other unexpected errors
             logger.error(f"Attempt {attempt + 1} failed: Unexpected Error - {e}", exc_info=True)

        # Delay before next retry (optional, simple backoff could be added)
        if attempt < max_retries - 1:
             await asyncio.sleep(1)

    # If all retries fail
    logger.error("LLM classification failed after all retries (direct call).")
    return DocumentType.UNKNOWN

# --- Remove the old validator function ---
# The @document_classifier.result_validator and fix_llm_output_shenanigans/extract_final_value are no longer needed.