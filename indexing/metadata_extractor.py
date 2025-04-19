import os
import json
import logging
import asyncio
from typing import Type, Optional

from indexing.models import DocumentType, models
from indexing.settings import *
from pydantic import BaseModel, ValidationError

# Import OpenAI library
from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError

logger = logging.getLogger(__name__)

# --- Instantiate OpenAI Client ---
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set.")
    # Decide how to handle this - raise error, exit, or disable LLM feature?
    # For now, let's allow it to potentially fail later if used.

client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


async def extract_metadata(document: str, document_type: DocumentType) -> Optional[BaseModel]:
    """
    Extracts metadata from document text using a direct OpenAI API call.

    Args:
        document: The text content of the document (e.g., OCR result).
        document_type: The classified type of the document.

    Returns:
        A Pydantic model instance containing the extracted metadata,
        or None if extraction fails after retries.
    """
    logger.info(f"Attempting metadata extraction for document type: {document_type.value}")

    metadata_schema: Type[BaseModel] = models[document_type] # Type hint for clarity

    if not document.strip():
        logger.warning("Document text is empty, cannot extract metadata.")
        # Return an empty instance of the specific schema type
        try:
            return metadata_schema()
        except Exception as e:
             logger.error(f"Failed to instantiate empty metadata model {metadata_schema.__name__}: {e}")
             return None # Or re-raise depending on desired behavior

    # Consider limiting document length if needed (e.g., first N pages/chars)
    # document_text_sample = document[:SOME_LIMIT] # Example

    # Construct the system prompt dynamically
    schema_dict = metadata_schema.model_json_schema()
    schema_json_string = json.dumps(schema_dict, indent=2)

    system_prompt = f"""
You are a metadata extraction specialist. Analyze the provided text from a document
and extract relevant metadata based on the document type: {document_type.value}.

Be thorough but do not make up information.

Return ONLY a valid JSON object conforming to the following schema. Do not include
any other text, explanations, or apologies.

JSON Schema:
```json
{schema_json_string}
```
"""

    max_retries = 10
    print(document)
    for attempt in range(max_retries):
        logger.info(f"Metadata extraction attempt {attempt + 1}/{max_retries}")
        try:
            # --- Make the OpenAI API Call ---
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    # Use the full document text for now, consider sampling if too long
                    {"role": "user", "content": f"""Extract metadata from the following document text:

{document}"""}
                ],
                response_format={"type": "json_object"},
                temperature=0.1, # Lower temperature for factual extraction
            )

            print('Response:', response)

            # --- Extract and Parse Content ---
            llm_output_str = response.choices[0].message.content
            if not llm_output_str:
                 logger.warning(f"Attempt {attempt + 1} resulted in empty content.")
                 continue

            logger.debug(f"LLM raw JSON string output: {llm_output_str}")
            parsed_data = json.loads(llm_output_str)

            # --- Validate with Pydantic ---
            # No need to lowercase keys/values here unless the schema expects it
            validated_metadata = metadata_schema.model_validate(parsed_data)

            logger.info(f"Metadata extraction successful for type {document_type.value}.")
            return validated_metadata

        # --- Error Handling ---
        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt + 1} failed: JSON Decode Error - {e}. Output: {llm_output_str}")
        except ValidationError as e:
            logger.warning(f"Attempt {attempt + 1} failed: Pydantic Validation Error - {e}. Raw data: {parsed_data}")
        except APIConnectionError as e:
             logger.error(f"Attempt {attempt + 1} failed: OpenAI Connection Error - {e}")
        except RateLimitError as e:
             logger.warning(f"Attempt {attempt + 1} failed: OpenAI Rate Limit Error - {e}. Retrying after delay...")
             await asyncio.sleep(5)
        except APIError as e:
             logger.error(f"Attempt {attempt + 1} failed: OpenAI API Error - Status={e.status_code}, Message={e.message}")
        except Exception as e:
             logger.error(f"Attempt {attempt + 1} failed: Unexpected Error - {e}", exc_info=True)

        # Delay before next retry
        if attempt < max_retries - 1:
             await asyncio.sleep(1)

    # If all retries fail
    logger.error(f"Metadata extraction failed after all retries for type {document_type.value}.")
    # Return an empty instance of the specific schema type on failure
    try:
        return metadata_schema()
    except Exception as e:
        logger.error(f"Failed to instantiate empty metadata model {metadata_schema.__name__} after retries: {e}")
        return None # Or re-raise