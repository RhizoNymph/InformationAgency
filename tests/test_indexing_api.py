import httpx
import asyncio
import os
import logging

# --- Configuration ---
# Adjust if your API runs on a different host or port
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
INDEX_ENDPOINT = "/index_document"  # Assuming this is the correct endpoint path
TEST_FILE_PATH = "test.pdf" # Make sure this file exists where you run the script
REQUEST_TIMEOUT = 120.0 # Timeout in seconds (indexing might take time)

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_index_pdf_document():
    """
    Tests the /index/ endpoint by uploading test.pdf.
    """
    test_file = os.path.abspath(TEST_FILE_PATH)
    logger.info(f"--- Starting Document Indexing Test ---")
    logger.info(f"Target API: {API_BASE_URL}{INDEX_ENDPOINT}")
    logger.info(f"Test File: {test_file}")

    # Check if the test file exists before proceeding
    if not os.path.exists(test_file):
        logger.error(f"Test file '{test_file}' not found. Aborting test.")
        assert False, f"Test file not found: {test_file}" # Make test fail clearly

    # Use httpx.AsyncClient for async requests
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=REQUEST_TIMEOUT) as client:
        try:
            # Open the file in binary read mode ('rb')
            with open(test_file, "rb") as f:
                # Prepare the 'files' dictionary for multipart/form-data upload.
                # The key 'file' should match the parameter name in your FastAPI endpoint.
                files_data = {'file': (os.path.basename(test_file), f, 'application/pdf')}

                logger.info(f"Sending POST request to {INDEX_ENDPOINT}...")
                response = await client.post(INDEX_ENDPOINT, files=files_data)

            logger.info(f"Response Status Code: {response.status_code}")
            logger.info(f"Response Body: {response.text}")

            # Check if the request was successful (status code 2xx)
            response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses

            # --- Add more specific assertions based on your API's response ---
            response_json = response.json()
            assert "file_hash" in response_json, "Response should contain 'file_hash'"
            assert "message" in response_json, "Response should contain 'message'"
            assert response_json["message"] == "Document indexed successfully" # Or similar success message
            assert len(response_json["file_hash"]) == 64 # SHA-256 hash length

            logger.info("Test finished successfully.")

        except httpx.RequestError as exc:
            logger.error(f"An error occurred while requesting {exc.request.url!r}: {exc}")
            assert False, f"HTTP Request failed: {exc}"
        except httpx.HTTPStatusError as exc:
            logger.error(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
            logger.error(f"Response body: {exc.response.text}")
            assert False, f"API returned error: {exc.response.status_code}"
        except FileNotFoundError:
             logger.error(f"Error: Could not open the test file '{test_file}'.")
             assert False, f"Failed to open test file: {test_file}" # Should be caught above, but safeguard
        except Exception as e:
             logger.error(f"An unexpected error occurred: {e}", exc_info=True)
             assert False, f"Unexpected error: {e}"


async def main():
    await test_index_pdf_document()
    logger.info("--- Test Complete ---")

if __name__ == "__main__":
    # Ensure the API server (uvicorn, etc.) is running before executing this script.
    print(f"Ensure the API server is running at {API_BASE_URL} before proceeding.")
    print("Press Enter to start the test...")
    input() # Simple way to pause before starting
    asyncio.run(main())