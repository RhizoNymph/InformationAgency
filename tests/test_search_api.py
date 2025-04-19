import httpx
import asyncio
import os
import logging

# --- Configuration ---
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
SEARCH_ENDPOINT = "/search"
REQUEST_TIMEOUT = 30.0 # Timeout in seconds for search requests

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Test Data ---
# Use a query that is likely to return results based on your indexed data
# For example, if you indexed documents about 'artificial intelligence'
TEST_QUERY = "artificial intelligence"
EXPECTED_RESULTS_COUNT = 5 # Set the number of results (k) you expect

async def test_search_endpoint():
    """
    Tests the /search endpoint by sending a query and checking the response.
    """
    logger.info(f"--- Starting Search API Test ---")
    logger.info(f"Target API: {API_BASE_URL}{SEARCH_ENDPOINT}")
    logger.info(f"Test Query: '{TEST_QUERY}', Expected Results (k): {EXPECTED_RESULTS_COUNT}")

    # Parameters for the GET request
    params = {
        "query": TEST_QUERY,
        "k": EXPECTED_RESULTS_COUNT
        # Add other parameters like 'doc_type' if needed for your test case
        # "doc_type": "research_paper"
    }

    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=REQUEST_TIMEOUT) as client:
        try:
            logger.info(f"Sending GET request to {SEARCH_ENDPOINT} with params: {params}")
            response = await client.get(SEARCH_ENDPOINT, params=params)

            logger.info(f"Response Status Code: {response.status_code}")
            logger.info(f"Response Body: {response.text}") # Log first 500 chars

            # Check if the request was successful (status code 200 OK)
            response.raise_for_status()

            # --- Assertions based on the expected search response structure ---
            response_json = response.json()

            assert "query" in response_json, "Response should contain 'query'"
            assert response_json["query"] == TEST_QUERY, f"Response query '{response_json['query']}' should match sent query '{TEST_QUERY}'"

            assert "retrieved_count" in response_json, "Response should contain 'retrieved_count'"
            # Note: retrieved_count might be less than k if fewer results are found
            assert response_json["retrieved_count"] <= EXPECTED_RESULTS_COUNT, f"Retrieved count {response_json['retrieved_count']} should be <= k ({EXPECTED_RESULTS_COUNT})"

            assert "results" in response_json, "Response should contain 'results'"
            assert isinstance(response_json["results"], list), "'results' should be a list"
            assert len(response_json["results"]) == response_json["retrieved_count"], "Length of 'results' list should match 'retrieved_count'"

            # Optionally check the structure of individual results
            if response_json["retrieved_count"] > 0:
                first_result = response_json["results"][0]
                assert "id" in first_result, "Each result should have an 'id'"
                assert "rrf_score" in first_result, "Each result should have an 'rrf_score'"
                assert isinstance(first_result["rrf_score"], float), "'rrf_score' should be a float"
                assert "metadata" in first_result, "Each result should have 'metadata'"
                # Add more checks for metadata content if needed

            logger.info(f"Search test successful. Retrieved {response_json['retrieved_count']} results.")

        except httpx.RequestError as exc:
            logger.error(f"An error occurred while requesting {exc.request.url!r}: {exc}")
            assert False, f"HTTP Request failed: {exc}"
        except httpx.HTTPStatusError as exc:
            logger.error(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
            logger.error(f"Response body: {exc.response.text}")
            assert False, f"API returned error: {exc.response.status_code}"
        except Exception as e:
             logger.error(f"An unexpected error occurred: {e}", exc_info=True)
             assert False, f"Unexpected error: {e}"


async def main():
    # Optional: Add setup steps here if needed (e.g., ensuring data is indexed)
    logger.info("Running search endpoint test...")
    await test_search_endpoint()
    logger.info("--- Search Test Complete ---")

if __name__ == "__main__":
    # Ensure the API server is running and potentially some documents are indexed
    print(f"Ensure the API server is running at {API_BASE_URL} and data is indexed before proceeding.")
    print("Press Enter to start the test...")
    input() # Simple pause
    asyncio.run(main())
