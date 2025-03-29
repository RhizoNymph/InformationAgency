An information indexing and retrieval information for LLMs and agents. 


Uses FastAPI, MinIO, OpenSearch, and Qdrant (with ColBERT embeddings via FastEmbed).

Uses an LLM with structured output for document classification and type specific metadata extraction.

Exposes index_document and search routes.

TODO:
- Add the ability to start websearch agents to find documents to index
- Add research agent example that uses this for agentic RAG
- Bring in experiments for using RL post training to make better models for metadata extraction