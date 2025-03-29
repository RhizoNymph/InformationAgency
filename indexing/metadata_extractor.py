from indexing.models import DocumentType, models
from pydantic import BaseModel
from pydantic_ai import Agent
from indexing.settings import model

async def extract_metadata(document: str, document_type: DocumentType) -> BaseModel:
    metadata_schema = models[document_type]
    metadata_extractor = Agent(
        model,
        result_type=metadata_schema,
        system_prompt=f"""
            You are a metadata extraction specialist. Analyze the OCR text from the first 10 pages 
            of a PDF document and extract relevant metadata based on the document type ({document_type.value}).
            Be thorough but do not make up information - if a field cannot be determined from the 
            provided text, leave it as None.

            The metadata schema is as follows:
            {metadata_schema.model_json_schema()}
        """
    )
    return metadata_extractor.run(document)