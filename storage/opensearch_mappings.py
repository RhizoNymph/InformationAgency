mappings = {
            'book': {
                "properties": {
                    "title": {"type": "text"},
                    "authors": {"type": "keyword"},
                    "publisher": {"type": "keyword"},
                    "publication_year": {"type": "integer"},
                    "isbn": {"type": "keyword"},
                    "edition": {"type": "keyword"},
                    "language": {"type": "keyword"},
                    "subject_areas": {"type": "keyword"},
                    "table_of_contents": {"type": "text"},
                    "content": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "file_hash": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            },
            'paper': {
                "properties": {
                    "title": {"type": "text"},
                    "authors": {"type": "keyword"},
                    "abstract": {"type": "text"},
                    "keywords": {"type": "keyword"},
                    "doi": {"type": "keyword"},
                    "journal": {"type": "keyword"},
                    "conference": {"type": "keyword"},
                    "publication_year": {"type": "integer"},
                    "institution": {"type": "keyword"},
                    "citations": {"type": "text"},
                    "content": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "file_hash": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            },
            'blog_article': {
                "properties": {
                    "title": {"type": "text"},
                    "authors": {"type": "keyword"},
                    "publication_date": {"type": "date"},
                    "blog_name": {"type": "keyword"},
                    "url": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "reading_time": {"type": "integer"},
                    "summary": {"type": "text"},
                    "series": {"type": "keyword"},
                    "content": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "file_hash": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            },
            'technical_report': {
                "properties": {
                    "title": {"type": "text"},
                    "authors": {"type": "keyword"},
                    "organization": {"type": "keyword"},
                    "report_number": {"type": "keyword"},
                    "date": {"type": "date"},
                    "executive_summary": {"type": "text"},
                    "keywords": {"type": "keyword"},
                    "classification": {"type": "keyword"},
                    "content": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "file_hash": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            },
            'thesis': {
                "properties": {
                    "title": {"type": "text"},
                    "author": {"type": "keyword"},
                    "degree": {"type": "keyword"},
                    "institution": {"type": "keyword"},
                    "department": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "advisors": {"type": "keyword"},
                    "abstract": {"type": "text"},
                    "keywords": {"type": "keyword"},
                    "content": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "file_hash": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            },
            'patent': {
                "properties": {
                    "title": {"type": "text"},
                    "inventors": {"type": "keyword"},
                    "assignee": {"type": "keyword"},
                    "patent_number": {"type": "keyword"},
                    "filing_date": {"type": "date"},
                    "publication_date": {"type": "date"},
                    "abstract": {"type": "text"},
                    "classification": {"type": "keyword"},
                    "claims": {"type": "text"},
                    "content": {"type": "text"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "file_hash": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            }
        }