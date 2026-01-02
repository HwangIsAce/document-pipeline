
# Document Pipeline
A pipeline project that parses documents and indexes them into a vector database for searchability, featuring flexible incremental updates.

# Key features
- PDF Document Parsing
- Text Embedding & Chunking
- Image Embedding 
- Vector Search
- Text & Image Hybrid Search

# Supported Extensions
- JPEG, PNG, PDF, DOCX, PPTX, XLSX, HWPX, 

# Usage

# To-be-Organized

## Postgres & Qdrant 실행
```
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=cocoindex_db \
  -p 5432:5432 \
  postgres

docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```
