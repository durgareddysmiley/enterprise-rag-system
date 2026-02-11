import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition
from .config import settings
from .database import AsyncSessionLocal
from .models import Document, DocumentStatus
from sqlalchemy import update, select
import asyncio

# Initialize clients
qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

def ensure_collection():
    collections = qdrant_client.get_collections().collections
    exists = any(c.name == settings.QDRANT_COLLECTION_NAME for c in collections)
    if not exists:
        qdrant_client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=rest.VectorParams(size=384, distance=rest.Distance.COSINE),
        )
        # Add full-text index for keyword search support in the payload
        qdrant_client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            field_name="text",
            field_schema=rest.TextIndexParams(
                type="text",
                tokenizer=rest.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=15,
                lowercase=True,
            )
        )

async def process_document(doc_id: str, file_path: str):
    async with AsyncSessionLocal() as db:
        try:
            # Update status to PROCESSING
            await db.execute(
                update(Document).where(Document.id == doc_id).values(status=DocumentStatus.PROCESSING)
            )
            await db.commit()

            # Parse document
            elements = partition(filename=file_path)
            text_content = "\n\n".join([str(el) for el in elements])
            
            # Simple chunking (Recursive character splitting equivalent)
            chunk_size = 1000
            overlap = 200
            chunks = []
            for i in range(0, len(text_content), chunk_size - overlap):
                chunks.append(text_content[i:i + chunk_size])

            if not chunks:
                raise ValueError("No text extracted from document")

            # Generate embeddings
            embeddings = embedding_model.encode(chunks)
            
            # Prepare points for Qdrant
            ensure_collection()
            points = [
                rest.PointStruct(
                    id=str(uuid_v4_from_doc_chunk(doc_id, i)),
                    vector=embeddings[i].tolist(),
                    payload={
                        "document_id": doc_id,
                        "text": chunks[i],
                        "chunk_index": i,
                        "filename": os.path.basename(file_path)
                    }
                ) for i in range(len(chunks))
            ]
            
            # Upsert into Qdrant
            qdrant_client.upsert(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points=points
            )

            # Update status to COMPLETED
            await db.execute(
                update(Document).where(Document.id == doc_id).values(status=DocumentStatus.COMPLETED)
            )
            await db.commit()

        except Exception as e:
            print(f"Error processing document {doc_id}: {e}")
            await db.execute(
                update(Document).where(Document.id == doc_id).values(
                    status=DocumentStatus.FAILED,
                    error_message=str(e)
                )
            )
            await db.commit()

async def delete_document_vectors(doc_id: str):
    qdrant_client.delete(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points_selector=rest.Filter(
            must=[rest.FieldCondition(key="document_id", match=rest.MatchValue(value=doc_id))]
        )
    )

def uuid_v4_from_doc_chunk(doc_id, chunk_idx):
    import uuid
    # Deterministic UUID for chunks
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{chunk_idx}")
