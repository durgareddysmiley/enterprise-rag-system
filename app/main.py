from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
import os
import shutil

from .database import get_db, engine, Base
from .models import Document, DocumentStatus
from .config import settings
# from .worker import process_document_task # We'll implement this soon

app = FastAPI(title=settings.APP_NAME)

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    doc_id = str(uuid.uuid4())
    file_path = f"uploads/{doc_id}_{file.filename}"
    
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    new_doc = Document(
        id=doc_id,
        filename=file.filename,
        status=DocumentStatus.PENDING,
        metadata_json={"path": file_path}
    )
    db.add(new_doc)
    await db.commit()
    
    # Trigger background task using Celery
    from .worker import process_document_task
    process_document_task.delay(doc_id, file_path)
    
    return {"document_id": doc_id}

@app.get("/api/documents/{document_id}")
async def get_document_status(document_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc.to_dict()

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete from Qdrant (To be implemented in retrieval/ingestion)
    from .ingestion import delete_document_vectors
    await delete_document_vectors(document_id)
    
    # Delete file
    if doc.metadata_json.get("path") and os.path.exists(doc.metadata_json["path"]):
        os.remove(doc.metadata_json["path"])
        
    await db.delete(doc)
    await db.commit()
    return {"message": "Document deleted successfully"}

@app.post("/api/query")
async def query_rag(query_data: dict, db: AsyncSession = Depends(get_db)):
    query_text = query_data.get("query")
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required")
        
    from .retrieval import perform_rag_query
    answer = await perform_rag_query(query_text)
    return answer
