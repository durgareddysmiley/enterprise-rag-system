from celery import Celery
from .config import settings
import asyncio
from .ingestion import process_document

celery_app = Celery(
    "worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

@celery_app.task(name="process_document_task")
def process_document_task(doc_id: str, file_path: str):
    # Run the async process_document function in a synchronous celery task
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(process_document(doc_id, file_path))
