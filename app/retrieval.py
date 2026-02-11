from .config import settings
from .ingestion import qdrant_client, embedding_model
from qdrant_client.http import models as rest
import cohere
from rank_bm25 import BM25Okapi
from .llm import generate_answer
from .database import AsyncSessionLocal
from .models import Document
from sqlalchemy import select

cohere_client = cohere.Client(settings.COHERE_API_KEY) if settings.COHERE_API_KEY else None

async def perform_rag_query(query_text: str):
    # 1. Semantic Search (Vector)
    query_vector = embedding_model.encode(query_text).tolist()
    vector_results = qdrant_client.search(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=20
    )
    
    # 2. Keyword Search (BM25)
    # Fetch random candidates or use vector results as base
    # To be truly hybrid, we'd ideally search a keyword index.
    # Here we'll take top 50 from vector and re-rank with BM25 + semantic score.
    
    candidate_results = qdrant_client.search(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=50
    )
    
    # 2. Keyword Search (BM25 logic on the candidates)
    # Since Qdrant has a full-text index, we could also use a separate filter-based search here
    # to find exact keyword matches that might have been missed by vector search.
    keyword_results = qdrant_client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter=rest.Filter(
            must=[rest.FieldCondition(key="text", match=rest.MatchText(text=query_text))]
        ),
        limit=20
    )[0]
    
    # Merge candidates
    seen_ids = set()
    all_candidates = []
    
    # Process vector candidates
    for r in candidate_results:
        if r.id not in seen_ids:
            all_candidates.append({
                "id": r.id,
                "text": r.payload["text"],
                "document_id": r.payload["document_id"],
                "vector_score": r.score,
                "filename": r.payload["filename"]
            })
            seen_ids.add(r.id)
            
    # Process keyword candidates (if any)
    for r in keyword_results:
        if r.id not in seen_ids:
            all_candidates.append({
                "id": r.id,
                "text": r.payload["text"],
                "document_id": r.payload["document_id"],
                "vector_score": 0.0, # Will be boosted by BM25
                "filename": r.payload["filename"]
            })
            seen_ids.add(r.id)

    if not all_candidates:
        return {"answer": "No relevant documents found.", "sources": []}

    # Apply BM25 to the merged candidates
    corpus = [c["text"] for c in all_candidates]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query_text.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Merge scores (Simple Reciprocal Rank Fusion or weighted sum)
    for i, candidate in enumerate(all_candidates):
        # Normalize BM25 score roughly (0-1 range)
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1
        norm_bm25 = bm25_scores[i] / max_bm25 if max_bm25 > 0 else 0
        candidate["hybrid_score"] = (candidate["vector_score"] * 0.7) + (norm_bm25 * 0.3)

    # Sort by hybrid score
    all_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    # 3. Re-ranking (Cohere)
    if cohere_client and all_candidates:
        texts = [c["text"] for c in all_candidates[:20]] # Rerank top 20 hybrid
        rerank_results = cohere_client.rerank(
            query=query_text,
            documents=texts,
            top_n=5,
            model="rerank-english-v2.0"
        )
        top_chunks = []
        for r in rerank_results:
            top_chunks.append(all_candidates[r.index])
    else:
        top_chunks = all_candidates[:5]

    # 4. Generate Answer
    answer_data = await generate_answer(query_text, top_chunks)
    
    return answer_data
