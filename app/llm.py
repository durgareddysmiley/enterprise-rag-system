from openai import AsyncOpenAI
from .config import settings
import json

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def generate_answer(query: str, contexts: list):
    """
    Generates an answer based on the provided context with citations.
    """
    if not settings.OPENAI_API_KEY:
        return {
            "answer": "OpenAI API Key not configured. Here is the context retrieved:",
            "sources": contexts
        }

    context_text = ""
    for i, ctx in enumerate(contexts):
        context_text += f"[Source ID: {i}] Document: {ctx['filename']}\nContent: {ctx['text']}\n\n"

    prompt = f"""
    You are an enterprise document search assistant. Use the provided context to answer the user's question.
    Strictly follow these rules:
    1. Only use the provided context to answer.
    2. If the answer is not in the context, say "I don't have enough information to answer this question."
    3. Cite your sources using the format [Source ID: N] for every claim you make.
    
    Context:
    {context_text}
    
    User Question: {query}
    
    Answer (with citations):
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "sources": contexts
        }
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": contexts
        }
