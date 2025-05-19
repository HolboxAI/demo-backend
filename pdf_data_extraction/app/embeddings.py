# import uuid
# from typing import List, Dict

# # Dummy in-memory vector DB for demo
# VECTOR_DB = {}

# def embed_text(text: str) -> List[float]:
#     """
#     Replace with real embedding API call.
#     Here returns dummy vector for demo.
#     """
#     return [float(ord(c)) for c in text[:10]]  # Dummy example

# def store_embeddings(pdf_id: str, chunks: List[str]) -> List[str]:
#     """
#     Store embeddings with metadata in VECTOR_DB.
#     Returns list of chunk IDs.
#     """
#     chunk_ids = []
#     for idx, chunk in enumerate(chunks):
#         chunk_id = str(uuid.uuid4())
#         VECTOR_DB[chunk_id] = {
#             "embedding": embed_text(chunk),
#             "metadata": {
#                 "pdf_id": pdf_id,
#                 "chunk_index": idx,
#                 "text": chunk,
#             },
#         }
#         chunk_ids.append(chunk_id)
#     return chunk_ids

# def query_embeddings(pdf_id: str, query_text: str, top_k=3) -> List[Dict]:
#     """
#     Simple dummy similarity: returns chunks from VECTOR_DB filtered by pdf_id.
#     Replace with actual similarity search.
#     """
#     results = []
#     for chunk_id, record in VECTOR_DB.items():
#         if record["metadata"]["pdf_id"] == pdf_id:
#             results.append(record)
#             if len(results) >= top_k:
#                 break
#     return results

# def delete_vectors_by_pdf_id(pdf_id: str):
#     to_delete = [cid for cid, rec in VECTOR_DB.items() if rec["metadata"]["pdf_id"] == pdf_id]
#     for cid in to_delete:
#         VECTOR_DB.pop(cid)


import os
from dotenv import load_dotenv
import openai
import numpy as np
import uuid
import faiss
from typing import List, Dict

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# Dimension of embedding (OpenAI's text-embedding-ada-002 outputs 1536-dimensional vectors)
EMBED_DIM = 1536

# FAISS index and metadata store
index = faiss.IndexFlatIP(EMBED_DIM)  # Using Inner Product for cosine similarity (after normalization)
metadata_store = {}  # Map vector ID (int) to metadata dict

# Keep track of vector IDs assigned
vector_id_counter = 0


def embed_text(text: str) -> np.ndarray:
    """
    Generate embedding using new OpenAI API method.
    """
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    vector = response.data[0].embedding
    vector = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def store_embeddings(pdf_id: str, chunks: List[str]) -> List[str]:
    """
    Embed chunks, store in FAISS index and metadata_store.
    Returns list of chunk UUIDs.
    """
    global vector_id_counter
    chunk_ids = []
    vectors = []
    ids = []

    for idx, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        vector = embed_text(chunk)
        vectors.append(vector)
        ids.append(vector_id_counter)

        metadata_store[vector_id_counter] = {
            "chunk_id": chunk_id,
            "pdf_id": pdf_id,
            "chunk_index": idx,
            "text": chunk,
        }
        chunk_ids.append(chunk_id)
        vector_id_counter += 1

    vectors_np = np.vstack(vectors)
    index.add(vectors_np)  # Add batch vectors to FAISS index

    return chunk_ids

def query_embeddings(pdf_id: str, query_text: str, top_k=3) -> List[Dict]:
    """
    Query top_k most similar chunks from the given PDF using FAISS.
    Returns list of metadata dicts.
    """
    if index.ntotal == 0:
        return []

    query_vector = embed_text(query_text).reshape(1, -1)

    # Search for top_k vectors
    D, I = index.search(query_vector, top_k * 5)  # Search more to filter by pdf_id

    results = []
    for idx in I[0]:
        if idx == -1:
            continue
        meta = metadata_store.get(idx)
        if meta and meta['pdf_id'] == pdf_id:
            results.append(meta)
            if len(results) >= top_k:
                break

    return results


def generate_answer(question: str, context_chunks: List[str]) -> str:
    system_prompt = "You are an AI assistant helping to answer questions based on the provided document excerpts."
    context_text = "\n\n---\n\n".join(context_chunks)
    user_prompt = (
        f"Use the following excerpts from a document to answer the question.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    response = openai.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=500,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def delete_vectors_by_pdf_id(pdf_id: str):
    """
    Deletes all vectors & metadata related to the given pdf_id.
    Note: FAISS does not support deletion in IndexFlatIP, so
    we'll recreate index excluding those vectors.
    """
    global index, metadata_store, vector_id_counter

    # Filter vectors to keep (exclude pdf_id)
    keep_ids = [vid for vid, meta in metadata_store.items() if meta['pdf_id'] != pdf_id]
    new_metadata_store = {vid: metadata_store[vid] for vid in keep_ids}

    # Extract corresponding vectors
    vectors_to_keep = []
    id_mapping = {}
    for new_id, old_id in enumerate(keep_ids):
        id_mapping[old_id] = new_id
        vectors_to_keep.append(index.reconstruct(old_id))

    # Create new index
    index = faiss.IndexFlatIP(EMBED_DIM)
    if vectors_to_keep:
        index.add(np.vstack(vectors_to_keep).astype(np.float32))

    metadata_store = new_metadata_store
    vector_id_counter = len(metadata_store)
