#!/usr/bin/env python3
"""
Free Embeddings Demo for Book Recommender
Uses Hugging Face API or local sentence-transformers
"""

import os
import requests
import numpy as np
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HuggingFaceEmbeddings:
    """Free embeddings using Hugging Face Inference API"""
    
    def __init__(self, api_token=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        
        if not self.api_token:
            print("Warning: No Hugging Face API token found.")
            print("You can get a free token from: https://huggingface.co/settings/tokens")
            print("Or use local embeddings instead.")
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text using Hugging Face API"""
        if not self.api_token:
            raise ValueError("Hugging Face API token is required")
        
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json={"inputs": text, "options": {"wait_for_model": True}}
            )
            response.raise_for_status()
            embedding = response.json()
            
            # Convert to list of floats
            if isinstance(embedding, list) and len(embedding) > 0:
                return embedding[0]
            else:
                return embedding
        except Exception as e:
            print(f"Error embedding text: {e}")
            # Return zero vector as fallback
            return [0.0] * 384  # Default size for all-MiniLM-L6-v2

class LocalEmbeddings:
    """Completely free local embeddings using sentence-transformers"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            print(f"Using local model: {model_name}")
        except ImportError:
            print("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise

    def embed_text(self, text: str) -> List[float]:
        """Embed text using local model"""
        return self.model.encode([text]).tolist()[0]
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def demo_embeddings():
    """Demo the free embedding functionality"""
    
    # Sample book descriptions
    books = [
        "A thrilling sci-fi adventure about space exploration and alien encounters",
        "A romantic love story set in Paris during the 1920s",
        "A mystery novel about a detective solving crimes in a small town",
        "A fantasy epic with dragons, magic, and epic battles",
        "A historical fiction novel about World War II"
    ]
    
    print("=== Free Embeddings Demo ===\n")
    
    # Try Hugging Face API first (if token available)
    try:
        print("1. Testing Hugging Face API embeddings...")
        hf_embeddings = HuggingFaceEmbeddings()
        
        # Test embedding
        test_text = "sci-fi space adventure"
        embedding = hf_embeddings.embed_text(test_text)
        print(f"✓ Hugging Face API working! Embedding size: {len(embedding)}")
        
        # Find most similar book
        query_embedding = hf_embeddings.embed_text(test_text)
        similarities = []
        
        for i, book in enumerate(books):
            book_embedding = hf_embeddings.embed_text(book)
            similarity = cosine_similarity(query_embedding, book_embedding)
            similarities.append((similarity, i, book))
        
        similarities.sort(reverse=True)
        print(f"\nTop recommendation for '{test_text}':")
        print(f"  {similarities[0][2]}")
        print(f"  Similarity: {similarities[0][0]:.3f}")
        
    except Exception as e:
        print(f"✗ Hugging Face API failed: {e}")
        print("Falling back to local embeddings...\n")
    
    # Try local embeddings
    try:
        print("2. Testing local embeddings...")
        local_embeddings = LocalEmbeddings()
        
        # Test embedding
        test_text = "romance love story"
        embedding = local_embeddings.embed_text(test_text)
        print(f"✓ Local embeddings working! Embedding size: {len(embedding)}")
        
        # Find most similar book
        query_embedding = local_embeddings.embed_text(test_text)
        similarities = []
        
        for i, book in enumerate(books):
            book_embedding = local_embeddings.embed_text(book)
            similarity = cosine_similarity(query_embedding, book_embedding)
            similarities.append((similarity, i, book))
        
        similarities.sort(reverse=True)
        print(f"\nTop recommendation for '{test_text}':")
        print(f"  {similarities[0][2]}")
        print(f"  Similarity: {similarities[0][0]:.3f}")
        
    except Exception as e:
        print(f"✗ Local embeddings failed: {e}")
        print("Please install sentence-transformers: pip install sentence-transformers")

if __name__ == "__main__":
    demo_embeddings() 