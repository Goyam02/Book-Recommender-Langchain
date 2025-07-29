#!/usr/bin/env python3
"""
Script to update the existing vector-search.ipynb with free embeddings
"""

import json
import os

def create_free_embeddings_cell():
    """Create the free embeddings cell content"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Free Embeddings Implementation\n",
            "import os\n",
            "import requests\n",
            "import numpy as np\n",
            "from typing import List\n",
            "from dotenv import load_dotenv\n",
            "\n",
            "load_dotenv()\n",
            "\n",
            "class LocalEmbeddings:\n",
            "    \"\"\"Completely free local embeddings using sentence-transformers\"\"\"\n",
            "    \n",
            "    def __init__(self, model_name=\"all-MiniLM-L6-v2\"):\n",
            "        try:\n",
            "            from sentence_transformers import SentenceTransformer\n",
            "            self.model = SentenceTransformer(model_name)\n",
            "            print(f\"Using local model: {model_name}\")\n",
            "        except ImportError:\n",
            "            print(\"sentence-transformers not installed. Run: pip install sentence-transformers\")\n",
            "            raise\n",
            "    \n",
            "    def embed_documents(self, texts: List[str]) -> List[List[float]]:\n",
            "        \"\"\"Embed documents using local model\"\"\"\n",
            "        return self.model.encode(texts).tolist()\n",
            "    \n",
            "    def embed_query(self, text: str) -> List[float]:\n",
            "        \"\"\"Embed a single query text\"\"\"\n",
            "        return self.model.encode([text]).tolist()[0]\n",
            "\n",
            "# Initialize free embeddings\n",
            "embeddings = LocalEmbeddings()\n",
            "print(\"Free embeddings initialized successfully!\")"
        ]
    }

def update_notebook():
    """Update the existing notebook with free embeddings"""
    
    # Read the existing notebook
    try:
        with open('vector-search.ipynb', 'r') as f:
            notebook = json.load(f)
    except FileNotFoundError:
        print("vector-search.ipynb not found. Creating a new one...")
        notebook = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.5"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
    
    # Find and replace OpenAI embeddings with free embeddings
    updated_cells = []
    
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # Replace OpenAI imports
            if "from langchain_openai import OpenAIEmbeddings" in source:
                # Skip this cell (we'll add our own)
                continue
            elif "OpenAIEmbeddings()" in source:
                # Replace with free embeddings
                new_source = source.replace("OpenAIEmbeddings()", "embeddings")
                cell["source"] = [new_source]
                updated_cells.append(cell)
            else:
                updated_cells.append(cell)
        else:
            updated_cells.append(cell)
    
    # Add free embeddings cell at the beginning
    free_embeddings_cell = create_free_embeddings_cell()
    updated_cells.insert(1, free_embeddings_cell)  # Insert after imports
    
    notebook["cells"] = updated_cells
    
    # Save the updated notebook
    with open('vector-search-free.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ Updated notebook saved as 'vector-search-free.ipynb'")
    print("📝 Changes made:")
    print("   - Removed OpenAI embeddings dependency")
    print("   - Added free local embeddings")
    print("   - Updated embedding initialization")
    print("\n🚀 You can now run the notebook with free embeddings!")

if __name__ == "__main__":
    update_notebook() 