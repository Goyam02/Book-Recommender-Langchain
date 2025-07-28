# Free Embeddings for Book Recommender

This project provides **completely free** alternatives to OpenAI embeddings for your book recommender system.

## 🆓 Free Options Available

### 1. **Hugging Face Inference API** (Recommended)
- **Free tier**: 30,000 requests/month
- **Quality**: Excellent
- **Setup**: Requires free API token

### 2. **Local Embeddings** (Completely Free)
- **Cost**: $0 (runs on your machine)
- **Quality**: Very good
- **Setup**: Just install sentence-transformers

## 🚀 Quick Start

### Option 1: Hugging Face API (Recommended)

1. **Get a free API token**:
   - Go to https://huggingface.co/settings/tokens
   - Create a free account
   - Generate a new token

2. **Set up your environment**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Create .env file
   echo "HUGGINGFACE_API_TOKEN=your_token_here" > .env
   ```

3. **Run the demo**:
   ```bash
   python free_embeddings_demo.py
   ```

### Option 2: Local Embeddings (No API Needed)

1. **Install dependencies**:
   ```bash
   pip install sentence-transformers langchain-community langchain-text-splitters langchain-chroma pandas
   ```

2. **Run the demo**:
   ```bash
   python free_embeddings_demo.py
   ```

## 📁 Files Created

- `vector-search-free.ipynb` - Updated notebook with free embeddings
- `free_embeddings_demo.py` - Standalone demo script
- `requirements.txt` - All necessary dependencies
- `README_free_embeddings.md` - This file

## 🔄 Migration from OpenAI

To replace your current OpenAI embeddings:

1. **Replace the import**:
   ```python
   # Old (OpenAI)
   from langchain_openai import OpenAIEmbeddings
   
   # New (Free)
   from free_embeddings_demo import LocalEmbeddings  # or HuggingFaceEmbeddings
   ```

2. **Update the embedding initialization**:
   ```python
   # Old
   embeddings = OpenAIEmbeddings()
   
   # New
   embeddings = LocalEmbeddings()  # or HuggingFaceEmbeddings()
   ```

3. **Everything else stays the same!**

## 💰 Cost Comparison

| Method | Cost | Requests/Month | Quality |
|--------|------|----------------|---------|
| OpenAI | $0.0001/1K tokens | Unlimited | Excellent |
| Hugging Face | $0 | 30,000 | Excellent |
| Local | $0 | Unlimited | Very Good |

## 🎯 Usage Examples

### Basic Usage
```python
from free_embeddings_demo import LocalEmbeddings

# Initialize embeddings
embeddings = LocalEmbeddings()

# Embed a text
text = "A sci-fi adventure about space exploration"
embedding = embeddings.embed_text(text)
print(f"Embedding size: {len(embedding)}")
```

### With LangChain
```python
from langchain_chroma import Chroma
from free_embeddings_demo import LocalEmbeddings

# Create vector database
embeddings = LocalEmbeddings()
db = Chroma.from_documents(documents, embedding=embeddings)

# Search
results = db.similarity_search("mystery novel", k=3)
```

## 🔧 Troubleshooting

### Hugging Face API Issues
- Make sure your API token is correct
- Check your monthly usage limit
- Try the local embeddings as fallback

### Local Embeddings Issues
- Install sentence-transformers: `pip install sentence-transformers`
- First run will download the model (~80MB)
- Requires ~2GB RAM for the model

### Performance Tips
- Local embeddings are faster for repeated queries
- Hugging Face API is better for one-off requests
- Both work great for book recommendations!

## 📊 Model Comparison

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| all-MiniLM-L6-v2 | 80MB | Fast | Good | General purpose |
| all-mpnet-base-v2 | 420MB | Medium | Excellent | High quality needed |
| paraphrase-MiniLM-L6-v2 | 80MB | Fast | Good | Semantic similarity |

## 🎉 Benefits

✅ **Completely free** - No API costs  
✅ **High quality** - Excellent for book recommendations  
✅ **Easy migration** - Drop-in replacement for OpenAI  
✅ **Flexible** - Choose API or local  
✅ **Reliable** - Production-ready  

Start using free embeddings today and save money while building great book recommendations! 